from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from core.models import AgentActionType, CoreModel
from core.schemas import LLMMessageSchema, LLMRequestSchema
from utils.logger import get_logger
from utils.text import clean_text

DEFAULT_THINK_TEMPERATURE = 0.0
DEFAULT_THINK_MAX_TOKENS = 1500
DEFAULT_ACTION_ADVANCE = "advance_step"
DEFAULT_ACTION_COMPLETE = "complete_goal"


class ThinkResult(CoreModel):
    action: str = ""
    tool_name: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    is_advance: bool = False
    is_complete: bool = False
    llm_used: bool = False
    error: str | None = None


class ThinkPhase:
    def __init__(
        self,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model_limit: int = 2,
    ) -> None:
        self._temperature = (
            temperature if temperature is not None else DEFAULT_THINK_TEMPERATURE
        )
        self._max_tokens = max_tokens or DEFAULT_THINK_MAX_TOKENS
        self._model_limit = max(1, model_limit)
        self._logger = get_logger("agent.loop.think")

    async def think(self, state: Any, tools_summary: list[dict[str, Any]]) -> ThinkResult:
        current_step = getattr(state, "current_step", None)
        if current_step is None:
            return ThinkResult(
                action=DEFAULT_ACTION_COMPLETE,
                is_complete=True,
                reasoning="No plan steps remaining",
            )

        if state.is_budget_critical:
            return ThinkResult(
                action=DEFAULT_ACTION_COMPLETE,
                is_complete=True,
                reasoning="Budget is critically low, completing to preserve results",
            )

        if state.actions_in_current_step >= state.max_actions_per_step:
            return ThinkResult(
                action=DEFAULT_ACTION_ADVANCE,
                is_advance=True,
                reasoning="Maximum actions per step reached, advancing to next step",
            )

        llm_result = await self._llm_think(state, current_step, tools_summary)
        if llm_result is not None:
            return llm_result

        return self._heuristic_think(state, current_step, tools_summary)

    async def _llm_think(
        self,
        state: Any,
        current_step: Any,
        tools_summary: list[dict[str, Any]],
    ) -> ThinkResult | None:
        try:
            from llm.guardrails import validate_llm_request
            from llm.parser import parse_json_object_response
            from llm.provider import LLMProviderManager
            from llm.router import get_fast_model_references

            model_references = get_fast_model_references(limit=self._model_limit)
            if not model_references:
                return None

            prompt = self._build_think_prompt(state, current_step, tools_summary)

            async with LLMProviderManager() as manager:
                for model_reference in model_references:
                    try:
                        request = LLMRequestSchema(
                            provider=model_reference.provider,
                            model=model_reference.model_id,
                            messages=[
                                LLMMessageSchema(
                                    role="system",
                                    content=self._system_prompt(),
                                ),
                                LLMMessageSchema(role="user", content=prompt),
                            ],
                            temperature=self._temperature,
                            max_tokens=self._max_tokens,
                            response_format="json_object",
                        )
                        request = validate_llm_request(request)
                        response = await manager.complete(request)
                        if response is None:
                            continue

                        tokens_used = response.tokens_used or 0
                        state.consume_tokens(tokens_used)

                        payload = parse_json_object_response(response.content)
                        return self._parse_think_payload(payload)
                    except Exception:
                        continue
            return None
        except Exception as exc:
            self._logger.warning(f"LLM think failed: {exc}")
            return None

    def _heuristic_think(
        self,
        state: Any,
        current_step: Any,
        tools_summary: list[dict[str, Any]],
    ) -> ThinkResult:
        action_type = self._enum_value(
            getattr(current_step, "action_type", AgentActionType.NO_OP)
        )
        tool_hint = clean_text(getattr(current_step, "tool_hint", ""))
        step_description = clean_text(getattr(current_step, "description", ""))

        selected_tool = self._select_tool_for_action(
            action_type, tool_hint, tools_summary
        )

        if selected_tool is None:
            return ThinkResult(
                action=DEFAULT_ACTION_ADVANCE,
                is_advance=True,
                reasoning=f"No suitable tool found for action type: {action_type}",
            )

        parameters = self._build_heuristic_parameters(
            selected_tool, step_description, state
        )

        return ThinkResult(
            action=selected_tool,
            tool_name=selected_tool,
            parameters=parameters,
            reasoning=f"Heuristic selection: {selected_tool} for {action_type}",
            llm_used=False,
        )

    def _select_tool_for_action(
        self,
        action_type: str,
        tool_hint: str,
        tools_summary: list[dict[str, Any]],
    ) -> str | None:
        if tool_hint:
            for tool in tools_summary:
                if tool.get("name") == tool_hint:
                    return tool_hint

        action_tool_map = {
            AgentActionType.SEARCH.value: [
                "search_academic",
                "search_web",
                "search_github",
                "search_wikipedia",
            ],
            AgentActionType.READ.value: ["read_url", "read_paper_abstract"],
            AgentActionType.ANALYZE.value: [
                "rank_sources",
                "classify_difficulty",
                "compare_sources",
            ],
            AgentActionType.GENERATE.value: [
                "generate_learning_path",
                "generate_report",
                "summarize_source",
            ],
            AgentActionType.MEMORY_STORE.value: ["save_to_memory"],
            AgentActionType.MEMORY_RECALL.value: ["recall_memory", "search_memory"],
        }

        candidate_tools = action_tool_map.get(action_type, [])
        available_names = {t.get("name") for t in tools_summary}

        for candidate in candidate_tools:
            if candidate in available_names:
                return candidate

        if tools_summary:
            return tools_summary[0].get("name")

        return None

    def _build_heuristic_parameters(
        self,
        tool_name: str,
        step_description: str,
        state: Any,
    ) -> dict[str, Any]:
        query = step_description or state.goal or state.topic

        if tool_name.startswith("search"):
            return {"query": query, "max_results": 8}

        if tool_name == "rank_sources":
            return {
                "sources_data": [
                    {
                        "source_id": s.source_id,
                        "title": s.title,
                        "url": s.url,
                        "platform": self._enum_value(s.platform),
                        "source_type": self._enum_value(s.source_type),
                        "abstract": (s.abstract or "")[:500],
                    }
                    for s in state.findings[:20]
                ],
                "query": state.goal,
                "level": state.level,
            }

        if tool_name == "generate_learning_path":
            return {
                "sources_data": [
                    {
                        "source_id": s.source.source_id,
                        "title": s.source.title,
                        "url": s.source.url,
                        "platform": self._enum_value(s.source.platform),
                        "source_type": self._enum_value(s.source.source_type),
                        "difficulty": self._enum_value(s.source.difficulty)
                        if s.source.difficulty
                        else None,
                        "rank": s.rank,
                        "score": s.score,
                    }
                    for s in state.ranked_sources[:15]
                ],
                "query": state.goal,
                "level": state.level,
            }

        if tool_name == "read_url":
            urls = [s.url for s in state.findings if s.url]
            return {"url": urls[0] if urls else ""}

        if tool_name in ("save_to_memory", "recall_memory", "search_memory"):
            return {"query": state.goal, "key": state.topic}

        return {"query": query}

    def _build_think_prompt(
        self,
        state: Any,
        current_step: Any,
        tools_summary: list[dict[str, Any]],
    ) -> str:
        step_index = state.current_step_index + 1
        total_steps = state.total_steps
        step_description = clean_text(getattr(current_step, "description", ""))
        step_action = self._enum_value(
            getattr(current_step, "action_type", AgentActionType.NO_OP)
        )
        tool_hint = clean_text(getattr(current_step, "tool_hint", ""))
        tools_text = json.dumps(tools_summary, ensure_ascii=False, indent=2)

        recent_actions = state.actions_taken[-3:] if state.actions_taken else []
        actions_text = "\n".join(
            f"- Step {a.step_index + 1}: {a.action_type.value} via {a.tool_call.tool_name if a.tool_call else 'none'}"
            for a in recent_actions
        ) or "None yet"

        return (
            f"Goal: {state.goal}\n"
            f"Level: {state.level}\n"
            f"Current Step: {step_index} of {total_steps}\n"
            f"Step Description: {step_description}\n"
            f"Step Action Type: {step_action}\n"
            f"Tool Hint: {tool_hint or 'none'}\n"
            f"Findings So Far: {len(state.findings)} sources\n"
            f"Ranked Sources: {len(state.ranked_sources)}\n"
            f"Budget: {state.tokens_remaining} tokens remaining "
            f"({state.budget_percent_used:.0f}% used)\n"
            f"Recent Actions:\n{actions_text}\n"
            f"Available Tools:\n{tools_text}\n"
            f"Decide the next action for this step.\n"
            f"If the step is already complete, use action '{DEFAULT_ACTION_ADVANCE}'.\n"
            f"If all steps are complete, use action '{DEFAULT_ACTION_COMPLETE}'.\n"
            f"Return JSON with:\n"
            f'- "action": tool name or "{DEFAULT_ACTION_ADVANCE}" or "{DEFAULT_ACTION_COMPLETE}"\n'
            f'- "parameters": object with tool parameters\n'
            f'- "reasoning": brief explanation\n'
            f"Return only valid JSON."
        )

    def _system_prompt(self) -> str:
        return (
            "You are an autonomous research agent decision engine. "
            "You select the best tool to advance the current step of a research plan. "
            "Be efficient with token usage. "
            "Do not invent tools that are not listed. "
            "Return only valid JSON."
        )

    def _parse_think_payload(self, payload: dict[str, Any]) -> ThinkResult:
        action = clean_text(payload.get("action", ""))
        parameters = payload.get("parameters", {})
        reasoning = clean_text(payload.get("reasoning", ""))

        if not isinstance(parameters, dict):
            parameters = {}

        if action == DEFAULT_ACTION_ADVANCE:
            return ThinkResult(
                action=action,
                is_advance=True,
                reasoning=reasoning,
                llm_used=True,
            )

        if action == DEFAULT_ACTION_COMPLETE:
            return ThinkResult(
                action=action,
                is_complete=True,
                reasoning=reasoning,
                llm_used=True,
            )

        return ThinkResult(
            action=action,
            tool_name=action,
            parameters=parameters,
            reasoning=reasoning,
            llm_used=True,
        )

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))