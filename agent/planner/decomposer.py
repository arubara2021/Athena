from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent.planner.templates import GoalType, PlanTemplates, TemplateStep
from core.models import AgentActionType, CoreModel
from core.schemas import LLMMessageSchema, LLMRequestSchema
from utils.logger import get_logger, get_trace_logger
from utils.text import clean_text


class SubTask(CoreModel):
    description: str
    action_type: AgentActionType = AgentActionType.NO_OP
    priority: int = Field(default=0, ge=0)
    estimated_tokens: int = Field(default=3000, ge=0)
    tool_hint: str = ""
    depends_on: list[int] = Field(default_factory=list)


class DecompositionResult(CoreModel):
    goal: str
    goal_type: GoalType
    sub_tasks: list[SubTask] = Field(default_factory=list)
    llm_used: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class GoalDecomposer:
    _DEFAULT_MAX_SUB_TASKS = 8
    _DEFAULT_MIN_SUB_TASKS = 2
    _DEFAULT_MAX_TOKENS_PER_TASK = 8000
    _DEFAULT_MIN_TOKENS_PER_TASK = 1000

    def __init__(
        self,
        use_llm: bool = True,
        max_sub_tasks: int | None = None,
        min_sub_tasks: int | None = None,
        templates: PlanTemplates | None = None,
    ) -> None:
        self._use_llm = use_llm
        self._max_sub_tasks = max(
            self._DEFAULT_MIN_SUB_TASKS,
            max_sub_tasks or self._DEFAULT_MAX_SUB_TASKS,
        )
        self._min_sub_tasks = max(
            1,
            min_sub_tasks or self._DEFAULT_MIN_SUB_TASKS,
        )
        self._templates = templates or PlanTemplates()
        self._logger = get_logger("agent.planner.decomposer")
        self._trace = get_trace_logger()

    async def decompose(
        self,
        goal: str,
        level: str = "",
        intent: str = "",
    ) -> DecompositionResult:
        cleaned_goal = clean_text(goal)
        if not cleaned_goal:
            return DecompositionResult(
                goal=goal,
                goal_type=GoalType.EXPLORE,
                sub_tasks=[],
                llm_used=False,
                confidence=0.0,
            )

        goal_type = self._templates.detect_goal_type(cleaned_goal, intent)

        self._trace.emit(
            "planner_goal_detected",
            goal=cleaned_goal,
            goal_type=goal_type.value,
            intent=intent,
        )

        if self._use_llm:
            llm_result = await self._llm_decompose(
                cleaned_goal, goal_type, level
            )
            if llm_result is not None and len(llm_result.sub_tasks) >= self._min_sub_tasks:
                self._trace.emit(
                    "planner_decomposition_llm",
                    goal=cleaned_goal,
                    sub_task_count=len(llm_result.sub_tasks),
                    goal_type=goal_type.value,
                )
                return llm_result

        template_result = self._template_decompose(cleaned_goal, goal_type)
        self._trace.emit(
            "planner_decomposition_template",
            goal=cleaned_goal,
            sub_task_count=len(template_result.sub_tasks),
            goal_type=goal_type.value,
        )
        return template_result

    def _template_decompose(
        self,
        goal: str,
        goal_type: GoalType,
    ) -> DecompositionResult:
        template_steps = self._templates.get_template_steps(goal_type)
        sub_tasks: list[SubTask] = []

        for index, step in enumerate(template_steps):
            sub_tasks.append(
                SubTask(
                    description=step.description,
                    action_type=step.action_type,
                    priority=step.priority,
                    estimated_tokens=step.estimated_tokens,
                    tool_hint=step.tool_hint,
                    depends_on=list(range(index)) if index > 0 else [],
                )
            )

        return DecompositionResult(
            goal=goal,
            goal_type=goal_type,
            sub_tasks=sub_tasks[: self._max_sub_tasks],
            llm_used=False,
            confidence=0.6,
        )

    async def _llm_decompose(
        self,
        goal: str,
        goal_type: GoalType,
        level: str,
    ) -> DecompositionResult | None:
        try:
            from llm.guardrails import validate_llm_request
            from llm.parser import parse_json_object_response
            from llm.provider import LLMProviderManager
            from llm.router import get_fast_model_references

            model_references = get_fast_model_references(limit=2)
            if not model_references:
                return None

            expected_output = {
                "sub_tasks": [
                    {
                        "description": "string",
                        "action_type": "search | read | analyze | generate | reflect | plan | synthesize | memory_store | memory_recall | no_op",
                        "priority": 1,
                        "estimated_tokens": 3000,
                        "tool_hint": "string",
                    }
                ],
                "confidence": 0.8,
            }

            system_prompt = (
                "You are a task decomposition engine for an autonomous research agent. "
                "Break down the user's goal into ordered, actionable sub-tasks. "
                "Each sub-task should be specific and achievable. "
                "Return only valid JSON."
            )

            user_prompt = (
                f"Goal: {goal}\n"
                f"Goal type: {goal_type.value}\n"
                f"Level: {level}\n"
                f"Maximum sub-tasks: {self._max_sub_tasks}\n"
                f"Minimum sub-tasks: {self._min_sub_tasks}\n\n"
                "Rules:\n"
                f"- Create between {self._min_sub_tasks} and {self._max_sub_tasks} sub-tasks.\n"
                "- Order sub-tasks logically from first to last.\n"
                "- Each sub-task must be specific and actionable.\n"
                "- action_type must be one of: search, read, analyze, generate, reflect, plan, synthesize, memory_store, memory_recall, no_op.\n"
                "- priority starts at 1 for the first task and increments.\n"
                f"- estimated_tokens should be between {self._DEFAULT_MIN_TOKENS_PER_TASK} and {self._DEFAULT_MAX_TOKENS_PER_TASK}.\n"
                "- tool_hint should suggest which tool to use (search_academic, search_web, search_github, search_wikipedia, read_url, rank_sources, generate_learning_path, generate_report, compare_sources, or empty string).\n"
                "- Do not invent facts.\n"
                f"\nExpected JSON output:\n{json.dumps(expected_output, ensure_ascii=False, indent=2)}\n"
                "Return only valid JSON."
            )

            messages = [
                LLMMessageSchema(role="system", content=system_prompt),
                LLMMessageSchema(role="user", content=user_prompt),
            ]

            async with LLMProviderManager() as manager:
                for model_reference in model_references:
                    try:
                        request = LLMRequestSchema(
                            provider=model_reference.provider,
                            model=model_reference.model_id,
                            messages=messages,
                            temperature=0.0,
                            max_tokens=2000,
                            response_format="json_object",
                        )
                        request = validate_llm_request(request)
                        response = await manager.complete(request)
                        if response is None:
                            continue
                        payload = parse_json_object_response(response.content)
                        return self._parse_llm_payload(goal, goal_type, payload)
                    except Exception:
                        continue

            return None
        except Exception as exc:
            self._logger.warning(f"LLM decomposition failed: {exc}")
            return None

    def _parse_llm_payload(
        self,
        goal: str,
        goal_type: GoalType,
        payload: dict[str, Any],
    ) -> DecompositionResult | None:
        raw_tasks = payload.get("sub_tasks", [])
        if not isinstance(raw_tasks, list) or not raw_tasks:
            return None

        sub_tasks: list[SubTask] = []
        for index, item in enumerate(raw_tasks[: self._max_sub_tasks]):
            if not isinstance(item, dict):
                continue

            description = clean_text(item.get("description", ""))
            if not description:
                continue

            action_type = self._parse_action_type(item.get("action_type"))
            priority = self._parse_priority(item.get("priority"), index)
            estimated_tokens = self._parse_estimated_tokens(item.get("estimated_tokens"))
            tool_hint = clean_text(item.get("tool_hint", ""))

            sub_tasks.append(
                SubTask(
                    description=description,
                    action_type=action_type,
                    priority=priority,
                    estimated_tokens=estimated_tokens,
                    tool_hint=tool_hint,
                    depends_on=list(range(index)) if index > 0 else [],
                )
            )

        if len(sub_tasks) < self._min_sub_tasks:
            return None

        confidence = self._parse_confidence(payload.get("confidence"))

        return DecompositionResult(
            goal=goal,
            goal_type=goal_type,
            sub_tasks=sub_tasks,
            llm_used=True,
            confidence=confidence,
        )

    def _parse_action_type(self, value: Any) -> AgentActionType:
        try:
            return AgentActionType(str(value).strip().lower())
        except (ValueError, AttributeError):
            return AgentActionType.NO_OP

    def _parse_priority(self, value: Any, index: int) -> int:
        try:
            return max(0, int(value))
        except (ValueError, TypeError):
            return index + 1

    def _parse_estimated_tokens(self, value: Any) -> int:
        try:
            tokens = int(value)
            return max(
                self._DEFAULT_MIN_TOKENS_PER_TASK,
                min(self._DEFAULT_MAX_TOKENS_PER_TASK, tokens),
            )
        except (ValueError, TypeError):
            return 3000

    def _parse_confidence(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (ValueError, TypeError):
            return 0.5