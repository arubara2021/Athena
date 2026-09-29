from __future__ import annotations

from typing import Any

from pydantic import Field

from core.models import AgentActionType, CoreModel
from utils.logger import get_logger
from utils.text import clean_text


DEFAULT_ACTION_ADVANCE = "advance_step"
DEFAULT_ACTION_COMPLETE = "complete_goal"

TOOL_INTENT_MAP = {
    "paper": "search_academic",
    "research": "search_academic",
    "study": "search_academic",
    "journal": "search_academic",
    "publication": "search_academic",
    "book": "search_books",
    "textbook": "search_books",
    "textbooks": "search_books",
    "course": "search_courses",
    "courses": "search_courses",
    "tutorial": "search_courses",
    "tutorials": "search_courses",
    "lecture": "search_courses",
    "video": "search_courses",
    "code": "search_code_models",
    "code_models": "search_code_models",
    "implementation": "search_code_models",
    "model": "search_code_models",
    "repository": "search_code_models",
    "github": "search_code_models",
    "dataset": "search_code_models",
    "basic": "search_explanation",
    "basics": "search_explanation",
    "beginner": "search_explanation",
    "introduction": "search_explanation",
    "explanation": "search_explanation",
    "what_is": "search_explanation",
    "definition": "search_explanation",
    "overview": "search_explanation",
}

INTENT_KEYWORDS = {
    "paper": [
        "paper",
        "research",
        "study",
        "journal",
        "publication",
        "peer-reviewed",
        "academic",
    ],
    "book": ["book", "textbook", "textbooks", "reading", "chapters"],
    "course": [
        "course",
        "courses",
        "tutorial",
        "tutorials",
        "lecture",
        "video",
        "class",
        "lesson",
    ],
    "code": [
        "code",
        "implementation",
        "repository",
        "github",
        "model",
        "dataset",
        "library",
        "package",
    ],
    "basic": [
        "basic",
        "basics",
        "beginner",
        "introduction",
        "what is",
        "definition",
        "overview",
        "explained",
        "explain",
    ],
}

_ACTION_TOOL_MAP = {
    AgentActionType.SEARCH: None,
    AgentActionType.READ: "read_paper_abstract",
    AgentActionType.ANALYZE: "rank_sources",
    AgentActionType.GENERATE: "generate_learning_path",
    AgentActionType.MEMORY_RECALL: "recall_memory",
    AgentActionType.MEMORY_STORE: "save_to_memory",
    AgentActionType.NO_OP: None,
}


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
        self._temperature = 0.0 if temperature is None else float(temperature)
        self._max_tokens = int(max_tokens or 0)
        self._model_limit = max(1, int(model_limit))
        self._logger = get_logger("agent.loop.think")

    def select_search_tool(
        self,
        query: str,
        level: str = "",
        goal: str = "",
    ) -> str:
        combined = f"{query} {level} {goal}".lower()

        if level and "beginner" in level.lower():
            return "search_explanation"

        scores: dict[str, int] = {
            "search_academic": 0,
            "search_books": 0,
            "search_courses": 0,
            "search_code_models": 0,
            "search_explanation": 0,
        }

        for intent, keywords in INTENT_KEYWORDS.items():
            tool = TOOL_INTENT_MAP.get(intent, "")

            if not tool:
                continue

            for keyword in keywords:
                if keyword in combined:
                    scores[tool] += 1

        best_tool = max(scores, key=lambda key: scores[key])

        if scores[best_tool] == 0:
            return "search_academic"

        return best_tool

    async def think(
        self,
        state: Any,
        tools_summary: list[dict[str, Any]],
    ) -> ThinkResult:
        current_step = getattr(state, "current_step", None)

        if current_step is None:
            return ThinkResult(
                action=DEFAULT_ACTION_COMPLETE,
                is_complete=True,
                reasoning="No plan steps remaining",
            )

        if getattr(state, "is_budget_critical", False):
            return ThinkResult(
                action=DEFAULT_ACTION_COMPLETE,
                is_complete=True,
                reasoning="Budget critically low, completing to preserve results",
            )

        actions_in_step = int(
            getattr(state, "actions_in_current_step", 0) or 0
        )
        max_actions = int(
            getattr(state, "max_actions_per_step", 4) or 4
        )

        if actions_in_step >= max_actions:
            return ThinkResult(
                action=DEFAULT_ACTION_ADVANCE,
                is_advance=True,
                reasoning="Maximum actions per step reached, advancing",
            )

        return self._sanitize_tool_choice(
            state,
            self._deterministic_think(state, current_step, tools_summary),
            tools_summary,
        )

    def _deterministic_think(
        self,
        state: Any,
        current_step: Any,
        tools_summary: list[dict[str, Any]],
    ) -> ThinkResult:
        step_action = getattr(
            current_step,
            "action_type",
            AgentActionType.NO_OP,
        )

        tool_hint = clean_text(
            getattr(current_step, "tool_hint", "") or ""
        )

        if tool_hint:
            return ThinkResult(
                action=tool_hint,
                tool_name=tool_hint,
                parameters={},
                reasoning=f"Tool hint from plan step: {tool_hint}",
            )

        if step_action == AgentActionType.SEARCH:
            query = clean_text(
                getattr(current_step, "description", "")
            )
            tool_name = self.select_search_tool(
                query=query,
                level=clean_text(getattr(state, "level", "")),
                goal=clean_text(getattr(state, "goal", "")),
            )
            return ThinkResult(
                action=tool_name,
                tool_name=tool_name,
                parameters={},
                reasoning=f"Search step routed to {tool_name}",
            )

        tool_name = _ACTION_TOOL_MAP.get(step_action)

        if tool_name is None:
            return ThinkResult(
                action=DEFAULT_ACTION_ADVANCE,
                is_advance=True,
                reasoning=f"No tool for action type {step_action}",
            )

        return ThinkResult(
            action=tool_name,
            tool_name=tool_name,
            parameters={},
            reasoning=f"Deterministic routing for {step_action}",
        )

    def _sanitize_tool_choice(
        self,
        state: Any,
        result: ThinkResult,
        tools_summary: list[dict[str, Any]],
    ) -> ThinkResult:
        if result.is_advance or result.is_complete:
            return result

        available = {
            str(tool.get("name", "")) for tool in tools_summary
        }
        tool = clean_text(result.tool_name or result.action)

        if not tool:
            return ThinkResult(
                action=DEFAULT_ACTION_ADVANCE,
                is_advance=True,
                reasoning="No tool selected",
            )

        if tool not in available:
            if tool == "read_url" and "read_paper_abstract" in available:
                tool = "read_paper_abstract"
            elif (
                tool == "compare_sources"
                and "rank_sources" in available
            ):
                counts = self._source_counts(state)

                if counts["source_count"] >= 1:
                    tool = "rank_sources"
                else:
                    return ThinkResult(
                        action=DEFAULT_ACTION_ADVANCE,
                        is_advance=True,
                        reasoning="Insufficient sources for comparison",
                    )
            else:
                return ThinkResult(
                    action=DEFAULT_ACTION_ADVANCE,
                    is_advance=True,
                    reasoning=f"Tool '{tool}' is not available",
                )

        if not self._is_tool_viable(state, tool, available):
            counts = self._source_counts(state)

            if tool == "read_url" and counts["readable_count"] >= 1:
                if "read_paper_abstract" in available:
                    tool = "read_paper_abstract"
                elif "extract_source_summary" in available:
                    tool = "extract_source_summary"
                else:
                    return ThinkResult(
                        action=DEFAULT_ACTION_ADVANCE,
                        is_advance=True,
                        reasoning="No URL available for read_url",
                    )
            else:
                return ThinkResult(
                    action=DEFAULT_ACTION_ADVANCE,
                    is_advance=True,
                    reasoning=(
                        f"Tool '{tool}' skipped because required "
                        f"inputs are unavailable"
                    ),
                )

        parameters = (
            result.parameters
            if isinstance(result.parameters, dict)
            else {}
        )
        step_description = clean_text(
            getattr(
                getattr(state, "current_step", None),
                "description",
                "",
            )
        )
        generated = self._build_heuristic_parameters(
            tool,
            step_description or getattr(state, "goal", ""),
            state,
        )

        for key, value in generated.items():
            parameters.setdefault(key, value)

        result.action = tool
        result.tool_name = tool
        result.parameters = parameters
        result.reasoning = (
            clean_text(result.reasoning) or f"Selected {tool}"
        )
        return result

    def _is_tool_viable(
        self,
        state: Any,
        tool_name: str,
        available: set[str] | None = None,
    ) -> bool:
        if available is not None and tool_name not in available:
            return False

        counts = self._source_counts(state)

        if tool_name == "compare_sources":
            return counts["source_count"] >= 2

        if tool_name in ("rank_sources", "classify_difficulty"):
            return counts["source_count"] >= 1

        if tool_name in (
            "extract_source_summary",
            "read_paper_abstract",
            "summarize_source",
            "check_relevance",
        ):
            return counts["readable_count"] >= 1

        if tool_name == "read_url":
            return counts["url_count"] >= 1

        if tool_name in (
            "generate_learning_path",
            "generate_report",
            "enhance_path",
        ):
            return counts["source_count"] >= 1

        if tool_name.startswith("search"):
            return True

        if tool_name in (
            "save_to_memory",
            "recall_memory",
            "search_memory",
        ):
            return True

        return True

    def _source_counts(self, state: Any) -> dict[str, int]:
        findings = getattr(state, "findings", []) or []
        ranked_sources = getattr(state, "ranked_sources", []) or []

        seen: set[str] = set()
        readable_count = 0
        url_count = 0
        unique_count = 0

        for source in findings:
            key = str(
                getattr(source, "source_id", "")
                or getattr(source, "title", "")
            )

            if key in seen:
                continue

            seen.add(key)
            unique_count += 1

            title = clean_text(getattr(source, "title", ""))
            abstract = clean_text(getattr(source, "abstract", ""))
            url = clean_text(getattr(source, "url", ""))

            if title or abstract:
                readable_count += 1

            if url:
                url_count += 1

        for item in ranked_sources:
            source = getattr(item, "source", None)

            if source is None:
                continue

            key = str(
                getattr(source, "source_id", "")
                or getattr(source, "title", "")
            )

            if key in seen:
                continue

            seen.add(key)
            unique_count += 1

            title = clean_text(getattr(source, "title", ""))
            abstract = clean_text(getattr(source, "abstract", ""))
            url = clean_text(getattr(source, "url", ""))

            if title or abstract:
                readable_count += 1

            if url:
                url_count += 1

        return {
            "findings_count": len(findings),
            "ranked_count": len(ranked_sources),
            "source_count": max(
                len(findings),
                len(ranked_sources),
                unique_count,
            ),
            "readable_count": readable_count,
            "url_count": url_count,
        }

    def _build_heuristic_parameters(
        self,
        tool_name: str,
        step_description: str,
        state: Any,
    ) -> dict[str, Any]:
        query = (
            step_description
            or getattr(state, "goal", "")
            or getattr(state, "topic", "")
        )
        sources = self._sources_payload(state)
        source = self._choose_source(state)

        if tool_name.startswith("search"):
            return {
                "query": query,
                "max_results": 8,
                "level": getattr(state, "level", "") or "",
                "goal": getattr(state, "goal", "") or "",
            }

        if tool_name == "rank_sources":
            return {
                "sources_data": sources,
                "query": query,
                "level": getattr(state, "level", "") or "",
                "goal": getattr(state, "goal", "") or "",
            }

        if tool_name == "classify_difficulty":
            return {"sources_data": sources}

        if tool_name == "compare_sources":
            return {"sources_data": sources, "query": query}

        if tool_name == "generate_learning_path":
            return {
                "sources_data": sources,
                "query": query,
                "level": getattr(state, "level", "") or "",
                "goal": getattr(state, "goal", "") or "",
            }

        if tool_name == "generate_report":
            return {"sources_data": sources, "query": query}

        if tool_name == "read_url":
            return {"url": self._first_url(state)}

        if tool_name == "read_paper_abstract":
            if source is not None:
                return {
                    "title": getattr(source, "title", "") or "",
                    "abstract": getattr(source, "abstract", "") or "",
                    "url": getattr(source, "url", "") or "",
                }

            return {"title": "", "abstract": "", "url": ""}

        if tool_name == "extract_source_summary":
            if source is not None:
                return {
                    "title": getattr(source, "title", "") or "",
                    "abstract": getattr(source, "abstract", "") or "",
                    "source_type": self._enum_value(
                        getattr(source, "source_type", "")
                    )
                    or "other",
                    "platform": self._enum_value(
                        getattr(source, "platform", "")
                    )
                    or "web",
                }

            return {
                "title": "",
                "abstract": "",
                "source_type": "",
                "platform": "",
            }

        if tool_name in ("summarize_source", "check_relevance"):
            if source is not None:
                return {
                    "title": getattr(source, "title", "") or "",
                    "abstract": getattr(source, "abstract", "") or "",
                    "query": query,
                }

        if tool_name in ("save_to_memory", "recall_memory"):
            return {
                "key": getattr(state, "topic", "")
                or getattr(state, "goal", "")
                or "agent_memory",
                "content": (
                    f"Researched {getattr(state, 'goal', '')}. "
                    f"Findings: {len(getattr(state, 'findings', []) or [])}."
                ),
                "memory_type": "episodic",
            }

        if tool_name == "search_memory":
            return {"query": query}

        return {"query": query}

    def _sources_payload(self, state: Any) -> list[dict[str, Any]]:
        ranked_sources = getattr(state, "ranked_sources", []) or []
        findings = getattr(state, "findings", []) or []
        payload: list[dict[str, Any]] = []

        if ranked_sources:
            for item in ranked_sources[:20]:
                source = getattr(item, "source", None)

                if source is None:
                    continue

                payload.append(
                    {
                        "source_id": getattr(
                            source, "source_id", ""
                        )
                        or "",
                        "title": getattr(source, "title", "") or "",
                        "url": getattr(source, "url", "") or "",
                        "platform": self._enum_value(
                            getattr(source, "platform", "")
                        )
                        or "web",
                        "source_type": self._enum_value(
                            getattr(source, "source_type", "")
                        )
                        or "other",
                        "abstract": (
                            getattr(source, "abstract", "") or ""
                        )[:500],
                        "year": getattr(source, "year", None),
                        "citation_count": getattr(
                            source, "citation_count", None
                        ),
                        "difficulty": self._enum_value(
                            getattr(source, "difficulty", None)
                        )
                        or None,
                        "rank": getattr(
                            item, "rank", len(payload) + 1
                        ),
                        "score": getattr(item, "score", 0.5),
                        "confidence": getattr(
                            item, "confidence", 0.5
                        ),
                    }
                )

            return payload

        for index, source in enumerate(findings[:20], start=1):
            payload.append(
                {
                    "source_id": getattr(source, "source_id", "")
                    or "",
                    "title": getattr(source, "title", "") or "",
                    "url": getattr(source, "url", "") or "",
                    "platform": self._enum_value(
                        getattr(source, "platform", "")
                    )
                    or "web",
                    "source_type": self._enum_value(
                        getattr(source, "source_type", "")
                    )
                    or "other",
                    "abstract": (
                        getattr(source, "abstract", "") or ""
                    )[:500],
                    "year": getattr(source, "year", None),
                    "citation_count": getattr(
                        source, "citation_count", None
                    ),
                    "difficulty": self._enum_value(
                        getattr(source, "difficulty", None)
                    )
                    or None,
                    "rank": index,
                    "score": 0.5,
                    "confidence": 0.5,
                }
            )

        return payload

    def _choose_source(self, state: Any) -> Any:
        findings = getattr(state, "findings", []) or []
        ranked_sources = getattr(state, "ranked_sources", []) or []

        for source in findings:
            if getattr(source, "abstract", None):
                return source

        for item in ranked_sources:
            source = getattr(item, "source", None)

            if source is not None and getattr(
                source, "abstract", None
            ):
                return source

        if findings:
            return findings[0]

        for item in ranked_sources:
            source = getattr(item, "source", None)

            if source is not None:
                return source

        return None

    def _first_url(self, state: Any) -> str:
        findings = getattr(state, "findings", []) or []
        ranked_sources = getattr(state, "ranked_sources", []) or []

        for source in findings:
            url = clean_text(getattr(source, "url", ""))

            if url:
                return url

        for item in ranked_sources:
            source = getattr(item, "source", None)

            if source is None:
                continue

            url = clean_text(getattr(source, "url", ""))

            if url:
                return url

        return ""

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value) or "")