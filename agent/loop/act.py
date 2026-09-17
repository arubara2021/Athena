from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from pydantic import Field

from core.models import AgentAction, AgentActionType, CoreModel, ToolCall
from utils.logger import get_logger, get_trace_logger

DEFAULT_ACT_TIMEOUT = 120.0


class ActResult(CoreModel):
    success: bool = False
    tool_name: str = ""
    action_type: str = ""
    data: Any = None
    tokens_used: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = None
    skipped: bool = False
    skip_reason: str = ""


class ActPhase:
    def __init__(self, timeout: float | None = None) -> None:
        self._timeout = timeout or DEFAULT_ACT_TIMEOUT
        self._logger = get_logger("agent.loop.act")
        self._trace = get_trace_logger()

    async def act(
        self,
        state: Any,
        think_result: Any,
        executor: Any,
    ) -> ActResult:
        if think_result.is_advance or think_result.is_complete:
            return ActResult(
                success=True,
                skipped=True,
                skip_reason="control_flow_action",
            )

        tool_name = str(think_result.tool_name or "").strip()

        if not tool_name:
            return ActResult(
                success=False,
                error="No tool selected",
                skipped=True,
                skip_reason="no_tool_selected",
            )

        if not executor.has_tool(tool_name):
            if tool_name == "read_url" and executor.has_tool("read_paper_abstract"):
                tool_name = "read_paper_abstract"
            else:
                return ActResult(
                    success=False,
                    tool_name=tool_name,
                    error=f"Tool '{tool_name}' is not available",
                    skipped=True,
                    skip_reason="tool_not_available",
                )

        parameters = self._prepare_parameters(state, tool_name, think_result)

        if tool_name == "read_url" and not str(parameters.get("url", "")).strip():
            if executor.has_tool("read_paper_abstract"):
                tool_name = "read_paper_abstract"
                parameters = self._prepare_parameters(state, tool_name, think_result)
            else:
                return ActResult(
                    success=False,
                    tool_name=tool_name,
                    error="No URL available to read",
                    skipped=True,
                    skip_reason="no_url_available",
                )

        if tool_name == "read_paper_abstract":
            if not str(parameters.get("title", "")).strip() and not str(
                parameters.get("abstract", "")
            ).strip():
                return ActResult(
                    success=False,
                    tool_name=tool_name,
                    error="No readable source available",
                    skipped=True,
                    skip_reason="no_readable_source",
                )

        start = time.perf_counter()

        self._trace.emit(
            "agent_acting",
            loop_id=getattr(state, "loop_id", ""),
            tool_name=tool_name,
            step_index=getattr(state, "current_step_index", 0),
            iteration=getattr(state, "iteration_count", 0),
        )

        try:
            tool_call = ToolCall(
                call_id=str(uuid4()),
                tool_name=tool_name,
                action_type=self._infer_action_type(tool_name),
                parameters=parameters,
            )

            result = await executor.execute(tool_call)

            latency_ms = (time.perf_counter() - start) * 1000
            tokens_used = getattr(result, "tokens_used", 0) or 0

            state.consume_tokens(tokens_used)

            action = AgentAction(
                action_id=str(uuid4()),
                step_index=getattr(state, "current_step_index", 0),
                action_type=self._infer_action_type(tool_name),
                description=think_result.reasoning,
                tool_call=result if hasattr(result, "tool_name") else tool_call,
                reasoning=think_result.reasoning,
                tokens_used=tokens_used,
            )

            state.add_action(action)

            if result.success:
                self._trace.emit(
                    "agent_tool_success",
                    loop_id=getattr(state, "loop_id", ""),
                    tool_name=tool_name,
                    tokens_used=tokens_used,
                    latency_ms=round(latency_ms, 1),
                )

                return ActResult(
                    success=True,
                    tool_name=tool_name,
                    action_type=self._enum_value(self._infer_action_type(tool_name)),
                    data=result.result,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                )

            error_msg = result.error or "Tool execution failed"
            state.add_warning(f"Tool {tool_name} failed: {error_msg}")

            self._trace.emit(
                "agent_tool_failed",
                loop_id=getattr(state, "loop_id", ""),
                tool_name=tool_name,
                error=error_msg,
                latency_ms=round(latency_ms, 1),
            )

            return ActResult(
                success=False,
                tool_name=tool_name,
                action_type=self._enum_value(self._infer_action_type(tool_name)),
                error=error_msg,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            error_msg = str(exc)

            state.add_error(f"Tool {tool_name} exception: {error_msg}")
            self._logger.warning(f"Act phase exception for {tool_name}: {exc}")

            self._trace.emit(
                "agent_tool_exception",
                loop_id=getattr(state, "loop_id", ""),
                tool_name=tool_name,
                error=error_msg,
                latency_ms=round(latency_ms, 1),
            )

            return ActResult(
                success=False,
                tool_name=tool_name,
                error=error_msg,
                latency_ms=latency_ms,
            )

    def _prepare_parameters(
        self,
        state: Any,
        tool_name: str,
        think_result: Any,
    ) -> dict[str, Any]:
        raw = think_result.parameters if isinstance(think_result.parameters, dict) else {}
        params = dict(raw)

        query = self._extract_query(params, state)

        if tool_name.startswith("search"):
            params["query"] = query
            if not params.get("max_results"):
                params["max_results"] = 8
            params.setdefault("level", getattr(state, "level", "") or "")
            params.setdefault("goal", getattr(state, "goal", "") or "")

        if tool_name == "rank_sources":
            sources_data = self._coerce_sources_data(params.get("sources_data"))
            if not sources_data:
                sources_data = self._sources_from_findings(state)
            params["sources_data"] = sources_data
            params["query"] = query
            params.setdefault("level", getattr(state, "level", "") or "")
            params.setdefault("goal", getattr(state, "goal", "") or "")

        if tool_name == "classify_difficulty":
            sources_data = self._coerce_sources_data(params.get("sources_data"))
            if not sources_data:
                sources_data = self._sources_from_findings(state)
            params["sources_data"] = sources_data

        if tool_name == "compare_sources":
            sources_data = self._coerce_sources_data(params.get("sources_data"))
            if not sources_data:
                sources_data = self._sources_from_findings(state)
            params["sources_data"] = sources_data
            params["query"] = query

        if tool_name == "generate_learning_path":
            sources_data = self._coerce_sources_data(params.get("sources_data"))
            if not sources_data:
                sources_data = self._sources_from_ranked(state)
            if not sources_data:
                sources_data = self._sources_from_findings(state)
            params["sources_data"] = sources_data
            params["query"] = query
            params.setdefault("level", getattr(state, "level", "") or "")
            params.setdefault("goal", getattr(state, "goal", "") or "")

        if tool_name == "generate_report":
            sources_data = self._coerce_sources_data(params.get("sources_data"))
            if not sources_data:
                sources_data = self._sources_from_ranked(state)
            if not sources_data:
                sources_data = self._sources_from_findings(state)
            params["sources_data"] = sources_data
            params["query"] = query

        if tool_name == "read_url":
            url = str(params.get("url", "") or "").strip()
            if not url:
                for source in getattr(state, "findings", []) or []:
                    candidate = str(getattr(source, "url", "") or "").strip()
                    if candidate:
                        url = candidate
                        break
            params = {"url": url}

        if tool_name == "read_paper_abstract":
            source = self._choose_readable_source(state, params)
            if source is not None:
                params = {
                    "title": getattr(source, "title", "") or "",
                    "abstract": getattr(source, "abstract", "") or "",
                    "url": getattr(source, "url", "") or "",
                }
            else:
                params = {
                    "title": str(params.get("title", "") or ""),
                    "abstract": str(params.get("abstract", "") or ""),
                    "url": str(params.get("url", "") or ""),
                }

        if tool_name in ("summarize_source", "check_relevance"):
            source = self._choose_readable_source(state, params)
            if source is not None:
                params["title"] = getattr(source, "title", "") or ""
                params["abstract"] = getattr(source, "abstract", "") or ""
                params.setdefault("query", query)

        if tool_name == "save_to_memory":
            params.setdefault("key", getattr(state, "topic", "") or getattr(state, "goal", "") or "agent_memory")
            params.setdefault(
                "content",
                f"Researched {getattr(state, 'goal', '')}. Findings: {len(getattr(state, 'findings', []) or [])}.",
            )
            params.setdefault("memory_type", "episodic")

        if tool_name == "recall_memory":
            params.setdefault("key", getattr(state, "topic", "") or getattr(state, "goal", "") or "agent_memory")

        if tool_name == "search_memory":
            params.setdefault("query", query)

        return params

    def _extract_query(self, params: dict[str, Any], state: Any) -> str:
        for key in ("query", "topic", "q", "search_query", "question"):
            value = str(params.get(key, "") or "").strip()
            if value:
                return value

        step = getattr(state, "current_step", None)
        if step is not None:
            description = str(getattr(step, "description", "") or "").strip()
            if description:
                return description

        return str(getattr(state, "goal", "") or getattr(state, "topic", "") or "")

    def _choose_readable_source(self, state: Any, params: dict[str, Any]) -> Any:
        findings = getattr(state, "findings", []) or []
        source_id = str(params.get("source_id", "") or "").strip()

        if source_id:
            for source in findings:
                if getattr(source, "source_id", "") == source_id:
                    return source

        for source in findings:
            if getattr(source, "abstract", None):
                return source

        if findings:
            return findings[0]

        return None

    def _coerce_sources_data(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []

        cleaned: list[dict[str, Any]] = []

        for item in value:
            if not isinstance(item, dict):
                continue
            if item.get("source_id") or item.get("title") or item.get("url"):
                cleaned.append(item)

        return cleaned[:20]

    def _sources_from_findings(self, state: Any) -> list[dict[str, Any]]:
        findings = getattr(state, "findings", []) or []
        result: list[dict[str, Any]] = []

        for index, source in enumerate(findings[:20], start=1):
            result.append(
                {
                    "source_id": getattr(source, "source_id", "") or "",
                    "title": getattr(source, "title", "") or "",
                    "url": getattr(source, "url", "") or "",
                    "platform": self._enum_value(getattr(source, "platform", "")) or "web",
                    "source_type": self._enum_value(getattr(source, "source_type", "")) or "other",
                    "abstract": (getattr(source, "abstract", "") or "")[:500],
                    "year": getattr(source, "year", None),
                    "citation_count": getattr(source, "citation_count", None),
                    "difficulty": self._enum_value(getattr(source, "difficulty", None)) or None,
                    "rank": index,
                    "score": 0.5,
                    "confidence": 0.5,
                }
            )

        return result

    def _sources_from_ranked(self, state: Any) -> list[dict[str, Any]]:
        ranked_sources = getattr(state, "ranked_sources", []) or []
        result: list[dict[str, Any]] = []

        for item in ranked_sources[:20]:
            source = getattr(item, "source", None)
            if source is None:
                continue

            result.append(
                {
                    "source_id": getattr(source, "source_id", "") or "",
                    "title": getattr(source, "title", "") or "",
                    "url": getattr(source, "url", "") or "",
                    "platform": self._enum_value(getattr(source, "platform", "")) or "web",
                    "source_type": self._enum_value(getattr(source, "source_type", "")) or "other",
                    "abstract": (getattr(source, "abstract", "") or "")[:500],
                    "year": getattr(source, "year", None),
                    "citation_count": getattr(source, "citation_count", None),
                    "difficulty": self._enum_value(getattr(source, "difficulty", None)) or None,
                    "rank": getattr(item, "rank", len(result) + 1),
                    "score": getattr(item, "score", 0.5),
                    "confidence": getattr(item, "confidence", 0.5),
                }
            )

        return result

    def _infer_action_type(self, tool_name: str) -> AgentActionType:
        name = tool_name.lower()

        if name == "search_memory":
            return AgentActionType.MEMORY_RECALL

        if name.startswith("search"):
            return AgentActionType.SEARCH

        if name.startswith("read"):
            return AgentActionType.READ

        if name in (
            "rank_sources",
            "classify_difficulty",
            "compare_sources",
            "check_relevance",
        ):
            return AgentActionType.ANALYZE

        if name.startswith("generate") or name.startswith("summarize") or name.startswith("enhance"):
            return AgentActionType.GENERATE

        if name.startswith("save"):
            return AgentActionType.MEMORY_STORE

        if name.startswith("recall"):
            return AgentActionType.MEMORY_RECALL

        return AgentActionType.NO_OP

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))