from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any

from pydantic import Field

from core.models import CoreModel, ToolCall
from agent.tools.registry import ToolRegistry
from utils.logger import get_logger, get_trace_logger


class ToolResult(CoreModel):
    tool_name: str = ""
    success: bool = False
    data: Any = None
    error: str | None = None
    tokens_used: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        cost_tracker: Any | None = None,
        default_timeout: float = 60.0,
    ) -> None:
        self._registry = registry
        self._cost_tracker = cost_tracker
        self._default_timeout = default_timeout
        self._logger = get_logger("agent.tools.executor")
        self._trace = get_trace_logger()

    def has_tool(self, name: str) -> bool:
        return self._registry.has_tool(name)

    def can_proceed(self, parameters: dict[str, Any] | None = None) -> bool:
        if self._cost_tracker is None:
            return True
        return self._cost_tracker.can_proceed()

    async def execute(self, tool_call: ToolCall) -> ToolCall:
        start = time.perf_counter()
        tool_name = tool_call.tool_name

        definition = self._registry.get(tool_name)
        if definition is None:
            tool_call.success = False
            tool_call.error = f"Tool '{tool_name}' not found in registry"
            tool_call.latency_ms = (time.perf_counter() - start) * 1000
            self._trace.emit(
                "tool_execution_failed",
                tool_name=tool_name,
                error=tool_call.error,
            )
            return tool_call

        if self._cost_tracker is not None:
            if not self._cost_tracker.can_proceed(
                estimated_tokens=definition.token_cost_est,
                category=definition.category,
            ):
                tool_call.success = False
                tool_call.error = "Insufficient token budget for this tool"
                tool_call.latency_ms = (time.perf_counter() - start) * 1000
                self._trace.emit(
                    "tool_budget_rejected",
                    tool_name=tool_name,
                    estimated_cost=definition.token_cost_est,
                    category=definition.category,
                )
                return tool_call

        executor_fn = self._registry.get_executor(tool_name)
        if executor_fn is None:
            tool_call.success = False
            tool_call.error = f"No executor function for tool '{tool_name}'"
            tool_call.latency_ms = (time.perf_counter() - start) * 1000
            return tool_call

        parameters = self._sanitize_parameters(executor_fn, tool_call.parameters)
        timeout = definition.timeout_seconds or self._default_timeout

        try:
            result = await asyncio.wait_for(
                executor_fn(**parameters),
                timeout=timeout,
            )

            if isinstance(result, ToolResult):
                tool_call.success = bool(result.success)
                tool_call.result = result.data
                tool_call.tokens_used = int(result.tokens_used or 0)
                tool_call.error = result.error
                if result.latency_ms:
                    tool_call.latency_ms = float(result.latency_ms)
            elif isinstance(result, dict):
                tool_call.success = bool(result.get("success", True))
                tool_call.result = result.get("data", result)
                tool_call.tokens_used = int(result.get("tokens_used", 0) or 0)
                tool_call.error = result.get("error")
            else:
                tool_call.success = True
                tool_call.result = result
                tool_call.tokens_used = 0

        except asyncio.TimeoutError:
            tool_call.success = False
            tool_call.error = f"Tool '{tool_name}' timed out after {timeout}s"
            self._logger.warning(f"Tool {tool_name} timed out")
        except Exception as exc:
            tool_call.success = False
            tool_call.error = f"Tool '{tool_name}' failed: {str(exc)}"
            self._logger.warning(f"Tool {tool_name} failed: {exc}")

        tool_call.latency_ms = (time.perf_counter() - start) * 1000

        if self._cost_tracker is not None and tool_call.tokens_used > 0:
            self._cost_tracker.record_usage(
                provider="agent_tool",
                model_id=tool_name,
                tokens_used=tool_call.tokens_used,
                category=definition.category,
                latency_ms=tool_call.latency_ms,
                success=tool_call.success,
            )

        self._trace.emit(
            "tool_execution_completed",
            tool_name=tool_name,
            success=tool_call.success,
            tokens_used=tool_call.tokens_used,
            latency_ms=round(tool_call.latency_ms, 1),
            error=tool_call.error,
        )

        return tool_call

    async def execute_by_name(
        self,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
    ) -> ToolCall:
        tool_call = ToolCall(
            call_id=str(__import__("uuid").uuid4()),
            tool_name=tool_name,
            parameters=parameters or {},
        )
        return await self.execute(tool_call)

    def get_available_tools_summary(self) -> list[dict[str, Any]]:
        return self._registry.to_summary()

    def _sanitize_parameters(
        self,
        executor_fn: Any,
        parameters: Any,
    ) -> dict[str, Any]:
        if not isinstance(parameters, dict):
            parameters = {}

        try:
            signature = inspect.signature(executor_fn)
        except Exception:
            return parameters

        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )

        if accepts_kwargs:
            return parameters

        aliases = {
            "topic": "query",
            "q": "query",
            "search_query": "query",
            "question": "query",
            "limit": "max_results",
            "results": "max_results",
            "max_sources": "max_results",
            "sources": "sources_data",
            "ranked_sources": "sources_data",
            "items": "sources_data",
            "source": "sources_data",
            "text": "content",
            "page_content": "content",
            "memory_key": "key",
        }

        valid_names = {
            param.name
            for param in signature.parameters.values()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }

        cleaned: dict[str, Any] = {}

        for key, value in parameters.items():
            name = aliases.get(key, key)
            if name in valid_names and name not in cleaned:
                cleaned[name] = value

        if "max_results" in cleaned:
            try:
                cleaned["max_results"] = int(cleaned["max_results"])
            except Exception:
                cleaned.pop("max_results", None)

        if "top_k" in cleaned:
            try:
                cleaned["top_k"] = int(cleaned["top_k"])
            except Exception:
                cleaned.pop("top_k", None)

        return cleaned