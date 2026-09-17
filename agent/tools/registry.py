from __future__ import annotations

from typing import Any, Callable, Awaitable
from pydantic import Field

from core.models import CoreModel
from utils.logger import get_logger


class ToolDefinition(CoreModel):
    name: str
    description: str = ""
    category: str = "general"
    token_cost_est: int = Field(default=500, ge=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    requires_llm: bool = False
    parameters_schema: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._executors: dict[str, Callable[..., Awaitable[Any]]] = {}
        self._logger = get_logger("agent.tools.registry")

    def register(
        self,
        definition: ToolDefinition,
        execute_fn: Callable[..., Awaitable[Any]],
    ) -> None:
        if definition.name in self._tools:
            self._logger.warning(f"Tool '{definition.name}' already registered, overwriting")
        self._tools[definition.name] = definition
        self._executors[definition.name] = execute_fn

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def get_executor(self, name: str) -> Callable[..., Awaitable[Any]] | None:
        return self._executors.get(name)

    def list_all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def list_by_category(self, category: str) -> list[ToolDefinition]:
        return [t for t in self._tools.values() if t.category == category]

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def find_tools_for_action(self, action_type: str) -> list[ToolDefinition]:
        category_map = {
            "search": "search",
            "read": "read",
            "analyze": "analysis",
            "generate": "generation",
            "memory_store": "memory",
            "memory_recall": "memory",
            "reflect": "analysis",
            "plan": "generation",
            "synthesize": "generation",
        }
        category = category_map.get(action_type, "general")
        return self.list_by_category(category)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def to_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "category": t.category,
                "description": t.description,
                "token_cost_est": t.token_cost_est,
                "requires_llm": t.requires_llm,
            }
            for t in self._tools.values()
        ]