from __future__ import annotations

import threading
from typing import Any
from datetime import datetime, timezone

from pydantic import Field

from core.models import CoreModel


class TaskContext(CoreModel):
    goal: str = ""
    level: str = ""
    topic: str = ""
    sub_tasks: list[str] = Field(default_factory=list)
    current_sub_task_index: int = Field(default=0, ge=0)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    intermediate_results: dict[str, Any] = Field(default_factory=dict)
    scratchpad: dict[str, Any] = Field(default_factory=dict)
    action_history: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ShortTermMemory:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._context: TaskContext | None = None

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._context is not None

    def initialize(
        self,
        goal: str,
        level: str = "",
        topic: str = "",
    ) -> TaskContext:
        with self._lock:
            self._context = TaskContext(
                goal=goal,
                level=level,
                topic=topic,
            )
            return self._context

    def get_context(self) -> TaskContext | None:
        with self._lock:
            return self._context

    def get_goal(self) -> str:
        with self._lock:
            if self._context is None:
                return ""
            return self._context.goal

    def get_level(self) -> str:
        with self._lock:
            if self._context is None:
                return ""
            return self._context.level

    def get_topic(self) -> str:
        with self._lock:
            if self._context is None:
                return ""
            return self._context.topic

    def set_sub_tasks(self, sub_tasks: list[str]) -> None:
        with self._lock:
            if self._context is None:
                return
            self._context.sub_tasks = list(sub_tasks)
            self._context.current_sub_task_index = 0
            self._touch()

    def get_sub_tasks(self) -> list[str]:
        with self._lock:
            if self._context is None:
                return []
            return list(self._context.sub_tasks)

    def get_current_sub_task(self) -> str:
        with self._lock:
            if self._context is None:
                return ""
            if self._context.current_sub_task_index >= len(self._context.sub_tasks):
                return ""
            return self._context.sub_tasks[self._context.current_sub_task_index]

    def advance_sub_task(self) -> bool:
        with self._lock:
            if self._context is None:
                return False
            next_index = self._context.current_sub_task_index + 1
            if next_index >= len(self._context.sub_tasks):
                return False
            self._context.current_sub_task_index = next_index
            self._touch()
            return True

    def add_finding(self, finding: dict[str, Any]) -> None:
        with self._lock:
            if self._context is None:
                return
            self._context.findings.append(finding)
            self._touch()

    def get_findings(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._context is None:
                return []
            return list(self._context.findings)

    def set_intermediate(self, key: str, value: Any) -> None:
        with self._lock:
            if self._context is None:
                return
            self._context.intermediate_results[key] = value
            self._touch()

    def get_intermediate(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if self._context is None:
                return default
            return self._context.intermediate_results.get(key, default)

    def set_scratchpad(self, key: str, value: Any) -> None:
        with self._lock:
            if self._context is None:
                return
            self._context.scratchpad[key] = value
            self._touch()

    def get_scratchpad(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if self._context is None:
                return default
            return self._context.scratchpad.get(key, default)

    def get_scratchpad_summary(self) -> dict[str, Any]:
        with self._lock:
            if self._context is None:
                return {}
            return dict(self._context.scratchpad)

    def record_action(self, action: dict[str, Any]) -> None:
        with self._lock:
            if self._context is None:
                return
            self._context.action_history.append(action)
            self._touch()

    def get_action_history(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._context is None:
                return []
            return list(self._context.action_history)

    def add_token_usage(self, category: str, tokens: int) -> None:
        with self._lock:
            if self._context is None:
                return
            current = self._context.token_usage.get(category, 0)
            self._context.token_usage[category] = current + max(0, tokens)
            self._touch()

    def get_total_tokens_used(self) -> int:
        with self._lock:
            if self._context is None:
                return 0
            return sum(self._context.token_usage.values())

    def get_token_usage_by_category(self) -> dict[str, int]:
        with self._lock:
            if self._context is None:
                return {}
            return dict(self._context.token_usage)

    def add_error(self, error: str) -> None:
        with self._lock:
            if self._context is None:
                return
            if error:
                self._context.errors.append(error)
                self._touch()

    def add_warning(self, warning: str) -> None:
        with self._lock:
            if self._context is None:
                return
            if warning:
                self._context.warnings.append(warning)
                self._touch()

    def get_errors(self) -> list[str]:
        with self._lock:
            if self._context is None:
                return []
            return list(self._context.errors)

    def get_warnings(self) -> list[str]:
        with self._lock:
            if self._context is None:
                return []
            return list(self._context.warnings)

    def to_summary(self) -> dict[str, Any]:
        with self._lock:
            if self._context is None:
                return {}
            return {
                "goal": self._context.goal,
                "level": self._context.level,
                "topic": self._context.topic,
                "total_sub_tasks": len(self._context.sub_tasks),
                "current_sub_task_index": self._context.current_sub_task_index,
                "total_findings": len(self._context.findings),
                "total_actions": len(self._context.action_history),
                "total_tokens_used": sum(self._context.token_usage.values()),
                "total_errors": len(self._context.errors),
                "total_warnings": len(self._context.warnings),
            }

    def clear(self) -> None:
        with self._lock:
            self._context = None

    def _touch(self) -> None:
        if self._context is not None:
            self._context.updated_at = datetime.now(timezone.utc).isoformat()