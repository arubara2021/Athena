from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import Field, model_validator

from core import constants
from core.models import (
    AgentAction,
    AgentPlan,
    AgentStatus,
    CoreModel,
    RankedSource,
    Source,
)


DEFAULT_MAX_ITERATIONS = int(
    getattr(constants, "AGENT_DEFAULT_MAX_ITERATIONS", constants.AGENT_DEFAULT_MAX_ITERATIONS)
)

DEFAULT_MAX_ACTIONS_PER_STEP = int(
    getattr(constants, "AGENT_DEFAULT_MAX_ACTIONS_PER_STEP", 4)
)

DEFAULT_MAX_FINDINGS = int(
    getattr(
        constants,
        "AGENT_DEFAULT_MAX_FINDINGS",
        max(20, int(constants.DEFAULT_MAX_RESULTS) * 2),
    )
)

DEFAULT_TOKEN_BUDGET = int(
    getattr(constants, "AGENT_DEFAULT_TOKEN_BUDGET", constants.AGENT_DEFAULT_TOKEN_BUDGET)
)


class LoopConfig(CoreModel):
    max_iterations: int = Field(default=DEFAULT_MAX_ITERATIONS, ge=1, le=50)
    max_actions_per_step: int = Field(default=DEFAULT_MAX_ACTIONS_PER_STEP, ge=1, le=10)
    max_findings: int = Field(default=DEFAULT_MAX_FINDINGS, ge=1, le=500)
    token_budget: int = Field(default=DEFAULT_TOKEN_BUDGET, ge=0)
    quality_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    enable_reflection: bool = True
    enable_memory: bool = True


class LoopState(CoreModel):
    loop_id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str = ""
    level: str = ""
    topic: str = ""
    plan: AgentPlan | None = None
    current_step_index: int = Field(default=0, ge=0)
    actions_taken: list[AgentAction] = Field(default_factory=list)
    findings: list[Source] = Field(default_factory=list)
    ranked_sources: list[RankedSource] = Field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    iteration_count: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=DEFAULT_MAX_ITERATIONS, ge=1)
    actions_in_current_step: int = Field(default=0, ge=0)
    max_actions_per_step: int = Field(default=DEFAULT_MAX_ACTIONS_PER_STEP, ge=1)
    max_findings: int = Field(default=DEFAULT_MAX_FINDINGS, ge=1)
    total_tokens_used: int = Field(default=0, ge=0)
    total_budget: int = Field(default=DEFAULT_TOKEN_BUDGET, ge=0)
    tokens_remaining: int = Field(default=DEFAULT_TOKEN_BUDGET, ge=0)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    memory_context: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    latency_ms: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_state(cls, data: Any) -> Any:
        if data is None:
            return {}

        if not isinstance(data, dict):
            return data

        normalized = dict(data)

        aliases = {
            "budget": "total_budget",
            "token_budget": "total_budget",
            "total_token_budget": "total_budget",
            "max_token_budget": "total_budget",
            "remaining_tokens": "tokens_remaining",
            "tokens_left": "tokens_remaining",
            "max_iteration": "max_iterations",
            "iteration_limit": "max_iterations",
            "max_finding": "max_findings",
            "finding_limit": "max_findings",
            "actions_per_step": "max_actions_per_step",
        }

        for old_key, new_key in aliases.items():
            if old_key in normalized:
                if new_key not in normalized or normalized.get(new_key) is None:
                    normalized[new_key] = normalized.pop(old_key)
                else:
                    normalized.pop(old_key)

        if normalized.get("total_budget") is None:
            normalized["total_budget"] = DEFAULT_TOKEN_BUDGET

        try:
            total_budget = max(0, int(normalized["total_budget"]))
        except Exception:
            total_budget = DEFAULT_TOKEN_BUDGET

        normalized["total_budget"] = total_budget

        if normalized.get("tokens_remaining") is None:
            normalized["tokens_remaining"] = total_budget

        try:
            tokens_remaining = int(normalized["tokens_remaining"])
        except Exception:
            tokens_remaining = total_budget

        normalized["tokens_remaining"] = max(0, min(total_budget, tokens_remaining))

        numeric_fields = (
            ("max_iterations", 1, DEFAULT_MAX_ITERATIONS),
            ("max_findings", 1, DEFAULT_MAX_FINDINGS),
            ("max_actions_per_step", 1, DEFAULT_MAX_ACTIONS_PER_STEP),
            ("iteration_count", 0, 0),
            ("actions_in_current_step", 0, 0),
            ("current_step_index", 0, 0),
            ("total_tokens_used", 0, 0),
        )

        for field_name, minimum_value, default_value in numeric_fields:
            if field_name not in normalized:
                continue

            value = normalized[field_name]

            if value is None:
                normalized[field_name] = default_value
                continue

            try:
                normalized[field_name] = max(minimum_value, int(value))
            except Exception:
                normalized[field_name] = default_value

        for old_key in aliases:
            normalized.pop(old_key, None)

        return normalized

    @property
    def token_budget(self) -> int:
        return self.total_budget

    @token_budget.setter
    def token_budget(self, value: Any) -> None:
        try:
            budget = max(0, int(value))
        except Exception:
            budget = self.total_budget

        used = max(0, self.total_budget - self.tokens_remaining)
        self.total_budget = budget
        self.tokens_remaining = max(0, budget - used)

    @property
    def current_step(self) -> Any:
        if self.plan is None:
            return None

        if self.current_step_index >= len(self.plan.steps):
            return None

        return self.plan.steps[self.current_step_index]

    @property
    def total_steps(self) -> int:
        if self.plan is None:
            return 0

        return len(self.plan.steps)

    @property
    def is_plan_complete(self) -> bool:
        if self.plan is None:
            return True

        return self.current_step_index >= len(self.plan.steps)

    @property
    def budget_percent_used(self) -> float:
        if self.total_budget <= 0:
            return 100.0

        return min(100.0, (self.total_tokens_used / self.total_budget) * 100.0)

    @property
    def is_budget_critical(self) -> bool:
        return self.budget_percent_used >= 90.0

    @property
    def is_budget_warning(self) -> bool:
        return self.budget_percent_used >= 75.0

    def mark_running(self) -> None:
        self.status = AgentStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def mark_completed(self) -> None:
        self.status = AgentStatus.COMPLETED
        self._finalize()

    def mark_failed(self, error: str | None = None) -> None:
        if error:
            self.add_error(error)

        self.status = AgentStatus.FAILED
        self._finalize()

    def mark_budget_exhausted(self) -> None:
        self.status = AgentStatus.BUDGET_EXHAUSTED
        self._finalize()

    def add_error(self, error: str) -> None:
        text = str(error or "").strip()

        if text and text not in self.errors:
            self.errors.append(text)

    def add_warning(self, warning: str) -> None:
        text = str(warning or "").strip()

        if text and text not in self.warnings:
            self.warnings.append(text)

    def add_finding(self, source: Source) -> None:
        if len(self.findings) >= self.max_findings:
            return

        existing_ids = {item.source_id for item in self.findings}

        if source.source_id not in existing_ids:
            self.findings.append(source)

    def add_action(self, action: AgentAction) -> None:
        self.actions_taken.append(action)
        self.actions_in_current_step += 1

    def advance_step(self) -> bool:
        if self.plan is None:
            return False

        next_index = self.current_step_index + 1

        if next_index >= len(self.plan.steps):
            return False

        self.current_step_index = next_index
        self.actions_in_current_step = 0
        return True

    def consume_tokens(self, tokens: int) -> None:
        consumed = max(0, int(tokens or 0))
        self.total_tokens_used += consumed
        self.tokens_remaining = max(0, self.total_budget - self.total_tokens_used)

    def increment_iteration(self) -> None:
        self.iteration_count += 1

    def _finalize(self) -> None:
        self.finished_at = datetime.now(timezone.utc)

        if self.started_at is not None:
            delta = (self.finished_at - self.started_at).total_seconds() * 1000
            self.latency_ms = max(0.0, delta)

    def to_summary(self) -> dict[str, Any]:
        return {
            "loop_id": self.loop_id,
            "goal": self.goal,
            "status": self._enum_value(self.status),
            "current_step": self.current_step_index + 1,
            "total_steps": self.total_steps,
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
            "actions_taken": len(self.actions_taken),
            "findings_count": len(self.findings),
            "ranked_count": len(self.ranked_sources),
            "tokens_used": self.total_tokens_used,
            "token_budget": self.total_budget,
            "total_budget": self.total_budget,
            "tokens_remaining": self.tokens_remaining,
            "budget_percent_used": round(self.budget_percent_used, 1),
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
        }

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))


class LoopOutput(CoreModel):
    loop_id: str = ""
    status: AgentStatus = AgentStatus.FAILED
    goal: str = ""
    level: str = ""
    findings: list[Source] = Field(default_factory=list)
    ranked_sources: list[RankedSource] = Field(default_factory=list)
    learning_path: Any = None
    enhanced_learning_path: Any = None
    actions_taken: list[AgentAction] = Field(default_factory=list)
    total_iterations: int = Field(default=0, ge=0)
    total_tokens_used: int = Field(default=0, ge=0)
    total_budget: int = Field(default=0, ge=0)
    tokens_remaining: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    latency_ms: float | None = None
    max_iterations: int = Field(default=DEFAULT_MAX_ITERATIONS, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _normalize_output(cls, data: Any) -> Any:
        if data is None:
            return {}

        if not isinstance(data, dict):
            return data

        normalized = dict(data)

        aliases = {
            "token_budget": "total_budget",
            "budget": "total_budget",
            "total_token_budget": "total_budget",
            "remaining_tokens": "tokens_remaining",
            "tokens_left": "tokens_remaining",
            "max_iteration": "max_iterations",
            "iteration_limit": "max_iterations",
        }

        for old_key, new_key in aliases.items():
            if old_key in normalized:
                if new_key not in normalized or normalized.get(new_key) is None:
                    normalized[new_key] = normalized.pop(old_key)
                else:
                    normalized.pop(old_key)

        if normalized.get("total_budget") is None:
            try:
                fallback_budget = max(0, int(normalized.get("total_tokens_used") or 0))
            except Exception:
                fallback_budget = 0

            normalized["total_budget"] = fallback_budget

        try:
            total_budget = max(0, int(normalized["total_budget"]))
        except Exception:
            total_budget = 0

        normalized["total_budget"] = total_budget

        if normalized.get("tokens_remaining") is None:
            try:
                used_tokens = max(0, int(normalized.get("total_tokens_used") or 0))
            except Exception:
                used_tokens = 0

            normalized["tokens_remaining"] = max(0, total_budget - used_tokens)

        try:
            tokens_remaining = int(normalized["tokens_remaining"])
        except Exception:
            tokens_remaining = total_budget

        normalized["tokens_remaining"] = max(0, min(total_budget, tokens_remaining))

        if normalized.get("max_iterations") is None:
            normalized["max_iterations"] = DEFAULT_MAX_ITERATIONS

        try:
            normalized["max_iterations"] = max(1, int(normalized["max_iterations"]))
        except Exception:
            normalized["max_iterations"] = DEFAULT_MAX_ITERATIONS

        for old_key in aliases:
            normalized.pop(old_key, None)

        return normalized

    @property
    def success(self) -> bool:
        status_value = str(getattr(self.status, "value", self.status)).lower()
        return status_value in {"completed", "success"}

    def to_summary(self) -> dict[str, Any]:
        return {
            "loop_id": self.loop_id,
            "status": str(getattr(self.status, "value", self.status)),
            "goal": self.goal,
            "findings_count": len(self.findings),
            "ranked_count": len(self.ranked_sources),
            "total_iterations": self.total_iterations,
            "total_tokens_used": self.total_tokens_used,
            "total_budget": self.total_budget,
            "tokens_remaining": self.tokens_remaining,
            "max_iterations": self.max_iterations,
            "errors": self.errors,
            "warnings": self.warnings,
            "latency_ms": self.latency_ms,
        }