from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import Field

from core.models import (
    AgentAction,
    AgentPlan,
    AgentStatus,
    CoreModel,
    RankedSource,
    Source,
)


DEFAULT_MAX_ITERATIONS = 15
DEFAULT_MAX_ACTIONS_PER_STEP = 4
DEFAULT_MAX_FINDINGS = 50


class LoopConfig(CoreModel):
    max_iterations: int = Field(default=DEFAULT_MAX_ITERATIONS, ge=1, le=50)
    max_actions_per_step: int = Field(default=DEFAULT_MAX_ACTIONS_PER_STEP, ge=1, le=10)
    max_findings: int = Field(default=DEFAULT_MAX_FINDINGS, ge=5, le=200)
    token_budget: int = Field(default=40000, ge=1000, le=500000)
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
    max_findings: int = Field(default=DEFAULT_MAX_FINDINGS, ge=5, le=200)
    total_tokens_used: int = Field(default=0, ge=0)
    total_budget: int = Field(default=40000, ge=0)
    tokens_remaining: int = Field(default=40000, ge=0)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    memory_context: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    latency_ms: float | None = None

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
        existing_ids = {s.source_id for s in self.findings}
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
        self.total_tokens_used += max(0, tokens)
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

    @property
    def success(self) -> bool:
        return self.status in (AgentStatus.COMPLETED, AgentStatus.SUCCESS)

    def to_summary(self) -> dict[str, Any]:
        return {
            "loop_id": self.loop_id,
            "status": str(getattr(self.status, "value", self.status)),
            "goal": self.goal,
            "findings_count": len(self.findings),
            "ranked_count": len(self.ranked_sources),
            "total_iterations": self.total_iterations,
            "total_tokens_used": self.total_tokens_used,
            "tokens_remaining": self.tokens_remaining,
            "errors": self.errors,
            "warnings": self.warnings,
            "latency_ms": self.latency_ms,
        }