from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field

from core.config import get_logs_directory
from core.models import CoreModel
from utils.logger import get_logger, get_trace_logger
from agent.token.budget import (
    ModelTier,
    TokenBudget,
    TokenBudgetConfig,
    TokenCategory,
    load_token_budget_config,
)
from agent.token.ledger import TokenLedger


class ModelUsageSummary(CoreModel):
    provider: str
    model_id: str
    call_count: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    tier: str = ModelTier.FAST.value


class CategoryUsageSummary(CoreModel):
    category: str
    total_tokens: int = 0
    call_count: int = 0
    max_allowed: int = 0
    usage_percent: float = 0.0


class TokenUsageReport(CoreModel):
    task_id: str
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_budget: int = 0
    total_spent: int = 0
    total_remaining: int = 0
    usage_percent: float = 0.0
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    wrap_up_mode_activated: bool = False
    circuit_breaker_triggered: bool = False
    per_model: list[ModelUsageSummary] = Field(default_factory=list)
    per_category: list[CategoryUsageSummary] = Field(default_factory=list)
    per_tier: dict[str, int] = Field(default_factory=dict)
    estimated_cost_weight: float = 0.0


class CostTracker:
    def __init__(
        self,
        task_id: str,
        budget: TokenBudget | None = None,
        config: TokenBudgetConfig | None = None,
    ) -> None:
        self._config = config or load_token_budget_config()
        self._task_id = task_id
        self._budget = budget or TokenBudget(task_id=task_id, config=self._config)
        self._ledger = TokenLedger(task_id=task_id, config=self._config)
        self._lock = threading.Lock()
        self._circuit_breaker_triggered = False
        self._circuit_breaker_reason = ""
        self._logger = get_logger("agent.token.cost_tracker")
        self._trace = get_trace_logger()

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def budget(self) -> TokenBudget:
        return self._budget

    @property
    def ledger(self) -> TokenLedger:
        return self._ledger

    @property
    def is_circuit_breaker_triggered(self) -> bool:
        with self._lock:
            return self._circuit_breaker_triggered

    @property
    def circuit_breaker_reason(self) -> str:
        with self._lock:
            return self._circuit_breaker_reason

    def can_proceed(
        self,
        estimated_tokens: int = 0,
        category: str | TokenCategory | None = None,
    ) -> bool:
        with self._lock:
            if self._circuit_breaker_triggered:
                return False

        if self._budget.should_stop():
            self._trigger_circuit_breaker(
                "max_consecutive_exceedances_reached"
            )
            return False

        if estimated_tokens > 0:
            return self._budget.is_within_budget(
                requested_tokens=estimated_tokens,
                category=category,
            )

        return self._budget.remaining > 0

    def record_usage(
        self,
        provider: str,
        model_id: str,
        tokens_used: int,
        category: str | TokenCategory | None = None,
        model_tier: str | ModelTier | None = None,
        latency_ms: float = 0.0,
        success: bool = True,
        error: str | None = None,
        prompt_preview: str = "",
    ) -> bool:
        resolved_category = ""
        if category is not None:
            if isinstance(category, TokenCategory):
                resolved_category = category.value
            else:
                resolved_category = str(category).strip().lower()

        resolved_tier = ModelTier.FAST.value
        if model_tier is not None:
            if isinstance(model_tier, ModelTier):
                resolved_tier = model_tier.value
            else:
                resolved_tier = str(model_tier).strip().lower()

        if success and tokens_used > 0:
            deducted = self._budget.deduct(
                tokens=tokens_used,
                category=resolved_category,
                model_id=model_id,
            )
            if not deducted:
                self._logger.warning(
                    f"Token deduction rejected for task {self._task_id}: "
                    f"model={model_id}, tokens={tokens_used}, "
                    f"category={resolved_category}"
                )

        self._ledger.record(
            provider=provider,
            model_id=model_id,
            category=resolved_category,
            model_tier=resolved_tier,
            total_tokens=tokens_used,
            latency_ms=latency_ms,
            success=success,
            error=error,
            prompt_preview=prompt_preview,
            budget_spent_after=self._budget.spent,
            budget_remaining_after=self._budget.remaining,
        )

        self._trace.emit(
            "token_usage_recorded",
            task_id=self._task_id,
            provider=provider,
            model_id=model_id,
            category=resolved_category,
            tokens_used=tokens_used,
            budget_spent=self._budget.spent,
            budget_remaining=self._budget.remaining,
            usage_percent=self._budget.get_usage_percent(),
            warning_level=self._budget.get_warning_level(),
            wrap_up_mode=self._budget.is_wrap_up_mode,
        )

        warning_level = self._budget.get_warning_level()
        if warning_level == "critical":
            self._logger.warning(
                f"CRITICAL token usage for task {self._task_id}: "
                f"{self._budget.get_usage_percent():.1f}% used"
            )
        elif warning_level == "warning":
            self._logger.info(
                f"Token usage warning for task {self._task_id}: "
                f"{self._budget.get_usage_percent():.1f}% used"
            )

        if self._budget.should_stop():
            self._trigger_circuit_breaker("budget_exhaustion")

        return success

    def check_budget_before_call(
        self,
        estimated_tokens: int,
        category: str | TokenCategory | None = None,
        model_id: str = "",
    ) -> bool:
        if self.is_circuit_breaker_triggered:
            self._trace.emit(
                "token_budget_rejected",
                task_id=self._task_id,
                reason="circuit_breaker",
                estimated_tokens=estimated_tokens,
                category=self._resolve_category_str(category),
                model_id=model_id,
            )
            return False

        if not self._budget.is_within_budget(
            requested_tokens=estimated_tokens,
            category=category,
        ):
            self._trace.emit(
                "token_budget_rejected",
                task_id=self._task_id,
                reason="insufficient_budget",
                estimated_tokens=estimated_tokens,
                category=self._resolve_category_str(category),
                model_id=model_id,
                budget_remaining=self._budget.remaining,
            )
            self._logger.warning(
                f"Budget check failed before call for task {self._task_id}: "
                f"estimated={estimated_tokens}, remaining={self._budget.remaining}, "
                f"category={self._resolve_category_str(category)}"
            )
            return False

        return True

    def generate_report(self) -> TokenUsageReport:
        entries = self._ledger.get_entries()

        per_model: dict[str, ModelUsageSummary] = {}
        per_category: dict[str, CategoryUsageSummary] = {}
        per_tier: dict[str, int] = {}
        total_latency = 0.0
        success_count = 0
        failure_count = 0
        estimated_cost_weight = 0.0

        for entry in entries:
            model_key = f"{entry.provider}/{entry.model_id}"
            if model_key not in per_model:
                per_model[model_key] = ModelUsageSummary(
                    provider=entry.provider,
                    model_id=entry.model_id,
                    tier=entry.model_tier,
                )
            model_summary = per_model[model_key]
            model_summary.call_count += 1
            model_summary.total_tokens += entry.total_tokens
            model_summary.total_latency_ms += entry.latency_ms
            if entry.success:
                model_summary.success_count += 1
                success_count += 1
            else:
                model_summary.failure_count += 1
                failure_count += 1

            cat_key = entry.category or "uncategorized"
            if cat_key not in per_category:
                cat_config = self._config.categories.get(cat_key)
                max_allowed = cat_config.max_tokens if cat_config else 0
                per_category[cat_key] = CategoryUsageSummary(
                    category=cat_key,
                    max_allowed=max_allowed,
                )
            cat_summary = per_category[cat_key]
            cat_summary.total_tokens += entry.total_tokens
            cat_summary.call_count += 1
            if cat_summary.max_allowed > 0:
                cat_summary.usage_percent = min(
                    100.0,
                    (cat_summary.total_tokens / cat_summary.max_allowed) * 100.0,
                )

            per_tier[entry.model_tier] = (
                per_tier.get(entry.model_tier, 0) + entry.total_tokens
            )

            tier_config = self._config.model_tiers.get(entry.model_tier)
            if tier_config:
                estimated_cost_weight += entry.total_tokens * tier_config.cost_weight
            else:
                estimated_cost_weight += entry.total_tokens

            total_latency += entry.latency_ms

        report = TokenUsageReport(
            task_id=self._task_id,
            total_budget=self._budget.total_budget,
            total_spent=self._budget.spent,
            total_remaining=self._budget.remaining,
            usage_percent=self._budget.get_usage_percent(),
            total_calls=len(entries),
            successful_calls=success_count,
            failed_calls=failure_count,
            wrap_up_mode_activated=self._budget.is_wrap_up_mode,
            circuit_breaker_triggered=self._circuit_breaker_triggered,
            per_model=sorted(
                per_model.values(),
                key=lambda m: m.total_tokens,
                reverse=True,
            ),
            per_category=sorted(
                per_category.values(),
                key=lambda c: c.total_tokens,
                reverse=True,
            ),
            per_tier=per_tier,
            estimated_cost_weight=estimated_cost_weight,
        )

        self._trace.emit(
            "token_usage_report_generated",
            task_id=self._task_id,
            total_spent=report.total_spent,
            total_budget=report.total_budget,
            usage_percent=report.usage_percent,
            total_calls=report.total_calls,
            circuit_breaker_triggered=report.circuit_breaker_triggered,
        )

        return report

    def save_report(self, report: TokenUsageReport | None = None) -> Path | None:
        if report is None:
            report = self.generate_report()

        if not self._config.reporting.save_report_to_file:
            return None

        try:
            logs_dir = get_logs_directory()
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = (
                f"{self._config.reporting.report_filename_prefix}"
                f"_{self._task_id[:8]}_{timestamp}.json"
            )
            report_path = logs_dir / filename

            report_data = report.model_dump(mode="json")

            if not self._config.reporting.include_per_model_breakdown:
                report_data.pop("per_model", None)
            if not self._config.reporting.include_per_category_breakdown:
                report_data.pop("per_category", None)

            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(report_data, handle, indent=2, ensure_ascii=False)

            self._logger.info(f"Token usage report saved to {report_path}")
            return report_path
        except Exception as exc:
            self._logger.warning(f"Failed to save token usage report: {exc}")
            return None

    def close(self) -> None:
        self._ledger.close()
        report = self.generate_report()
        self.save_report(report)

    def _trigger_circuit_breaker(self, reason: str) -> None:
        with self._lock:
            if not self._circuit_breaker_triggered:
                self._circuit_breaker_triggered = True
                self._circuit_breaker_reason = reason
                self._logger.warning(
                    f"CIRCUIT BREAKER TRIGGERED for task {self._task_id}: "
                    f"reason={reason}, spent={self._budget.spent}, "
                    f"budget={self._budget.total_budget}"
                )
                self._trace.emit(
                    "token_circuit_breaker_triggered",
                    task_id=self._task_id,
                    reason=reason,
                    budget_spent=self._budget.spent,
                    total_budget=self._budget.total_budget,
                )

    def _resolve_category_str(self, category: str | TokenCategory | None) -> str:
        if category is None:
            return ""
        if isinstance(category, TokenCategory):
            return category.value
        return str(category).strip().lower()