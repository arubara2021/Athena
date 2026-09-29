from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field

from agent.token.budget import TokenBudget, load_token_budget_config
from core.config import get_logs_directory
from core.models import CoreModel
from utils.logger import get_logger, get_trace_logger


__all__ = [
    "CostTracker",
    "TokenUsageReport",
    "ModelUsageSummary",
    "CategoryUsageSummary",
]


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


class ModelUsageSummary(CoreModel):
    model: str = ""
    calls: int = Field(default=0, ge=0)
    tokens_used: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)


class CategoryUsageSummary(CoreModel):
    category: str = ""
    calls: int = Field(default=0, ge=0)
    tokens_used: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)


class TokenUsageReport(CoreModel):
    task_id: str = ""
    generated_at: str = ""
    total_calls: int = Field(default=0, ge=0)
    total_tokens_used: int = Field(default=0, ge=0)
    total_budget: int = Field(default=0, ge=0)
    tokens_remaining: int = Field(default=0, ge=0)
    usage_percent: float = Field(default=0.0, ge=0.0)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    circuit_breaker_triggered: bool = False
    circuit_breaker_reason: str = ""
    per_model: list[ModelUsageSummary] = Field(default_factory=list)
    per_category: list[CategoryUsageSummary] = Field(default_factory=list)
    entries_count: int = Field(default=0, ge=0)


class CostTracker:
    def __init__(
        self,
        task_id: str,
        budget: TokenBudget | int | float | None = None,
        config: Any | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._config = config or load_token_budget_config()
        self._task_id = str(task_id or uuid4())
        self._budget = self._resolve_budget(budget)
        self._entries: list[dict[str, Any]] = []
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
    def config(self) -> Any:
        return self._config

    @property
    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._entries)

    @property
    def is_circuit_breaker_triggered(self) -> bool:
        with self._lock:
            return self._circuit_breaker_triggered

    @property
    def circuit_breaker_reason(self) -> str:
        with self._lock:
            return self._circuit_breaker_reason

    @property
    def total_tokens_used(self) -> int:
        return self._tokens_used()

    @property
    def tokens_used(self) -> int:
        return self._tokens_used()

    @property
    def total_spent(self) -> int:
        return self._tokens_used()

    @property
    def spent(self) -> int:
        return self._tokens_used()

    @property
    def total_budget(self) -> int:
        return self._total_budget()

    @property
    def remaining(self) -> int:
        return self._remaining()

    def can_proceed(
        self,
        estimated_tokens: int = 0,
        category: Any = None,
    ) -> bool:
        if self.is_circuit_breaker_triggered:
            return False

        if self._budget_should_stop():
            self._trigger_circuit_breaker("budget_should_stop")
            return False

        estimated = _safe_int(estimated_tokens, 0)

        if estimated <= 0:
            return self._remaining() > 0

        within_budget = getattr(self._budget, "is_within_budget", None)

        if callable(within_budget):
            try:
                return bool(
                    within_budget(
                        requested_tokens=estimated,
                        category=category,
                    )
                )
            except TypeError:
                try:
                    return bool(within_budget(estimated, category))
                except Exception:
                    pass
            except Exception:
                pass

        return self._remaining() >= estimated

    def should_stop(self) -> bool:
        if self.is_circuit_breaker_triggered:
            return True

        if self._budget_should_stop():
            return True

        return self._remaining() <= 0

    def on_token_usage(
        self,
        provider: str = "",
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        latency_ms: float = 0.0,
        **metadata: Any,
    ) -> None:
        resolved_total = _safe_int(total_tokens, 0)

        if resolved_total <= 0:
            resolved_input = _safe_int(input_tokens, 0)
            resolved_output = _safe_int(output_tokens, 0)
            resolved_total = resolved_input + resolved_output

        if resolved_total <= 0:
            return

        self.record_usage(
            provider=str(provider or ""),
            model_id=str(model or ""),
            tokens_used=resolved_total,
            latency_ms=float(latency_ms or 0.0),
            success=True,
            input_tokens=_safe_int(input_tokens, 0),
            output_tokens=_safe_int(output_tokens, 0),
        )

    def record_usage(
        self,
        provider: str = "",
        model_id: str = "",
        tokens_used: int = 0,
        category: Any = None,
        latency_ms: float = 0.0,
        success: bool = True,
        **metadata: Any,
    ) -> bool:
        tokens = _safe_int(tokens_used, 0)

        if tokens <= 0:
            return bool(success)

        entry = {
            "entry_id": str(uuid4()),
            "task_id": self._task_id,
            "provider": str(provider or ""),
            "model_id": str(model_id or ""),
            "tokens_used": tokens,
            "category": str(getattr(category, "value", category) or ""),
            "latency_ms": float(latency_ms or 0.0),
            "success": bool(success),
            "metadata": dict(metadata or {}),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        with self._lock:
            self._entries.append(entry)

        self._consume_budget_tokens(tokens, category)

        self._trace.emit(
            "token_usage_recorded",
            task_id=self._task_id,
            provider=entry["provider"],
            model_id=entry["model_id"],
            tokens_used=tokens,
            category=entry["category"],
            success=entry["success"],
        )

        if self.should_stop():
            self._trigger_circuit_breaker("budget_exhaustion")

        return bool(success)

    def get_usage_percent(self) -> float:
        total = self._total_budget()
        used = self._tokens_used()

        if total <= 0:
            return 100.0 if used > 0 else 0.0

        return min(100.0, (used / total) * 100.0)

    def get_warning_level(self) -> str:
        warning_threshold = _safe_int(
            self._config_value(
                "circuit_breaker",
                "warning_threshold_percent",
                80,
            ),
            80,
        )
        critical_threshold = _safe_int(
            self._config_value(
                "circuit_breaker",
                "critical_threshold_percent",
                95,
            ),
            95,
        )
        percent = self.get_usage_percent()

        if percent >= critical_threshold or self._remaining() <= 0:
            return "critical"

        if percent >= warning_threshold:
            return "warning"

        return "normal"

    def generate_report(self) -> TokenUsageReport:
        per_model: dict[str, dict[str, Any]] = {}
        per_category: dict[str, dict[str, Any]] = {}

        with self._lock:
            entries = list(self._entries)

        total_latency = 0.0
        success_count = 0
        failure_count = 0

        for entry in entries:
            model_key = "/".join(
                part
                for part in (
                    entry.get("provider", ""),
                    entry.get("model_id", ""),
                )
                if part
            ) or "unknown"

            category_key = entry.get("category", "") or "general"
            tokens = _safe_int(entry.get("tokens_used"), 0)
            latency = float(entry.get("latency_ms", 0.0) or 0.0)
            succeeded = bool(entry.get("success", False))

            total_latency += latency

            if succeeded:
                success_count += 1
            else:
                failure_count += 1

            model_summary = per_model.setdefault(
                model_key,
                {
                    "model": model_key,
                    "calls": 0,
                    "tokens_used": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "latency_ms": 0.0,
                },
            )

            model_summary["calls"] += 1
            model_summary["tokens_used"] += tokens
            model_summary["latency_ms"] += latency
            model_summary["success_count"] += 1 if succeeded else 0
            model_summary["failure_count"] += 0 if succeeded else 1

            category_summary = per_category.setdefault(
                category_key,
                {
                    "category": category_key,
                    "calls": 0,
                    "tokens_used": 0,
                    "success_count": 0,
                    "failure_count": 0,
                },
            )

            category_summary["calls"] += 1
            category_summary["tokens_used"] += tokens
            category_summary["success_count"] += 1 if succeeded else 0
            category_summary["failure_count"] += 0 if succeeded else 1

        return TokenUsageReport(
            task_id=self._task_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_calls=len(entries),
            total_tokens_used=self._tokens_used(),
            total_budget=self._total_budget(),
            tokens_remaining=self._remaining(),
            usage_percent=round(self.get_usage_percent(), 2),
            success_count=success_count,
            failure_count=failure_count,
            total_latency_ms=round(total_latency, 2),
            circuit_breaker_triggered=self.is_circuit_breaker_triggered,
            circuit_breaker_reason=self.circuit_breaker_reason,
            per_model=[ModelUsageSummary(**item) for item in per_model.values()],
            per_category=[
                CategoryUsageSummary(**item)
                for item in per_category.values()
            ],
            entries_count=len(entries),
        )

    def save_report(
        self,
        report: TokenUsageReport | dict[str, Any] | None = None,
    ) -> Path | None:
        resolved_report = report or self.generate_report()

        if not bool(
            self._config_value("reporting", "save_report_to_file", True)
        ):
            return None

        try:
            if hasattr(resolved_report, "model_dump"):
                payload = resolved_report.model_dump(mode="json")
            elif isinstance(resolved_report, dict):
                payload = resolved_report
            else:
                payload = json.loads(
                    json.dumps(resolved_report, default=str)
                )

            logs_dir = get_logs_directory()
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            prefix = str(
                self._config_value(
                    "reporting",
                    "report_filename_prefix",
                    type(self).__name__.lower(),
                )
            )

            filename = f"{prefix}_{self._task_id[:8]}_{timestamp}.json"
            report_path = logs_dir / filename

            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(
                    payload,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )

            self._logger.info(f"Token usage report saved to {report_path}")
            return report_path
        except Exception as exc:
            self._logger.warning(
                f"Failed to save token usage report: {exc}"
            )
            return None

    def close(self) -> None:
        try:
            self.save_report()
        except Exception:
            pass

    def _resolve_budget(self, budget: Any) -> TokenBudget:
        if isinstance(budget, TokenBudget):
            return budget

        total_budget = self._normalize_budget_value(budget)

        if total_budget is None:
            return TokenBudget(task_id=self._task_id, config=self._config)

        try:
            return TokenBudget(
                task_id=self._task_id,
                total_budget=total_budget,
                config=self._config,
            )
        except TypeError:
            token_budget = TokenBudget(
                task_id=self._task_id,
                config=self._config,
            )
            self._assign_total_budget(token_budget, total_budget)
            return token_budget
        except Exception:
            token_budget = TokenBudget(
                task_id=self._task_id,
                config=self._config,
            )
            self._assign_total_budget(token_budget, total_budget)
            return token_budget

    def _normalize_budget_value(self, value: Any) -> int | None:
        if value is None:
            return None

        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            return max(0, int(value))

        text = str(value).strip()

        if not text:
            return None

        if text.isdigit():
            return max(0, int(text))

        try:
            return max(0, int(float(text)))
        except Exception:
            return None

    def _assign_total_budget(
        self,
        token_budget: Any,
        total_budget: int,
    ) -> None:
        for method_name in (
            "set_total_budget",
            "configure_total_budget",
            "reset_total_budget",
            "update_total_budget",
        ):
            method = getattr(token_budget, method_name, None)

            if callable(method):
                try:
                    method(total_budget)
                    return
                except Exception:
                    continue

        for attribute_name in (
            "total_budget",
            "_total_budget",
            "budget",
            "_budget",
        ):
            try:
                setattr(token_budget, attribute_name, total_budget)
            except Exception:
                try:
                    object.__setattr__(
                        token_budget,
                        attribute_name,
                        total_budget,
                    )
                except Exception:
                    continue

        for method_name in (
            "reset_remaining",
            "_reset_remaining",
            "recalculate_remaining",
        ):
            method = getattr(token_budget, method_name, None)

            if callable(method):
                try:
                    method()
                except Exception:
                    continue

    def _budget_should_stop(self) -> bool:
        method = getattr(self._budget, "should_stop", None)

        if callable(method):
            try:
                return bool(method())
            except Exception:
                return False

        return False

    def _tokens_used(self) -> int:
        for attribute_name in (
            "spent",
            "total_spent",
            "tokens_used",
            "total_tokens_used",
            "_spent",
        ):
            value = getattr(self._budget, attribute_name, None)

            if value is not None and not callable(value):
                try:
                    return max(0, int(value))
                except Exception:
                    continue

        with self._lock:
            return sum(
                _safe_int(entry.get("tokens_used"), 0)
                for entry in self._entries
            )

    def _total_budget(self) -> int:
        for attribute_name in (
            "total_budget",
            "_total_budget",
            "budget",
            "_budget",
            "total_tokens",
            "_total_tokens",
        ):
            value = getattr(self._budget, attribute_name, None)

            if value is not None and not callable(value):
                try:
                    return max(0, int(value))
                except Exception:
                    continue

        return _safe_int(
            self._config_value("defaults", "total_budget_per_task", 0),
            0,
        )

    def _remaining(self) -> int:
        for attribute_name in (
            "remaining",
            "tokens_remaining",
            "remaining_budget",
            "budget_remaining",
            "_remaining",
        ):
            value = getattr(self._budget, attribute_name, None)

            if value is not None and not callable(value):
                try:
                    return max(0, int(value))
                except Exception:
                    continue

        method = getattr(self._budget, "remaining_tokens", None)

        if callable(method):
            try:
                return max(0, int(method()))
            except Exception:
                pass

        return max(0, self._total_budget() - self._tokens_used())

    def _consume_budget_tokens(self, tokens: int, category: Any) -> None:
        for method_name in (
            "consume",
            "consume_tokens",
            "record_usage",
            "spend",
            "add_usage",
            "deduct",
            "force_deduct",
        ):
            method = getattr(self._budget, method_name, None)

            if not callable(method):
                continue

            attempts = (
                lambda: method(tokens, category=category),
                lambda: method(tokens),
                lambda: method(tokens_used=tokens, category=category),
                lambda: method(tokens_used=tokens),
            )

            for attempt in attempts:
                try:
                    attempt()
                    return
                except TypeError:
                    continue
                except Exception:
                    break

        current = self._tokens_used()

        for attribute_name in (
            "_spent",
            "spent",
            "total_spent",
            "tokens_used",
            "total_tokens_used",
        ):
            try:
                setattr(self._budget, attribute_name, current + tokens)
                return
            except Exception:
                try:
                    object.__setattr__(
                        self._budget,
                        attribute_name,
                        current + tokens,
                    )
                    return
                except Exception:
                    continue

    def _trigger_circuit_breaker(self, reason: str) -> None:
        with self._lock:
            if self._circuit_breaker_triggered:
                return

            self._circuit_breaker_triggered = True
            self._circuit_breaker_reason = str(reason or "")

        self._logger.warning(
            f"Circuit breaker triggered for task {self._task_id}: {reason}"
        )

    def _config_value(
        self,
        section: str,
        key: str,
        default: Any = None,
    ) -> Any:
        config = self._config

        if config is None:
            return default

        container = None

        if isinstance(config, dict):
            container = config.get(section, {})
        else:
            container = getattr(config, section, None)

        if isinstance(container, dict):
            value = container.get(key)

            if value is not None:
                return value
        elif container is not None:
            value = getattr(container, key, None)

            if value is not None:
                return value

        value = getattr(config, f"{section}_{key}", None)

        if value is not None:
            return value

        return default