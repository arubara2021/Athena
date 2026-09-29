from __future__ import annotations

import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from core import constants
from core.exceptions import ResearchAgentError
from core.models import CoreModel
from utils.logger import get_logger


class TokenBudgetExceededError(ResearchAgentError):
    pass


class TokenBudgetWarningError(ResearchAgentError):
    pass


class TokenCategory(str, Enum):
    PLANNING = "planning"
    QUERY_EXPANSION = "query_expansion"
    SEARCH = "search"
    READING = "reading"
    RANKING = "ranking"
    SUMMARIZATION = "summarization"
    LEARNING_PATH = "learning_path"
    PATH_ENHANCEMENT = "path_enhancement"
    REFLECTION = "reflection"
    JUDGE = "judge"


class ModelTier(str, Enum):
    FAST = "fast"
    STRONG = "strong"
    JUDGE = "judge"


class CategoryBudget(CoreModel):
    max_tokens: int = Field(ge=0)
    priority: int = Field(default=0, ge=0)
    model_tier: ModelTier = ModelTier.FAST


class TierConfig(CoreModel):
    cost_weight: float = Field(default=1.0, ge=0.0)
    max_tokens_per_call: int = Field(default=2048, ge=1)
    preferred_providers: list[str] = Field(default_factory=list)


class CircuitBreakerConfig(CoreModel):
    enabled: bool = True
    warning_threshold_percent: float = Field(default=80.0, ge=0.0, le=100.0)
    critical_threshold_percent: float = Field(default=95.0, ge=0.0, le=100.0)
    cooldown_seconds: float = Field(default=0.0, ge=0.0)
    max_consecutive_exceedances: int = Field(default=3, ge=1)


class LedgerConfig(CoreModel):
    filename: str = "token_ledger.jsonl"
    flush_interval_entries: int = Field(default=10, ge=1)
    max_entries_in_memory: int = Field(default=10000, ge=1)
    include_raw_response: bool = False
    include_prompt_preview_length: int = Field(default=100, ge=0)


class ReportingConfig(CoreModel):
    include_per_model_breakdown: bool = True
    include_per_category_breakdown: bool = True
    include_timeline: bool = False
    save_report_to_file: bool = True
    report_filename_prefix: str = "token_usage_report"


class DefaultBudgetConfig(CoreModel):
    total_budget_per_task: int = Field(
        default=constants.AGENT_DEFAULT_TOKEN_BUDGET,
        ge=1,
    )
    wrap_up_threshold: int = Field(default=8000, ge=0)
    currency: str = "tokens"
    allow_burst: bool = False
    burst_multiplier: float = Field(default=1.15, ge=1.0)


class ModeBudgetConfig(CoreModel):
    total_budget_per_task: int = Field(ge=1)
    wrap_up_threshold: int = Field(default=0, ge=0)


class TokenBudgetConfig(CoreModel):
    defaults: DefaultBudgetConfig = Field(default_factory=DefaultBudgetConfig)
    modes: dict[str, ModeBudgetConfig] = Field(default_factory=dict)
    categories: dict[str, CategoryBudget] = Field(default_factory=dict)
    model_tiers: dict[str, TierConfig] = Field(default_factory=dict)
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig
    )
    ledger: LedgerConfig = Field(default_factory=LedgerConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)


_CONFIG_CACHE: TokenBudgetConfig | None = None
_CONFIG_LOCK = threading.Lock()


def load_token_budget_config(
    config_path: Path | str | None = None,
) -> TokenBudgetConfig:
    global _CONFIG_CACHE

    with _CONFIG_LOCK:
        if _CONFIG_CACHE is not None and config_path is None:
            return _CONFIG_CACHE

        resolved_path = _resolve_config_path(config_path)
        config = _load_from_yaml(resolved_path)

        if config_path is None:
            _CONFIG_CACHE = config

        return config


def reload_token_budget_config(
    config_path: Path | str | None = None,
) -> TokenBudgetConfig:
    global _CONFIG_CACHE

    with _CONFIG_LOCK:
        _CONFIG_CACHE = None

    return load_token_budget_config(config_path)


def _resolve_config_path(config_path: Path | str | None) -> Path:
    if config_path is not None:
        return Path(config_path)

    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "token_budgets.yaml"
    )


def _load_from_yaml(path: Path) -> TokenBudgetConfig:
    logger = get_logger("agent.token.budget")

    if not path.exists():
        logger.warning(
            f"Token budget config not found at {path}, using defaults"
        )
        return TokenBudgetConfig()

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except Exception as exc:
        logger.warning(
            f"Failed to parse token budget config: {exc}, using defaults"
        )
        return TokenBudgetConfig()

    if not isinstance(raw, dict):
        return TokenBudgetConfig()

    try:
        defaults = DefaultBudgetConfig.model_validate(
            raw.get("defaults", {})
        )

        modes: dict[str, ModeBudgetConfig] = {}

        raw_modes = raw.get("modes", {})

        if isinstance(raw_modes, dict):
            for key, value in raw_modes.items():
                if not isinstance(value, dict):
                    continue

                try:
                    modes[str(key).strip().lower()] = (
                        ModeBudgetConfig.model_validate(value)
                    )
                except Exception:
                    continue

        categories: dict[str, CategoryBudget] = {}

        raw_categories = raw.get("categories", {})

        if isinstance(raw_categories, dict):
            for key, value in raw_categories.items():
                if not isinstance(value, dict):
                    continue

                try:
                    categories[str(key)] = CategoryBudget.model_validate(
                        value
                    )
                except Exception:
                    continue

        model_tiers: dict[str, TierConfig] = {}

        raw_tiers = raw.get("model_tiers", {})

        if isinstance(raw_tiers, dict):
            for key, value in raw_tiers.items():
                if not isinstance(value, dict):
                    continue

                try:
                    model_tiers[str(key)] = TierConfig.model_validate(value)
                except Exception:
                    continue

        circuit_breaker = CircuitBreakerConfig.model_validate(
            raw.get("circuit_breaker", {})
        )
        ledger = LedgerConfig.model_validate(raw.get("ledger", {}))
        reporting = ReportingConfig.model_validate(
            raw.get("reporting", {})
        )

        return TokenBudgetConfig(
            defaults=defaults,
            modes=modes,
            categories=categories,
            model_tiers=model_tiers,
            circuit_breaker=circuit_breaker,
            ledger=ledger,
            reporting=reporting,
        )
    except Exception as exc:
        logger.warning(
            f"Failed to validate token budget config: {exc}, using defaults"
        )
        return TokenBudgetConfig()


class TokenBudget:
    def __init__(
        self,
        task_id: str,
        config: TokenBudgetConfig | None = None,
        total_budget: int | None = None,
        mode: str | None = None,
    ) -> None:
        self._config = config or load_token_budget_config()
        self._task_id = task_id
        self._mode = self._normalize_mode(mode)

        mode_config = self._config.modes.get(self._mode)

        if mode_config is not None:
            resolved_budget = (
                mode_config.total_budget_per_task
                if total_budget is None
                else total_budget
            )
            resolved_threshold = mode_config.wrap_up_threshold
        else:
            resolved_budget = (
                self._config.defaults.total_budget_per_task
                if total_budget is None
                else total_budget
            )
            resolved_threshold = self._config.defaults.wrap_up_threshold

        self._total_budget = max(1, int(resolved_budget))
        self._wrap_up_threshold = max(0, int(resolved_threshold))

        self._spent: int = 0
        self._category_spent: dict[str, int] = {}
        self._lock = threading.Lock()

        self._wrap_up_mode: bool = False
        self._synthesize_only: bool = False
        self._consecutive_exceedances: int = 0

        self._created_at = datetime.now(timezone.utc)
        self._logger = get_logger("agent.token.budget")

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def total_budget(self) -> int:
        return self._total_budget

    @property
    def wrap_up_threshold(self) -> int:
        return self._wrap_up_threshold

    @property
    def spent(self) -> int:
        with self._lock:
            return self._spent

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self._total_budget - self._spent)

    @property
    def is_wrap_up_mode(self) -> bool:
        with self._lock:
            return self._wrap_up_mode

    @property
    def is_synthesize_only(self) -> bool:
        with self._lock:
            return self._synthesize_only

    @property
    def category_spent(self) -> dict[str, int]:
        with self._lock:
            return dict(self._category_spent)

    @property
    def config(self) -> TokenBudgetConfig:
        return self._config

    def set_mode(self, mode: str) -> None:
        normalized = self._normalize_mode(mode)

        with self._lock:
            self._mode = normalized

        self._logger.info(
            f"TokenBudget mode switched to '{normalized}' for task "
            f"{self._task_id}"
        )

    def deduct(
        self,
        tokens: int,
        category: str | TokenCategory | None = None,
        model_id: str = "",
    ) -> bool:
        if tokens < 0:
            raise ValueError(
                "Token deduction amount cannot be negative"
            )

        if tokens == 0:
            return True

        category_key = self._resolve_category_key(category)

        with self._lock:
            if self._spent + tokens > self._total_budget:
                if self._config.defaults.allow_burst:
                    burst_limit = int(
                        self._total_budget
                        * self._config.defaults.burst_multiplier
                    )

                    if self._spent + tokens > burst_limit:
                        self._consecutive_exceedances += 1
                        self._logger.warning(
                            f"Token budget exceeded (with burst) for "
                            f"task {self._task_id}: requested={tokens}, "
                            f"spent={self._spent}, "
                            f"total={self._total_budget}, "
                            f"burst_limit={burst_limit}"
                        )
                        return False
                else:
                    self._consecutive_exceedances += 1
                    self._logger.warning(
                        f"Token budget exceeded for task "
                        f"{self._task_id}: requested={tokens}, "
                        f"spent={self._spent}, "
                        f"total={self._total_budget}"
                    )
                    return False

            if category_key:
                category_config = self._config.categories.get(
                    category_key
                )

                if category_config is not None:
                    current_category_spent = self._category_spent.get(
                        category_key, 0
                    )

                    if (
                        current_category_spent + tokens
                        > category_config.max_tokens
                    ):
                        self._logger.warning(
                            f"Category budget exceeded for "
                            f"'{category_key}' in task "
                            f"{self._task_id}: requested={tokens}, "
                            f"category_spent={current_category_spent}, "
                            f"category_max={category_config.max_tokens}"
                        )
                        return False

            self._spent += tokens
            self._consecutive_exceedances = 0

            if category_key:
                self._category_spent[category_key] = (
                    self._category_spent.get(category_key, 0) + tokens
                )

            self._check_wrap_up_transition()

        return True

    def force_deduct(
        self,
        tokens: int,
        category: str | TokenCategory | None = None,
    ) -> None:
        if tokens < 0:
            raise ValueError(
                "Token deduction amount cannot be negative"
            )

        category_key = self._resolve_category_key(category)

        with self._lock:
            self._spent += tokens

            if category_key:
                self._category_spent[category_key] = (
                    self._category_spent.get(category_key, 0) + tokens
                )

            self._check_wrap_up_transition()

    def is_within_budget(
        self,
        requested_tokens: int = 0,
        category: str | TokenCategory | None = None,
    ) -> bool:
        category_key = self._resolve_category_key(category)

        with self._lock:
            if self._spent + requested_tokens > self._total_budget:
                return False

            if category_key:
                category_config = self._config.categories.get(
                    category_key
                )

                if category_config is not None:
                    current_spent = self._category_spent.get(
                        category_key, 0
                    )

                    if (
                        current_spent + requested_tokens
                        > category_config.max_tokens
                    ):
                        return False

        return True

    def get_remaining(
        self,
        category: str | TokenCategory | None = None,
    ) -> int:
        category_key = self._resolve_category_key(category)

        with self._lock:
            if category_key:
                category_config = self._config.categories.get(
                    category_key
                )

                if category_config is not None:
                    current_spent = self._category_spent.get(
                        category_key, 0
                    )
                    return max(
                        0,
                        category_config.max_tokens - current_spent,
                    )

            return max(0, self._total_budget - self._spent)

    def get_usage_percent(self) -> float:
        with self._lock:
            if self._total_budget <= 0:
                return 100.0

            return min(
                100.0,
                (self._spent / self._total_budget) * 100.0,
            )

    def get_warning_level(self) -> str:
        percent = self.get_usage_percent()
        breaker = self._config.circuit_breaker

        if percent >= breaker.critical_threshold_percent:
            return "critical"

        if percent >= breaker.warning_threshold_percent:
            return "warning"

        return "normal"

    def should_stop(self) -> bool:
        if not self._config.circuit_breaker.enabled:
            return False

        with self._lock:
            return (
                self._consecutive_exceedances
                >= self._config.circuit_breaker.max_consecutive_exceedances
            )

    def should_stop_exploring(self) -> bool:
        with self._lock:
            return self._wrap_up_mode or self._synthesize_only

    def activate_wrap_up_mode(self) -> None:
        with self._lock:
            if self._wrap_up_mode:
                return

            self._wrap_up_mode = True
            self._synthesize_only = True
            self._logger.info(
                f"Wrap-up mode activated for task {self._task_id}. "
                f"Remaining total budget: "
                f"{max(0, self._total_budget - self._spent)} tokens. "
                f"Synthesize from accumulated material only."
            )

    def to_summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "task_id": self._task_id,
                "mode": self._mode,
                "total_budget": self._total_budget,
                "spent": self._spent,
                "remaining": max(0, self._total_budget - self._spent),
                "usage_percent": self.get_usage_percent(),
                "wrap_up_mode": self._wrap_up_mode,
                "synthesize_only": self._synthesize_only,
                "wrap_up_threshold": self._wrap_up_threshold,
                "warning_level": self.get_warning_level(),
                "category_spent": dict(self._category_spent),
                "created_at": self._created_at.isoformat(),
            }

    def _check_wrap_up_transition(self) -> None:
        if self._wrap_up_mode:
            return

        remaining = self._total_budget - self._spent

        if remaining <= self._wrap_up_threshold:
            self._wrap_up_mode = True
            self._synthesize_only = True

            self._logger.info(
                f"Wrap-up mode auto-activated for task {self._task_id}. "
                f"Spent={self._spent}, remaining={remaining}, "
                f"threshold={self._wrap_up_threshold}"
            )

    def _resolve_category_key(
        self,
        category: str | TokenCategory | None,
    ) -> str:
        if category is None:
            return ""

        if isinstance(category, TokenCategory):
            return category.value

        return str(category).strip().lower()

    def _normalize_mode(self, mode: str | None) -> str:
        text = str(mode or constants.DEFAULT_MODE).strip().lower()

        if text in constants.VALID_MODES:
            return text

        return constants.DEFAULT_MODE