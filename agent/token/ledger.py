from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field

from core.config import get_logs_directory
from core.models import CoreModel
from utils.logger import get_logger
from agent.token.budget import ModelTier, TokenCategory, load_token_budget_config


class TokenLedgerEntry(CoreModel):
    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    provider: str
    model_id: str
    category: str = ""
    model_tier: str = ModelTier.FAST.value
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: str | None = None
    prompt_preview: str = ""
    budget_spent_after: int = 0
    budget_remaining_after: int = 0


class TokenLedger:
    def __init__(
        self,
        task_id: str,
        log_path: Path | str | None = None,
        config: Any | None = None,
    ) -> None:
        self._config = config or load_token_budget_config()
        self._task_id = task_id
        self._entries: list[TokenLedgerEntry] = []
        self._lock = threading.Lock()
        self._logger = get_logger("agent.token.ledger")

        if log_path is not None:
            self._log_path = Path(log_path)
        else:
            self._log_path = (
                get_logs_directory() / self._config.ledger.filename
            )

        self._flush_interval = self._config.ledger.flush_interval_entries
        self._max_in_memory = self._config.ledger.max_entries_in_memory
        self._unflushed_count = 0

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def total_tokens_recorded(self) -> int:
        with self._lock:
            return sum(entry.total_tokens for entry in self._entries)

    def record(
        self,
        provider: str,
        model_id: str,
        category: str | TokenCategory | None = None,
        model_tier: str | ModelTier | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
        latency_ms: float = 0.0,
        success: bool = True,
        error: str | None = None,
        prompt_preview: str = "",
        budget_spent_after: int = 0,
        budget_remaining_after: int = 0,
    ) -> TokenLedgerEntry:
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

        resolved_total = total_tokens
        if resolved_total is None:
            resolved_total = prompt_tokens + completion_tokens

        preview = ""
        if self._config.ledger.include_prompt_preview_length > 0 and prompt_preview:
            max_len = self._config.ledger.include_prompt_preview_length
            preview = prompt_preview[:max_len]

        entry = TokenLedgerEntry(
            task_id=self._task_id,
            provider=provider,
            model_id=model_id,
            category=resolved_category,
            model_tier=resolved_tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=resolved_total,
            latency_ms=latency_ms,
            success=success,
            error=error,
            prompt_preview=preview,
            budget_spent_after=budget_spent_after,
            budget_remaining_after=budget_remaining_after,
        )

        with self._lock:
            self._entries.append(entry)
            self._unflushed_count += 1

            if len(self._entries) > self._max_in_memory:
                overflow = len(self._entries) - self._max_in_memory
                self._entries = self._entries[overflow:]

        if self._unflushed_count >= self._flush_interval:
            self.flush()

        return entry

    def record_from_response(
        self,
        response: Any,
        category: str | TokenCategory | None = None,
        model_tier: str | ModelTier | None = None,
        budget_spent_after: int = 0,
        budget_remaining_after: int = 0,
    ) -> TokenLedgerEntry:
        provider = ""
        model_id = ""
        tokens_used = 0
        latency_ms = 0.0

        if hasattr(response, "provider"):
            provider = str(getattr(response.provider, "value", response.provider))
        if hasattr(response, "model"):
            model_id = str(response.model)
        if hasattr(response, "tokens_used") and response.tokens_used is not None:
            tokens_used = int(response.tokens_used)
        if hasattr(response, "latency_ms") and response.latency_ms is not None:
            latency_ms = float(response.latency_ms)

        return self.record(
            provider=provider,
            model_id=model_id,
            category=category,
            model_tier=model_tier,
            total_tokens=tokens_used,
            latency_ms=latency_ms,
            success=True,
            budget_spent_after=budget_spent_after,
            budget_remaining_after=budget_remaining_after,
        )

    def flush(self) -> int:
        with self._lock:
            entries_to_flush = list(self._entries)
            self._unflushed_count = 0

        if not entries_to_flush:
            return 0

        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as handle:
                for entry in entries_to_flush:
                    handle.write(
                        json.dumps(entry.model_dump(mode="json"), ensure_ascii=False)
                        + "\n"
                    )
            return len(entries_to_flush)
        except Exception as exc:
            self._logger.warning(f"Failed to flush token ledger: {exc}")
            return 0

    def get_entries(
        self,
        category: str | None = None,
        provider: str | None = None,
        success_only: bool = False,
    ) -> list[TokenLedgerEntry]:
        with self._lock:
            entries = list(self._entries)

        if category:
            entries = [e for e in entries if e.category == category.lower()]
        if provider:
            entries = [e for e in entries if e.provider == provider]
        if success_only:
            entries = [e for e in entries if e.success]

        return entries

    def get_total_tokens(self) -> int:
        with self._lock:
            return sum(e.total_tokens for e in self._entries)

    def get_tokens_by_category(self) -> dict[str, int]:
        with self._lock:
            result: dict[str, int] = {}
            for entry in self._entries:
                key = entry.category or "uncategorized"
                result[key] = result.get(key, 0) + entry.total_tokens
            return result

    def get_tokens_by_model(self) -> dict[str, int]:
        with self._lock:
            result: dict[str, int] = {}
            for entry in self._entries:
                key = f"{entry.provider}/{entry.model_id}"
                result[key] = result.get(key, 0) + entry.total_tokens
            return result

    def get_tokens_by_tier(self) -> dict[str, int]:
        with self._lock:
            result: dict[str, int] = {}
            for entry in self._entries:
                result[entry.model_tier] = (
                    result.get(entry.model_tier, 0) + entry.total_tokens
                )
            return result

    def close(self) -> None:
        self.flush()