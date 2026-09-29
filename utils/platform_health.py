from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.logger import get_logger

_STATUS_OK = "ok"
_STATUS_EMPTY = "empty"
_STATUS_RATE_LIMITED = "rate_limited"
_STATUS_FAILED = "failed"
_STATUS_DISABLED = "disabled"

_VALID_STATUSES = frozenset({
    _STATUS_OK,
    _STATUS_EMPTY,
    _STATUS_RATE_LIMITED,
    _STATUS_FAILED,
    _STATUS_DISABLED,
})

_FAILURE_STATUSES = frozenset({
    _STATUS_RATE_LIMITED,
    _STATUS_FAILED,
})

_COOLDOWN_STEPS = (300.0, 600.0, 1200.0, 1800.0)
_MAX_COOLDOWN = _COOLDOWN_STEPS[-1]
_PERSISTED_FILENAME = "platform_health.json"


def _persisted_path() -> Path:
    try:
        from core.config import get_data_directory

        return get_data_directory() / _PERSISTED_FILENAME
    except Exception:
        return Path(_PERSISTED_FILENAME)


def _load_persisted() -> dict[str, float]:
    try:
        path = _persisted_path()

        if not path.exists():
            return {}

        raw = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(raw, dict):
            return {}

        now = time.monotonic()
        loaded: dict[str, float] = {}

        for platform, seconds_remaining in raw.items():
            try:
                remaining = float(seconds_remaining)
            except Exception:
                continue

            if remaining > 0.0:
                loaded[str(platform)] = now + remaining

        return loaded
    except Exception:
        return {}


def _save_persisted(cooldowns: dict[str, float]) -> None:
    try:
        now = time.monotonic()
        payload: dict[str, float] = {}

        for platform, until in cooldowns.items():
            remaining = until - now

            if remaining <= 0.0:
                continue

            capped = min(remaining, _MAX_COOLDOWN)
            payload[str(platform)] = round(capped, 2)

        path = _persisted_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


@dataclass
class PlatformHealthRecord:
    platform: str
    last_status: str = _STATUS_OK
    last_error: str | None = None
    last_source_count: int = 0
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    cooldown_until: float = 0.0
    last_updated: float = field(default_factory=time.monotonic)


class PlatformHealthTracker:
    _instance: PlatformHealthTracker | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._records: dict[str, PlatformHealthRecord] = {}
        self._disabled_platforms: set[str] = set()
        self._warnings: list[str] = []
        self._state_lock = threading.RLock()
        self._logger = get_logger("utils.platform_health")
        self._persisted_cooldowns: dict[str, float] = _load_persisted()

    @classmethod
    def get_instance(cls) -> PlatformHealthTracker:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def create_fresh(cls) -> PlatformHealthTracker:
        return cls()

    def reset_for_new_run(self) -> None:
        with self._state_lock:
            self._records.clear()
            self._warnings.clear()

    def mark_degraded(
        self,
        platform: str,
        reason: str,
        duration_seconds: float | None = None,
    ) -> None:
        normalized = self._normalize(platform)

        if not normalized:
            return

        with self._state_lock:
            record = self._get_or_create(normalized)
            record.consecutive_failures += 1
            record.total_failures += 1
            record.last_status = _STATUS_FAILED
            record.last_error = str(reason or "")
            record.last_updated = time.monotonic()

            if duration_seconds is None:
                cooldown = self._cooldown_for(
                    record.consecutive_failures
                )
            else:
                try:
                    cooldown = max(0.0, float(duration_seconds))
                except Exception:
                    cooldown = self._cooldown_for(
                        record.consecutive_failures
                    )

            now = time.monotonic()
            record.cooldown_until = max(
                record.cooldown_until, now + cooldown
            )

            self._persisted_cooldowns[normalized] = (
                record.cooldown_until
            )

            warning = (
                f"platform_degraded:{normalized}:"
                f"{reason}:cooldown={cooldown:.0f}s"
            )

            if warning not in self._warnings:
                self._warnings.append(warning)

        _save_persisted(self._persisted_cooldowns)

        self._logger.warning(
            f"Platform {normalized} marked degraded for "
            f"{cooldown:.0f}s: {reason}"
        )

    def mark_disabled(self, platform: str, reason: str) -> None:
        normalized = self._normalize(platform)

        if not normalized:
            return

        with self._state_lock:
            self._disabled_platforms.add(normalized)
            record = self._get_or_create(normalized)
            record.last_status = _STATUS_DISABLED
            record.last_error = str(reason or "")
            record.last_updated = time.monotonic()

            warning = f"platform_disabled:{normalized}:{reason}"

            if warning not in self._warnings:
                self._warnings.append(warning)

    def record_result(
        self,
        platform: str,
        status: str,
        source_count: int = 0,
        error: str | None = None,
    ) -> None:
        normalized = self._normalize(platform)

        if not normalized:
            return

        try:
            count = max(0, int(source_count))
        except Exception:
            count = 0

        normalized_status = self._normalize_status(status)

        if normalized_status == _STATUS_OK and count <= 0:
            normalized_status = _STATUS_EMPTY

        should_persist = False

        with self._state_lock:
            record = self._get_or_create(normalized)
            record.last_status = normalized_status
            record.last_source_count = count
            record.last_updated = time.monotonic()

            if error:
                record.last_error = str(error)
            elif normalized_status in {_STATUS_OK, _STATUS_EMPTY}:
                record.last_error = None

            if normalized_status == _STATUS_OK:
                record.consecutive_failures = 0
                record.total_successes += 1
                record.cooldown_until = 0.0
                self._persisted_cooldowns.pop(normalized, None)
                should_persist = True
            elif normalized_status == _STATUS_EMPTY:
                record.cooldown_until = 0.0
            elif normalized_status == _STATUS_DISABLED:
                self._disabled_platforms.add(normalized)

                warning = (
                    f"platform_disabled:{normalized}:"
                    f"{error or 'disabled'}"
                )

                if warning not in self._warnings:
                    self._warnings.append(warning)
            elif normalized_status in _FAILURE_STATUSES:
                record.consecutive_failures += 1
                record.total_failures += 1

                cooldown = self._cooldown_for(
                    record.consecutive_failures
                )
                now = time.monotonic()
                record.cooldown_until = max(
                    record.cooldown_until, now + cooldown
                )

                self._persisted_cooldowns[normalized] = (
                    record.cooldown_until
                )
                should_persist = True

                warning = (
                    f"platform_{normalized_status}:{normalized}:"
                    f"cooldown={cooldown:.0f}s"
                )

                if warning not in self._warnings:
                    self._warnings.append(warning)

        if should_persist:
            _save_persisted(self._persisted_cooldowns)

    def record_success(
        self, platform: str, source_count: int = 0
    ) -> None:
        self.record_result(
            platform, _STATUS_OK, source_count=source_count
        )

    def record_failure(
        self,
        platform: str,
        error: str | None = None,
        rate_limited: bool = False,
    ) -> None:
        status = (
            _STATUS_RATE_LIMITED if rate_limited else _STATUS_FAILED
        )
        self.record_result(
            platform, status, source_count=0, error=error
        )

    def record_empty(self, platform: str) -> None:
        self.record_result(platform, _STATUS_EMPTY, source_count=0)

    def is_usable(self, platform: str) -> bool:
        normalized = self._normalize(platform)

        if not normalized:
            return False

        now = time.monotonic()

        with self._state_lock:
            if normalized in self._disabled_platforms:
                return False

            persisted_until = self._persisted_cooldowns.get(
                normalized, 0.0
            )

            if persisted_until and now < persisted_until:
                return False

            if persisted_until and now >= persisted_until:
                self._persisted_cooldowns.pop(normalized, None)

            record = self._records.get(normalized)

            if record is None:
                return True

            if record.cooldown_until and now < record.cooldown_until:
                return False

            if (
                record.cooldown_until
                and now >= record.cooldown_until
            ):
                record.cooldown_until = 0.0

            return True

    def get_cooldown_remaining(self, platform: str) -> float:
        normalized = self._normalize(platform)

        if not normalized:
            return 0.0

        now = time.monotonic()

        with self._state_lock:
            persisted_remaining = max(
                0.0,
                self._persisted_cooldowns.get(normalized, 0.0)
                - now,
            )

            record = self._records.get(normalized)

            if record is None:
                return persisted_remaining

            record_remaining = max(
                0.0, record.cooldown_until - now
            )

            return max(persisted_remaining, record_remaining)

    def get_record(self, platform: str) -> dict[str, Any] | None:
        normalized = self._normalize(platform)

        if not normalized:
            return None

        with self._state_lock:
            record = self._records.get(normalized)

            if record is None:
                persisted_remaining = max(
                    0.0,
                    self._persisted_cooldowns.get(normalized, 0.0)
                    - time.monotonic(),
                )

                if persisted_remaining > 0.0:
                    return {
                        "platform": normalized,
                        "last_status": _STATUS_FAILED,
                        "last_error": "persisted_cooldown",
                        "last_source_count": 0,
                        "consecutive_failures": 0,
                        "total_failures": 0,
                        "total_successes": 0,
                        "cooldown_remaining_seconds": round(
                            persisted_remaining, 2
                        ),
                        "last_updated_seconds_ago": 0.0,
                    }

                return None

            return self._serialize_record(record, time.monotonic())

    def get_warnings(self) -> list[str]:
        with self._state_lock:
            return list(self._warnings)

    def get_report(self) -> dict[str, Any]:
        now = time.monotonic()
        per_platform: list[dict[str, Any]] = []

        with self._state_lock:
            all_keys = set(self._records.keys()) | set(
                self._persisted_cooldowns.keys()
            )

            for platform in all_keys:
                record = self._records.get(platform)

                if record is not None:
                    entry = self._serialize_record(record, now)
                else:
                    entry = {
                        "platform": platform,
                        "last_status": _STATUS_FAILED,
                        "last_error": "persisted_cooldown",
                        "last_source_count": 0,
                        "consecutive_failures": 0,
                        "total_failures": 0,
                        "total_successes": 0,
                        "cooldown_remaining_seconds": round(
                            max(
                                0.0,
                                self._persisted_cooldowns.get(
                                    platform, 0.0
                                )
                                - now,
                            ),
                            2,
                        ),
                        "last_updated_seconds_ago": 0.0,
                    }

                entry["disabled"] = (
                    platform in self._disabled_platforms
                )
                entry["usable"] = self._is_usable_locked(
                    platform, now
                )
                per_platform.append(entry)

            warnings = list(self._warnings)
            disabled_count = len(self._disabled_platforms)

        usable_count = sum(
            1 for item in per_platform if item["usable"]
        )
        cooling_count = sum(
            1
            for item in per_platform
            if item["cooldown_remaining_seconds"] > 0.0
        )
        failed_count = sum(
            1
            for item in per_platform
            if item["last_status"] in _FAILURE_STATUSES
        )

        return {
            "per_platform": per_platform,
            "warnings": warnings,
            "totals": {
                "tracked_platforms": len(per_platform),
                "usable_platforms": usable_count,
                "cooling_platforms": cooling_count,
                "disabled_platforms": disabled_count,
                "failed_platforms": failed_count,
            },
        }

    def clear(self) -> None:
        with self._state_lock:
            self._records.clear()
            self._disabled_platforms.clear()
            self._warnings.clear()
            self._persisted_cooldowns.clear()

        _save_persisted(self._persisted_cooldowns)

    def _is_usable_locked(
        self, platform: str, now: float
    ) -> bool:
        if platform in self._disabled_platforms:
            return False

        persisted_until = self._persisted_cooldowns.get(
            platform, 0.0
        )

        if persisted_until and now < persisted_until:
            return False

        record = self._records.get(platform)

        if record is None:
            return True

        if record.cooldown_until and now < record.cooldown_until:
            return False

        return True

    def _get_or_create(
        self, platform: str
    ) -> PlatformHealthRecord:
        record = self._records.get(platform)

        if record is None:
            record = PlatformHealthRecord(platform=platform)
            self._records[platform] = record

        return record

    @staticmethod
    def _serialize_record(
        record: PlatformHealthRecord,
        now: float,
    ) -> dict[str, Any]:
        cooldown_remaining = max(
            0.0, record.cooldown_until - now
        )

        return {
            "platform": record.platform,
            "last_status": record.last_status,
            "last_error": record.last_error,
            "last_source_count": record.last_source_count,
            "consecutive_failures": record.consecutive_failures,
            "total_failures": record.total_failures,
            "total_successes": record.total_successes,
            "cooldown_remaining_seconds": round(
                cooldown_remaining, 2
            ),
            "last_updated_seconds_ago": round(
                max(0.0, now - record.last_updated), 2
            ),
        }

    @staticmethod
    def _cooldown_for(consecutive_failures: int) -> float:
        if consecutive_failures <= 0:
            return _COOLDOWN_STEPS[0]

        idx = min(
            consecutive_failures - 1, len(_COOLDOWN_STEPS) - 1
        )
        return _COOLDOWN_STEPS[idx]

    @staticmethod
    def _normalize(platform: str) -> str:
        return str(platform or "").strip().lower()

    @staticmethod
    def _normalize_status(status: str) -> str:
        text = str(status or "").strip().lower()

        if text in _VALID_STATUSES:
            return text

        return _STATUS_FAILED