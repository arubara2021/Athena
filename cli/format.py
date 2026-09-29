from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _is_finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def format_bytes(value: Any) -> str:
    size = _to_float(value)
    if size is None or size < 0 or not _is_finite(size):
        return "-"
    units = ("B", "KB", "MB", "GB", "TB", "PB", "EB")
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(size)} B"
    return f"{size:.2f} {units[idx]}"


def format_latency(value: Any) -> str:
    ms = _to_float(value)
    if ms is None or ms < 0 or not _is_finite(ms):
        return "-"
    if ms < 1.0:
        return "<1 ms"
    if ms < 1000.0:
        return f"{ms:.0f} ms"
    if ms < 60_000.0:
        return f"{ms / 1000.0:.2f} s"
    minutes = int(ms // 60_000)
    seconds = (ms % 60_000) / 1000.0
    if minutes < 60:
        return f"{minutes}m {seconds:04.1f}s"
    hours = minutes // 60
    minutes_rem = minutes % 60
    return f"{hours}h {minutes_rem:02d}m"


def format_duration(seconds: Any) -> str:
    s = _to_float(seconds)
    if s is None or s < 0 or not _is_finite(s):
        return "-"
    return format_latency(s * 1000.0)


def format_count(value: Any) -> str:
    n = _to_int(value)
    if n is None:
        return "-"
    sign = "-" if n < 0 else ""
    a = abs(n)
    if a >= 1_000_000_000:
        return f"{sign}{a / 1_000_000_000:.2f}B"
    if a >= 1_000_000:
        return f"{sign}{a / 1_000_000:.2f}M"
    if a >= 10_000:
        return f"{sign}{a / 1_000:.1f}K"
    return f"{sign}{a}"


def format_float(value: Any, places: int = 2) -> str:
    f = _to_float(value)
    if f is None or not _is_finite(f):
        return "-"
    try:
        p = max(0, min(6, int(places)))
    except (TypeError, ValueError):
        p = 2
    return f"{f:.{p}f}"


def format_percent(value: Any, places: int = 1) -> str:
    f = _to_float(value)
    if f is None or not _is_finite(f):
        return "-"
    try:
        p = max(0, min(4, int(places)))
    except (TypeError, ValueError):
        p = 1
    return f"{f:.{p}f}%"


def format_ratio(value: Any, places: int = 0) -> str:
    f = _to_float(value)
    if f is None or not _is_finite(f):
        return "-"
    try:
        p = max(0, min(4, int(places)))
    except (TypeError, ValueError):
        p = 0
    return f"{f * 100:.{p}f}%"


def ellipsis(value: Any, limit: int = 80) -> str:
    text = str(value or "")
    try:
        cap = int(limit)
    except (TypeError, ValueError):
        return text
    if cap <= 0:
        return ""
    if len(text) <= cap:
        return text
    if cap <= 3:
        return text[:cap]
    return text[: cap - 3].rstrip() + "..."


def truncate(value: Any, limit: int = 80, suffix: str = "...") -> str:
    text = str(value or "")
    try:
        cap = int(limit)
    except (TypeError, ValueError):
        return text
    if cap <= 0:
        return ""
    if len(text) <= cap:
        return text
    tail = str(suffix or "")
    if len(tail) >= cap:
        return text[:cap]
    return text[: cap - len(tail)].rstrip() + tail


def _to_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def format_timestamp(value: Any) -> str:
    dt = _to_datetime(value)
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_short_timestamp(value: Any) -> str:
    dt = _to_datetime(value)
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M")


def format_clock(value: Any) -> str:
    dt = _to_datetime(value)
    if dt is None:
        return "-"
    return dt.strftime("%H:%M:%S")


def score_band(value: Any) -> str:
    f = _to_float(value)
    if f is None:
        return "unknown"
    if f >= 0.75:
        return "high"
    if f >= 0.50:
        return "medium"
    if f >= 0.25:
        return "low"
    return "very_low"


def confidence_band(value: Any) -> str:
    f = _to_float(value)
    if f is None:
        return "unknown"
    if f >= 0.75:
        return "high"
    if f >= 0.50:
        return "medium"
    if f >= 0.20:
        return "low"
    return "very_low"


def format_plural(count: Any, singular: str, plural: str | None = None) -> str:
    n = _to_int(count)
    if n is None:
        n = 0
    word = singular if n == 1 else (plural or f"{singular}s")
    return f"{n} {word}"


def format_enabled(value: Any) -> str:
    return "yes" if bool(value) else "no"


def clip_lines(value: Any, max_lines: int = 3) -> str:
    text = str(value or "")
    if not text:
        return ""
    lines = text.splitlines()
    try:
        cap = max(1, int(max_lines))
    except (TypeError, ValueError):
        cap = 3
    if len(lines) <= cap:
        return text
    return "\n".join(lines[:cap] + ["..."])


def safe_str(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default