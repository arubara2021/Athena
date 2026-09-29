from __future__ import annotations

from typing import Any, Iterable, Sequence

from rich import box
from rich.align import Align
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli import format as fmt


def _coerce_cell(value: Any, default_style: str = "athena.value") -> Text:
    if isinstance(value, Text):
        return value
    if value is None:
        return Text("-", style="athena.muted")
    return Text(str(value), style=default_style)


def panel_title(text: str, glyph: str = "") -> Text:
    label = str(text or "").strip().upper()
    if glyph:
        label = f"{glyph}  {label}"
    else:
        label = f" {label} "
    return Text(label, style="athena.heading")


def panel(
    body: RenderableType,
    title: str = "",
    border: str = "athena.border",
    padding: tuple[int, int] = (1, 2),
    subtitle: str = "",
) -> Panel:
    kwargs: dict[str, Any] = {
        "border_style": border,
        "box": box.ROUNDED,
        "padding": padding,
    }
    if title:
        kwargs["title"] = panel_title(title)
    if subtitle:
        kwargs["subtitle"] = Text(str(subtitle), style="athena.muted")
    return Panel(body, **kwargs)


def kv_grid(
    rows: Iterable[tuple[str, Any]],
    label_width: int = 18,
) -> Table:
    grid = Table.grid(padding=(0, 2), expand=False)
    grid.add_column(style="athena.label", justify="right", no_wrap=True, min_width=label_width)
    grid.add_column(style="athena.value", overflow="fold")
    for label, value in rows:
        grid.add_row(
            Text(str(label).upper(), style="athena.label"),
            _coerce_cell(value),
        )
    return grid


def status_badge(status: Any) -> Text:
    text = str(status or "").strip().lower()
    if text in ("success", "completed", "done", "ok"):
        return Text(" SUCCESS ", style="bold black on #34d399")
    if text in ("partial", "warn", "warning"):
        return Text(" PARTIAL ", style="bold black on #fbbf24")
    if text in ("failed", "error", "err"):
        return Text(" FAILED ", style="bold white on #dc2626")
    if text in ("running", "active"):
        return Text(" RUNNING ", style="bold white on #3b82f6")
    if text in ("pending", "idle"):
        return Text(" PENDING ", style="bold black on #c084fc")
    label = text.upper() or "UNKNOWN"
    return Text(f" {label} ", style="bold white on #4b5563")


def difficulty_badge(difficulty: Any) -> Text:
    text = str(difficulty or "").strip().lower()
    if text == "beginner":
        return Text(" beginner ", style="bold black on #34d399")
    if text == "intermediate":
        return Text(" intermediate ", style="bold black on #fbbf24")
    if text == "advanced":
        return Text(" advanced ", style="bold white on #dc2626")
    return Text(" unknown ", style="dim")


def score_badge(score: Any) -> Text:
    band = fmt.score_band(score)
    label = fmt.format_float(score)
    if band == "high":
        return Text(f" {label} ", style="bold black on #34d399")
    if band == "medium":
        return Text(f" {label} ", style="bold black on #fbbf24")
    if band == "low":
        return Text(f" {label} ", style="bold white on #f87171")
    if band == "very_low":
        return Text(f" {label} ", style="bold white on #dc2626")
    return Text(f" {label} ", style="dim")


def confidence_badge(confidence: Any) -> Text:
    band = fmt.confidence_band(confidence)
    label = fmt.format_float(confidence)
    if band == "high":
        return Text(f" {label} ", style="bold black on #34d399")
    if band == "medium":
        return Text(f" {label} ", style="bold black on #fbbf24")
    if band == "low":
        return Text(f" {label} ", style="bold white on #f87171")
    if band == "very_low":
        return Text(f" {label} ", style="bold white on #dc2626")
    return Text(f" {label} ", style="dim")


def bar(percent: Any, width: int = 24) -> Text:
    try:
        value = float(percent)
    except (TypeError, ValueError):
        value = 0.0
    if value != value:
        value = 0.0
    value = max(0.0, min(100.0, value))
    try:
        size = max(1, int(width))
    except (TypeError, ValueError):
        size = 24
    fill_count = max(0, min(size, int(round(size * value / 100.0))))
    empty_count = size - fill_count
    if value >= 75.0:
        style_name = "athena.success"
    elif value >= 40.0:
        style_name = "athena.warning"
    else:
        style_name = "athena.danger"
    text = Text()
    text.append("█" * fill_count, style=style_name)
    text.append("░" * empty_count, style="athena.muted")
    text.append(f"  {value:5.1f}%", style="athena.value")
    return text


def mini_bar(count: Any, max_count: Any, width: int = 16) -> Text:
    try:
        n = float(count)
    except (TypeError, ValueError):
        n = 0.0
    try:
        m = float(max_count)
    except (TypeError, ValueError):
        m = 0.0
    if m <= 0 or n != n or m != m:
        return Text("")
    ratio = max(0.0, min(1.0, n / m))
    try:
        size = max(1, int(width))
    except (TypeError, ValueError):
        size = 16
    fill_count = int(round(size * ratio))
    text = Text()
    text.append("▇" * fill_count, style="athena.accent")
    text.append("·" * (size - fill_count), style="athena.muted")
    return text


def score_bar(score: Any, width: int = 12) -> Text:
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 0.0
    if value != value:
        value = 0.0
    value = max(0.0, min(1.0, value))
    band = fmt.score_band(value)
    style_name = {
        "high": "athena.success",
        "medium": "athena.warning",
        "low": "athena.danger",
        "very_low": "athena.danger",
    }.get(band, "athena.muted")
    try:
        size = max(1, int(width))
    except (TypeError, ValueError):
        size = 12
    fill_count = int(round(size * value))
    text = Text()
    text.append("█" * fill_count, style=style_name)
    text.append("░" * (size - fill_count), style="athena.muted")
    return text


def data_table(
    columns: Sequence[tuple[str, str]],
    rows: Iterable[Sequence[Any]],
    title: str = "",
    expand: bool = True,
    show_lines: bool = False,
) -> Table:
    right_keys = {
        "rank", "score", "confidence", "count", "n",
        "tokens", "latency", "year", "cites", "share", "pct",
    }
    table = Table(
        box=box.ROUNDED,
        show_lines=show_lines,
        expand=expand,
        header_style="athena.heading",
        border_style="athena.border",
    )
    if title:
        table.title = panel_title(title)
    for key, label in columns:
        justify = "right" if key in right_keys else "left"
        table.add_column(str(label), justify=justify, overflow="fold")
    for row in rows:
        table.add_row(*[_coerce_cell(cell) for cell in row])
    return table


def empty_panel(message: str, title: str = "EMPTY") -> Panel:
    body = Align.center(
        Text(str(message or "Nothing to show."), style="athena.muted"),
        vertical="middle",
    )
    return panel(body, title=title, border="athena.border")


def error_panel(message: str, title: str = "ERROR", hint: str = "") -> Panel:
    body = Text(str(message or "Unknown error."), style="athena.text")
    if hint:
        body.append("\n")
        body.append(str(hint), style="athena.muted")
    return panel(body, title=title, border="athena.danger")


def warning_panel(message: str, title: str = "WARNING") -> Panel:
    body = Text(str(message or ""), style="athena.text")
    return panel(body, title=title, border="athena.warning")


def success_panel(message: str, title: str = "OK") -> Panel:
    body = Text(str(message or ""), style="athena.text")
    return panel(body, title=title, border="athena.success")


def bullet(text: str, glyph: str = "•") -> Text:
    body = Text()
    body.append(f" {glyph} ", style="athena.accent")
    body.append(str(text or ""), style="athena.text")
    return body


def key_value_lines(rows: Iterable[tuple[str, Any]], separator: str = "   ") -> Text:
    text = Text()
    first = True
    for label, value in rows:
        if not first:
            text.append(separator)
        first = False
        text.append(f"{label} ", style="athena.label")
        text.append(str(value), style="athena.value")
    return text


def divider(width: int = 60, char: str = "─") -> Text:
    try:
        size = max(1, int(width))
    except (TypeError, ValueError):
        size = 60
    return Text(char * size, style="athena.border")


def section_label(label: str) -> Text:
    return Text(f"  {str(label).upper()}", style="athena.heading")


def source_title(title: Any, url: Any = None, limit: int = 120) -> Text:
    clean = fmt.ellipsis(title or "Untitled", limit)
    text = Text(str(clean), style="athena.value")
    text.overflow = "fold"
    link = str(url or "").strip()
    if link and " " not in link:
        try:
            text.stylize(f"link {link}")
        except Exception:
            pass
    return text


def fallback_panel(ladder: Sequence[Any]) -> Panel | None:
    if not ladder:
        return None
    body = Text()
    for idx, step in enumerate(ladder):
        if idx:
            body.append("\n")
        stage = str(getattr(step, "stage", "") or "-")
        reason = str(getattr(step, "reason", "") or "")
        action = str(getattr(step, "action", "") or "")
        severity = str(getattr(step, "severity", "info") or "info").lower()
        if severity == "error":
            style_name = "athena.danger"
        elif severity in ("warn", "warning"):
            style_name = "athena.warning"
        elif severity == "ok":
            style_name = "athena.success"
        else:
            style_name = "athena.muted"
        body.append(f" {stage} ", style=f"bold {style_name}")
        if reason:
            body.append(f" {reason}", style="athena.text")
        if action:
            body.append(f"  →  {action}", style="athena.muted")
    return panel(body, title="FALLBACK LADDER", border="athena.border")