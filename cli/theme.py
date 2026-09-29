from __future__ import annotations

import os
import threading
from typing import Any

from rich.console import Console
from rich.theme import Theme

from cli import style

MIDNIGHT_NAME = "athena-midnight"
PAPER_NAME = "athena-paper"
MONO_NAME = "athena-mono"

DEFAULT_THEME_NAME = MIDNIGHT_NAME

MIDNIGHT = Theme(
    {
        "athena.primary": f"bold {style.PRIMARY_BRIGHT}",
        "athena.secondary": f"bold {style.SECONDARY_BRIGHT}",
        "athena.accent": f"bold {style.ACCENT}",
        "athena.success": f"bold {style.SUCCESS}",
        "athena.warning": f"bold {style.WARNING}",
        "athena.danger": f"bold {style.DANGER}",
        "athena.info": f"bold {style.INFO}",
        "athena.text": style.TEXT,
        "athena.muted": f"dim {style.MUTED}",
        "athena.border": style.BORDER,
        "athena.panel": style.PANEL_BG,
        "athena.surface": style.SURFACE,
        "athena.background": style.BACKGROUND,
        "athena.label": f"bold {style.SECONDARY_BRIGHT}",
        "athena.value": f"bold {style.TEXT}",
        "athena.heading": f"bold {style.PRIMARY_BRIGHT}",
        "athena.metric": f"bold {style.ACCENT}",
    },
    inherit=False,
)

PAPER = Theme(
    {
        "athena.primary": "bold #6d28d9",
        "athena.secondary": "bold #2563eb",
        "athena.accent": "bold #0e7490",
        "athena.success": "bold #15803d",
        "athena.warning": "bold #b45309",
        "athena.danger": "bold #b91c1c",
        "athena.info": "bold #1d4ed8",
        "athena.text": "#0f172a",
        "athena.muted": "dim #64748b",
        "athena.border": "#c7d2fe",
        "athena.panel": "#eef2ff",
        "athena.surface": "#ffffff",
        "athena.background": "#f8fafc",
        "athena.label": "bold #2563eb",
        "athena.value": "bold #0f172a",
        "athena.heading": "bold #6d28d9",
        "athena.metric": "bold #0e7490",
    },
    inherit=False,
)

MONO = Theme(
    {
        "athena.primary": "bold",
        "athena.secondary": "bold",
        "athena.accent": "bold",
        "athena.success": "bold",
        "athena.warning": "bold",
        "athena.danger": "bold",
        "athena.info": "bold",
        "athena.text": "default",
        "athena.muted": "dim",
        "athena.border": "default",
        "athena.panel": "default",
        "athena.surface": "default",
        "athena.background": "default",
        "athena.label": "bold",
        "athena.value": "default",
        "athena.heading": "bold",
        "athena.metric": "bold",
    },
    inherit=False,
)

THEMES: dict[str, Theme] = {
    MIDNIGHT_NAME: MIDNIGHT,
    PAPER_NAME: PAPER,
    MONO_NAME: MONO,
}

THEME_ORDER: tuple[str, ...] = (MIDNIGHT_NAME, PAPER_NAME, MONO_NAME)

THEME_ALIASES: dict[str, str] = {
    "midnight": MIDNIGHT_NAME,
    "dark": MIDNIGHT_NAME,
    "default": MIDNIGHT_NAME,
    "paper": PAPER_NAME,
    "light": PAPER_NAME,
    "mono": MONO_NAME,
    "monochrome": MONO_NAME,
    "bw": MONO_NAME,
}

_lock = threading.RLock()
_active_name = DEFAULT_THEME_NAME
_console: Console | None = None
_no_color = False


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_name(name: Any = None) -> str:
    if name is not None:
        text = str(name).strip().lower()
        if text:
            if text in THEMES:
                return text
            if text in THEME_ALIASES:
                return THEME_ALIASES[text]

    env = os.environ.get("ATHENA_THEME", "").strip().lower()
    if env:
        if env in THEMES:
            return env
        if env in THEME_ALIASES:
            return THEME_ALIASES[env]

    if _env_truthy("NO_COLOR"):
        return MONO_NAME

    return DEFAULT_THEME_NAME


def get_theme(name: Any = None) -> Theme:
    with _lock:
        if name is None:
            return THEMES[_active_name]
        return THEMES[resolve_name(name)]


def get_active_name() -> str:
    with _lock:
        return _active_name


def list_names() -> tuple[str, ...]:
    return THEME_ORDER


def build_console(name: Any = None, no_color: bool | None = None) -> Console:
    resolved = resolve_name(name)
    resolved_theme = THEMES[resolved]
    resolved_no_color = _no_color if no_color is None else bool(no_color)
    try:
        return Console(
            theme=resolved_theme,
            no_color=resolved_no_color,
            highlight=False,
            markup=True,
            soft_wrap=False,
            emoji=False,
        )
    except Exception:
        try:
            return Console(highlight=False)
        except Exception:
            return Console()


def set_console(console: Console) -> None:
    global _console
    with _lock:
        _console = console


def get_console() -> Console:
    global _console
    with _lock:
        if _console is None:
            _console = build_console()
        return _console


def reset_console() -> None:
    global _console
    with _lock:
        _console = None


def set_theme(name: Any) -> str:
    global _active_name, _console
    resolved = resolve_name(name)
    with _lock:
        _active_name = resolved
        _console = build_console(resolved)
    return resolved


def cycle_theme() -> str:
    with _lock:
        current = _active_name
    try:
        idx = THEME_ORDER.index(current)
    except ValueError:
        idx = 0
    nxt = THEME_ORDER[(idx + 1) % len(THEME_ORDER)]
    return set_theme(nxt)


def set_no_color(enabled: bool) -> None:
    global _no_color, _console
    with _lock:
        _no_color = bool(enabled)
        _console = build_console(_active_name, no_color=_no_color)


def color_enabled() -> bool:
    if _no_color or _env_truthy("NO_COLOR"):
        return False
    term = os.environ.get("TERM", "").strip().lower()
    return term != "dumb"