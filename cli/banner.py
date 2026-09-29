from __future__ import annotations

from rich.align import Align
from rich.text import Text

from cli import style
from cli import theme as theme_module


def _console():
    return theme_module.get_console()


def render_wordmark() -> Text:
    text = Text()
    for line in style.WORDMARK.splitlines():
        text.append(line, style="athena.primary")
        text.append("\n")
    if text.plain.endswith("\n"):
        text = Text(text.plain.rstrip("\n"), style="athena.primary")
    return text


def render_header(subtitle: str = "") -> Text:
    text = Text()
    text.append("  ")
    text.append(style.APP_NAME, style="athena.heading")
    if subtitle:
        text.append("  ")
        text.append(style.GLYPH_DOT, style="athena.muted")
        text.append("  ")
        text.append(str(subtitle), style="athena.accent")
    return text


def print_banner(subtitle: str = "", tagline: bool = True) -> None:
    console = _console()
    try:
        console.print()
        console.print(Align.center(render_wordmark()))
        if tagline:
            console.print(Align.center(Text(style.APP_TAGLINE, style="athena.muted")))
        if subtitle:
            console.print(Align.center(Text(str(subtitle), style="athena.accent")))
        console.print()
    except Exception:
        return


def print_title(title: str, subtitle: str = "", glyph: str = "") -> None:
    console = _console()
    try:
        line = Text()
        line.append("  ")
        line.append(f"{style.APP_NAME}", style="athena.heading")
        line.append("  ")
        line.append(style.GLYPH_DOT, style="athena.muted")
        line.append("  ")
        line.append(str(title or "").strip(), style="athena.primary")
        if subtitle:
            line.append("  ")
            line.append(style.GLYPH_DOT, style="athena.muted")
            line.append("  ")
            line.append(str(subtitle), style="athena.accent")
        if glyph:
            line.append("  ")
            line.append(str(glyph), style="athena.muted")
        rule = Text(style.DIVIDER_MINOR * 60, style="athena.border")
        console.print()
        console.print(line)
        console.print(rule)
        console.print()
    except Exception:
        return


def print_section(label: str) -> None:
    console = _console()
    try:
        console.print()
        console.print(Text(f"  {str(label).upper()}", style="athena.heading"))
        console.print(Text("  " + style.DIVIDER_MINOR * 56, style="athena.border"))
    except Exception:
        return


def print_hint(text: str) -> None:
    console = _console()
    try:
        console.print(Text(f"  {text}", style="athena.muted"))
    except Exception:
        return


def print_closing(message: str = "Done.", subtitle: str = "") -> None:
    console = _console()
    try:
        line = Text()
        line.append("  ")
        line.append(style.GLYPH_OK, style="athena.success")
        line.append("  ")
        line.append(str(message or "Done."), style="athena.value")
        if subtitle:
            line.append("  ")
            line.append(style.GLYPH_DOT, style="athena.muted")
            line.append("  ")
            line.append(str(subtitle), style="athena.muted")
        console.print()
        console.print(line)
        console.print()
    except Exception:
        return