from __future__ import annotations

import sys
from typing import Any, Iterable, Iterator

from rich.console import Console
from rich.text import Text

from cli import theme as theme_module


def _console(console: Console | None = None) -> Console:
    return console or theme_module.get_console()


def stream_text(
    text: str,
    console: Console | None = None,
    style: str = "athena.text",
    newline: bool = True,
    end: str = "",
) -> int:
    con = _console(console)
    payload = str(text or "")
    if not payload:
        if newline:
            try:
                con.print()
            except Exception:
                pass
        return 0
    try:
        con.print(Text(payload, style=style), end=end, soft_wrap=True)
        if newline:
            con.print()
        return len(payload)
    except KeyboardInterrupt:
        try:
            con.print()
        except Exception:
            pass
        raise
    except Exception:
        try:
            sys.stdout.write(payload)
            if newline:
                sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception:
            pass
        return len(payload)


def stream_chunks(
    chunks: Iterable[str],
    console: Console | None = None,
    style: str = "athena.text",
    newline: bool = True,
) -> int:
    con = _console(console)
    written = 0
    try:
        for chunk in chunks:
            piece = str(chunk or "")
            if not piece:
                continue
            con.print(Text(piece, style=style), end="", soft_wrap=True)
            written += len(piece)
    except KeyboardInterrupt:
        try:
            con.print()
        except Exception:
            pass
        raise
    except Exception:
        pass
    if newline:
        try:
            con.print()
        except Exception:
            pass
    return written


def stream_in_chunks(
    text: str,
    console: Console | None = None,
    style: str = "athena.text",
    chunk_size: int = 64,
    newline: bool = True,
) -> int:
    payload = str(text or "")
    if not payload:
        if newline:
            try:
                _console(console).print()
            except Exception:
                pass
        return 0
    try:
        size = max(1, int(chunk_size))
    except (TypeError, ValueError):
        size = 64

    def _iter() -> Iterator[str]:
        for index in range(0, len(payload), size):
            yield payload[index:index + size]

    return stream_chunks(
        _iter(), console=console, style=style, newline=newline
    )