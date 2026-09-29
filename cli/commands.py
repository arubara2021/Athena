from __future__ import annotations

import typer

_registered = False


def register(app: typer.Typer) -> None:
    global _registered
    if _registered:
        return
    try:
        from cli import diagnostics
        diagnostics.register(app)
    except Exception:
        pass
    try:
        from cli import runs
        runs.register(app)
    except Exception:
        pass
    try:
        from cli import search
        search.register(app)
    except Exception:
        pass
    try:
        from cli import agent
        agent.register(app)
    except Exception:
        pass
    try:
        from cli import chat
        chat.register(app)
    except Exception:
        pass
    _registered = True


def is_registered() -> bool:
    return _registered