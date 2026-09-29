from __future__ import annotations

import traceback
from typing import Any

import typer
from rich.console import Console
from rich.text import Text

from core.exceptions import (
    CacheError,
    ConfigError,
    ConsensusError,
    EnsembleError,
    LearningPathError,
    LLMRequestError,
    LLMResponseParseError,
    MissingAPIKeyError,
    ModelNotFoundError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    RankingError,
    ResearchAgentError,
    SearchError,
    StorageError,
)

from cli import components as comp
from cli import theme as theme_module

_debug = False


def set_debug(enabled: bool) -> None:
    global _debug
    _debug = bool(enabled)


def is_debug() -> bool:
    return _debug


def _console() -> Console:
    return theme_module.get_console()


def _hint_for(exc: BaseException) -> str:
    if isinstance(exc, MissingAPIKeyError):
        return "Add the provider key to .env, then rerun."
    if isinstance(exc, ProviderAuthError):
        return "The provider rejected the credential. Regenerate the key and update .env."
    if isinstance(exc, ProviderRateLimitError):
        return "Rate limit hit on every configured key. Wait a minute, or add another key."
    if isinstance(exc, ProviderTimeoutError):
        return "The provider did not answer in time. Retry, or lower the max token count."
    if isinstance(exc, ModelNotFoundError):
        return "The model id is not served by this provider. Update configs/models.yaml."
    if isinstance(exc, ProviderResponseError):
        return "The provider returned an unexpected response. Retry, or switch models."
    if isinstance(exc, ProviderError):
        return "A provider call failed. Check configs/provider_config.yaml and .env."
    if isinstance(exc, LLMResponseParseError):
        return "The model returned malformed JSON. Retry, or lower the temperature."
    if isinstance(exc, LLMRequestError):
        return "The LLM request was rejected before it left the process."
    if isinstance(exc, ConsensusError):
        return "The voting models did not agree. The heuristic order will be used instead."
    if isinstance(exc, EnsembleError):
        return "The ensemble could not complete. A single model will be attempted."
    if isinstance(exc, RankingError):
        return "Ranking failed. Sources will fall back to heuristic order."
    if isinstance(exc, LearningPathError):
        return "Learning path generation failed. A heuristic path will be used."
    if isinstance(exc, StorageError):
        return "A disk write failed. Check folder permissions and free space."
    if isinstance(exc, CacheError):
        return "The cache layer failed. It is safe to clear the cache directory."
    if isinstance(exc, ConfigError):
        return "Check configs/*.yaml for syntax errors and required values."
    if isinstance(exc, SearchError):
        return "A search stage failed. Other platforms may still return results."
    if isinstance(exc, typer.BadParameter):
        return "Recheck the flag or argument. Run with --help for accepted values."
    if isinstance(exc, typer.Abort):
        return ""
    if isinstance(exc, KeyboardInterrupt):
        return "The run was cancelled."
    if isinstance(exc, FileNotFoundError):
        return "A required file is missing. Check the path and rerun."
    if isinstance(exc, PermissionError):
        return "Permission denied. Run from a folder you can write to."
    if isinstance(exc, ConnectionError):
        return "A network connection failed. Check connectivity and retry."
    return ""


def _title_for(exc: BaseException) -> str:
    if isinstance(exc, KeyboardInterrupt):
        return "CANCELLED"
    if isinstance(exc, typer.BadParameter):
        return "INVALID ARGUMENT"
    if isinstance(exc, typer.Abort):
        return "ABORTED"
    if isinstance(exc, ResearchAgentError):
        return exc.__class__.__name__.upper()
    return "ERROR"


def _message_for(exc: BaseException) -> str:
    if isinstance(exc, ResearchAgentError):
        base = str(getattr(exc, "message", "") or "").strip()
        if base:
            return base
    text = str(exc or "").strip()
    if text:
        return text
    return exc.__class__.__name__


def _render_traceback(exc: BaseException) -> Text:
    tb = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    ).rstrip()
    return Text(tb, style="athena.muted")


def print_error(
    exc: BaseException,
    console: Console | None = None,
    compact: bool = False,
) -> None:
    con = console or _console()
    message = _message_for(exc)
    hint = _hint_for(exc)
    title = _title_for(exc)

    try:
        if compact:
            line = Text()
            line.append("  ")
            line.append("✗", style="athena.danger")
            line.append("  ")
            line.append(message, style="athena.text")
            con.print(line)
            return

        body = Text(message, style="athena.text")
        if hint:
            body.append("\n")
            body.append(hint, style="athena.muted")
        if _debug and not isinstance(exc, (KeyboardInterrupt, typer.Abort)):
            body.append("\n\n")
            body.append(_render_traceback(exc))
        con.print()
        con.print(comp.panel(body, title=title, border="athena.danger"))
        con.print()
    except Exception:
        return


def print_warning(message: str, console: Console | None = None) -> None:
    con = console or _console()
    try:
        line = Text()
        line.append("  ")
        line.append("!", style="athena.warning")
        line.append("  ")
        line.append(str(message or ""), style="athena.text")
        con.print(line)
    except Exception:
        return


def print_info(message: str, console: Console | None = None) -> None:
    con = console or _console()
    try:
        line = Text()
        line.append("  ")
        line.append("i", style="athena.info")
        line.append("  ")
        line.append(str(message or ""), style="athena.text")
        con.print(line)
    except Exception:
        return