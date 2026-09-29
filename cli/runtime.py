from __future__ import annotations

import logging
import os
from typing import Any

_original_emit: Any = None
_installed = False


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def silence_console_logger(force: bool = False) -> None:
    global _installed
    if _installed and not force:
        return
    try:
        _detach_existing_stream_handlers()
    except Exception:
        pass
    try:
        _install_emit_patch()
    except Exception:
        pass
    _installed = True


def restore_console_logger() -> None:
    global _original_emit, _installed
    if _original_emit is not None:
        try:
            logging.StreamHandler.emit = _original_emit
        except Exception:
            pass
        _original_emit = None
    _installed = False


def is_silenced() -> bool:
    return _installed


def _detach_existing_stream_handlers() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            continue
        try:
            root.removeHandler(handler)
        except Exception:
            pass

    try:
        for name in list(logging.Logger.manager.loggerDict.keys()):
            try:
                logger = logging.getLogger(name)
            except Exception:
                continue
            for handler in list(logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    continue
                try:
                    logger.removeHandler(handler)
                except Exception:
                    pass
    except Exception:
        pass


def _install_emit_patch() -> None:
    global _original_emit
    if _original_emit is not None:
        return
    if _env_truthy("ATHENA_SHOW_LOGS"):
        return

    original = logging.StreamHandler.emit

    def _patched_emit(self: logging.Handler, record: logging.LogRecord) -> None:
        if isinstance(self, logging.FileHandler):
            return original(self, record)
        return

    _original_emit = original
    logging.StreamHandler.emit = _patched_emit