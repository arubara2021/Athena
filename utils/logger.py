from __future__ import annotations

import json
import logging
import logging.handlers
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core import constants
from core.exceptions import LoggingError

_SANITIZE_RULES = (
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.=+/]{16,}"), r"\1***"),
    (
        re.compile(
            r"(?i)((api[_-]?key|token|secret|password|authorization)\s*[:=]\s*)['\"]?[A-Za-z0-9_\-\.=+/]{8,}['\"]?"
        ),
        r"\1***",
    ),
    (re.compile(r"(?i)(nvapi-|gsk_|sk-|AIza)[A-Za-z0-9_\-]{8,}"), "***"),
)


def sanitize_text(value: str) -> str:
    if not value:
        return value
    for pattern, replacement in _SANITIZE_RULES:
        value = pattern.sub(replacement, value)
    return value


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_text(record.getMessage()),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = sanitize_text(self.formatException(record.exc_info))
        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            return json.dumps(
                {
                    "timestamp": payload["timestamp"],
                    "level": record.levelname,
                    "message": "log_serialization_error",
                },
                ensure_ascii=False,
            )


class SensitiveFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = sanitize_text(str(record.msg))
        if isinstance(record.args, dict):
            record.args = {
                key: sanitize_text(str(value)) if isinstance(value, str) else value
                for key, value in record.args.items()
            }
        elif record.args:
            record.args = tuple(
                sanitize_text(str(arg)) if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True


def _resolve_logging_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    candidate = str(level or constants.DEFAULT_LOG_LEVEL).upper()
    resolved = getattr(logging, candidate, None)
    if isinstance(resolved, int):
        return resolved
    return logging.INFO


def setup_logger(
    name: str = constants.APP_NAME,
    level: str | int | None = None,
    log_dir: Path | None = None,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    try:
        if level is None or log_dir is None:
            try:
                from core.config import get_settings
                settings = get_settings()
                level = level or settings.log_level
                log_dir = log_dir or settings.logs_dir
            except Exception:
                level = level or constants.DEFAULT_LOG_LEVEL
                log_dir = log_dir or Path(constants.DEFAULT_LOGS_DIR)

        logger = logging.getLogger(name)
        logger.setLevel(_resolve_logging_level(level))
        logger.handlers.clear()
        logger.propagate = False

        formatter = JSONFormatter()
        sensitive_filter = SensitiveFilter()

        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.addFilter(sensitive_filter)
            logger.addHandler(console_handler)

        if file:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_path / f"{name}.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(sensitive_filter)
            logger.addHandler(file_handler)

        return logger
    except Exception as exc:
        raise LoggingError(
            "Failed to configure logger",
            details={"logger": name, "error": str(exc)},
        ) from exc


def get_logger(child: str | None = None) -> logging.Logger:
    base_logger = logging.getLogger(constants.APP_NAME)
    if not base_logger.handlers:
        try:
            setup_logger()
        except LoggingError:
            logging.basicConfig(level=logging.INFO)
    if child:
        return base_logger.getChild(child)
    return base_logger


class TraceLogger:
    _instance: TraceLogger | None = None
    _lock = threading.Lock()

    def __init__(self, log_dir: Path | None = None) -> None:
        if log_dir is None:
            try:
                from core.config import get_settings
                log_dir = get_settings().logs_dir
            except Exception:
                log_dir = Path(constants.DEFAULT_LOGS_DIR)

        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._trace_path = self._log_dir / "debug_trace.jsonl"
        self._file_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> TraceLogger:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def emit(self, event: str, **kwargs: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **kwargs,
        }
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._file_lock:
                with open(self._trace_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            pass

    @property
    def trace_path(self) -> Path:
        return self._trace_path


def get_trace_logger() -> TraceLogger:
    return TraceLogger.get_instance()