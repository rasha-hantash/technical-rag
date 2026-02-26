"""Structured JSON logger matching Go slog format.

Outputs logs in the format:
{"time":"2026-02-03T14:06:20.829529-05:00","level":"INFO","source":{"function":"main","file":"app.py","line":43},"msg":"message"}
"""

import inspect
import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# Context variable for storing additional log fields
_log_context: ContextVar[dict[str, Any]] = ContextVar("log_context", default={})


class StructuredFormatter(logging.Formatter):
    """JSON formatter that outputs logs in Go slog-compatible format."""

    def format(self, record: logging.LogRecord) -> str:
        # Get current time with timezone
        now = datetime.now(timezone.utc).astimezone()
        time_str = now.isoformat()

        # Build the log entry
        log_entry: dict[str, Any] = {
            "time": time_str,
            "level": record.levelname,
            "source": {
                "function": record.funcName,
                "file": record.pathname,
                "line": record.lineno,
            },
            "msg": record.getMessage(),
        }

        # Add context fields
        ctx_fields = _log_context.get()
        if ctx_fields:
            log_entry.update(ctx_fields)

        # Add any extra fields passed to the log call
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)


class StructuredLogger:
    """Logger that outputs structured JSON logs with context support."""

    def __init__(self, name: str = "app"):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        self._logger.handlers.clear()

        # Add JSON handler to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        self._logger.addHandler(handler)

        # Prevent propagation to root logger
        self._logger.propagate = False

    def _log(
        self,
        level: int,
        msg: str,
        stacklevel: int = 3,
        **fields: Any,
    ) -> None:
        """Internal log method that handles extra fields."""
        # Create a LogRecord with extra fields
        extra = {"extra_fields": fields} if fields else {}
        self._logger.log(level, msg, stacklevel=stacklevel, extra=extra)

    def debug(self, msg: str, **fields: Any) -> None:
        """Log a debug message with optional fields."""
        self._log(logging.DEBUG, msg, **fields)

    def info(self, msg: str, **fields: Any) -> None:
        """Log an info message with optional fields."""
        self._log(logging.INFO, msg, **fields)

    def warn(self, msg: str, **fields: Any) -> None:
        """Log a warning message with optional fields."""
        self._log(logging.WARNING, msg, **fields)

    def error(self, msg: str, **fields: Any) -> None:
        """Log an error message with optional fields."""
        self._log(logging.ERROR, msg, **fields)


def set_context(**fields: Any) -> None:
    """Set context fields that will be included in all subsequent log messages.

    Example:
        set_context(request_id="abc-123", user_id="user-456")
        logger.info("processing request")  # includes request_id and user_id
    """
    current = _log_context.get()
    _log_context.set({**current, **fields})


def clear_context() -> None:
    """Clear all context fields."""
    _log_context.set({})


def get_context() -> dict[str, Any]:
    """Get current context fields."""
    return _log_context.get().copy()


# Default logger instance
logger = StructuredLogger("pdf_llm_server")
