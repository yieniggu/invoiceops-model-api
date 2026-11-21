"""Logging of the service: INFO lines on stdout, ``key=value`` after a fixed prefix.

uvicorn only configures its own ``uvicorn.*`` loggers, so without this the ``invoiceops_model_api``
logger has no handler: INFO audit lines (``audit principal=...``), ``deploy_check`` and
``deployment_loaded/unloaded`` would be dropped and only WARNING+ would reach stderr through the
last-resort handler. Every line looks like
``ts=2026-01-01T00:00:00Z level=INFO logger=invoiceops_model_api msg="audit principal=... "``.
"""

from __future__ import annotations

import logging
import sys
import time

LOGGER_NAME = "invoiceops_model_api"
_MARK = "_invoiceops_stdout_handler"
MAX_VALUE_LENGTH = 200  # one logged value (a model name, an id) never grows past this
MAX_MESSAGE_LENGTH = 2000  # backstop for the whole message of one line


def _escape(text: str) -> str:
    """Backslash, double quote and every control character, so one record is one line.

    Besides the C0/C1 controls this covers NEL, LINE SEPARATOR and PARAGRAPH SEPARATOR, which
    ``str.splitlines`` and many log viewers also treat as line breaks.
    """
    named = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
    out = []
    for char in text:
        code = ord(char)
        if char == "\\":
            out.append("\\\\")
        elif char == '"':
            out.append('\\"')
        elif char in named:
            out.append(named[char])
        elif code < 0x20 or 0x7F <= code <= 0x9F or code in (0x2028, 0x2029):
            out.append(f"\\u{code:04x}")
        else:
            out.append(char)
    return "".join(out)


def log_value(value: object, limit: int = MAX_VALUE_LENGTH) -> str:
    """A client-controlled value capped at ``limit`` characters before it is logged.

    Escaping is done once, by ``KeyValueFormatter``, for the whole line (escaping here as well
    would double every backslash).
    """
    text = str(value)
    if len(text) > limit:
        return text[:limit] + f"...(+{len(text) - limit})"
    return text


class _StdoutHandler(logging.StreamHandler):
    """Writes to the CURRENT ``sys.stdout`` (so it survives stream replacement, e.g. in tests)."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property  # type: ignore[override]
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value) -> None:
        pass


class KeyValueFormatter(logging.Formatter):
    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        stamp = self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ")
        message = record.getMessage()
        if len(message) > MAX_MESSAGE_LENGTH:
            message = message[:MAX_MESSAGE_LENGTH] + "..."
        message = _escape(message)
        line = f'ts={stamp} level={record.levelname} logger={record.name} msg="{message}"'
        if record.exc_info:
            line += " exc=" + self.formatException(record.exc_info).replace("\n", "\\n")
        return line


def configure_logging() -> None:
    """Idempotent: attach one stdout handler at INFO to the service logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    if any(getattr(h, _MARK, False) for h in logger.handlers):
        return
    handler = _StdoutHandler()
    handler.setFormatter(KeyValueFormatter())
    setattr(handler, _MARK, True)
    logger.addHandler(handler)
