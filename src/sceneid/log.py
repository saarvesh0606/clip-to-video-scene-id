"""Structured logging.

``logger.info("match.done", extra={"video_id": "x", "latency_ms": 12})`` becomes one JSON
line with those fields plus the current request id, so logs can be filtered by field
instead of grepped.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Attributes every LogRecord has; anything else on a record came from `extra=`.
# (uvicorn also attaches an ANSI-coloured copy of each message as `color_message`.)
_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {
    "message",
    "asctime",
    "taskName",
    "color_message",
}


def _extra_fields(record: logging.LogRecord) -> dict:
    return {k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        payload.update(_extra_fields(record))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = f"{record.levelname:<7} {record.name}: {record.getMessage()}"
        fields = _extra_fields(record)
        if fields:
            line += "  " + " ".join(f"{k}={v}" for k, v in fields.items())
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stderr`` is at the time, so redirection keeps working."""

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, _value):
        pass


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    handler = _StderrHandler()
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # Third-party libraries are chatty at INFO.
    for noisy in ("sentence_transformers", "transformers", "urllib3", "httpx", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
