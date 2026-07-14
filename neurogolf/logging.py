"""Small structured-logging setup shared by command-line entry points."""

from __future__ import annotations

import json
import logging
from typing import Any


class JsonFormatter(logging.Formatter):
    """Render one compact JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for name in ("event", "task_num", "candidate_sha256"):
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = value
        return json.dumps(payload, sort_keys=True, default=str)


def configure_logging(*, debug: bool = False, json_output: bool = False) -> None:
    """Configure deterministic stderr logging without replacing caller handlers."""

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    if not root.handlers:
        handler = logging.StreamHandler()
        root.addHandler(handler)
    formatter: logging.Formatter
    if json_output:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
    for handler in root.handlers:
        handler.setFormatter(formatter)
