"""JSONL logger for SDK call tracing.

Writes one JSON object per line to {DEEPCLAUDE_WORKSPACE}/research.log.
Transparent to Claude — called internally by evaluate/validate/submit.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_writer = None


def _get_writer():
    global _writer
    if _writer is None:
        workspace = Path(os.environ.get("DEEPCLAUDE_WORKSPACE", "."))
        workspace.mkdir(parents=True, exist_ok=True)
        _writer = open(workspace / "research.log", "a", encoding="utf-8")
    return _writer


def log(event: str, **kwargs) -> None:
    """Append a JSONL entry to research.log."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session": os.environ.get("DEEPCLAUDE_SESSION_ID", "local"),
        "event": event,
        **kwargs,
    }
    w = _get_writer()
    w.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    w.flush()
