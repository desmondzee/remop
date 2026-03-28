"""Append Gemini request/response records as JSON Lines (one object per line) under backend/logs/."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_DIR = Path(__file__).resolve().parent / "logs"
_locks: dict[str, threading.Lock] = {}
_paths: dict[str, Path] = {}
_meta_lock = threading.Lock()


def _safe_session_id(sid: str) -> str:
    s = sid.strip() or "default"
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:120] or "default"


def _get_session_lock(sid_key: str) -> threading.Lock:
    with _meta_lock:
        if sid_key not in _locks:
            _locks[sid_key] = threading.Lock()
        return _locks[sid_key]


def append_gemini_log(session_id: str, entry: dict[str, Any]) -> None:
    """Thread-safe append; O(1) per entry (no full-file rewrite)."""
    sid_key = (session_id or "").strip() or "default"
    safe = _safe_session_id(sid_key)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    lock = _get_session_lock(sid_key)
    with lock:
        if sid_key not in _paths:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            _paths[sid_key] = _LOG_DIR / f"gemini_{safe}_{ts}.jsonl"
        path = _paths[sid_key]
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
