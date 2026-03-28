"""
Per-session LATEST_STATE for decoupled vision (fast) vs agent (slow).
Snapshot copy-on-read under a short asyncio lock; never hold across Gemini.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any


def _task_anchor_max_len() -> int:
    return max(8, min(512, int(os.environ.get("AGENT_TASK_ANCHOR_MAX_LEN", "128"))))


def _memory_turns() -> int:
    return max(0, min(50, int(os.environ.get("AGENT_MEMORY_TURNS", "5"))))


def _memory_max_chars() -> int:
    return max(0, min(20000, int(os.environ.get("AGENT_MEMORY_MAX_CHARS", "2000"))))


def merge_task_anchor_from_model(current: str, model_out: str) -> str:
    """
    Tri-state task_anchor from the model:
    - "" or whitespace -> keep current
    - CLEAR (case-insensitive; outer quotes stripped) -> session anchor cleared
    - else -> replace with truncated text
    """
    s = (model_out or "").strip()
    while len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        s = s[1:-1].strip()
    if not s:
        return current
    if s.upper() == "CLEAR":
        return ""
    cap = _task_anchor_max_len()
    return s[:cap] if cap else s


@dataclass
class LatestSnapshot:
    frame_webp: bytes
    detections: dict[str, Any]
    grounded: list[dict[str, Any]]
    version: int


@dataclass
class _SessionBrainState:
    latest: LatestSnapshot | None = None
    recent_action_labels: list[str] | None = None
    last_agent_ms: float = 0.0
    agent_busy: bool = False
    task_anchor: str = ""
    turn_log: list[dict[str, Any]] = field(default_factory=list)
    last_step_monotonic: float = 0.0
    last_issued_action_labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.recent_action_labels is None:
            self.recent_action_labels = []


_sessions: dict[str, _SessionBrainState] = {}
_lock = asyncio.Lock()


def _now_ms() -> float:
    return time.monotonic() * 1000.0


async def publish_latest(
    session_id: str,
    frame_webp: bytes,
    detections: dict[str, Any],
    grounded: list[dict[str, Any]],
) -> None:
    """Vision path: overwrite latest state (full replace)."""
    async with _lock:
        st = _sessions.setdefault(session_id, _SessionBrainState())
        ver = (st.latest.version + 1) if st.latest is not None else 1
        st.latest = LatestSnapshot(
            frame_webp=bytes(frame_webp),
            detections={**detections},
            grounded=[dict(g) for g in grounded],
            version=ver,
        )


async def copy_snapshot(session_id: str) -> LatestSnapshot | None:
    """Agent path: deep-enough copy for use outside lock."""
    async with _lock:
        st = _sessions.get(session_id)
        if st is None or st.latest is None:
            return None
        s = st.latest
        return LatestSnapshot(
            frame_webp=bytes(s.frame_webp),
            detections={**s.detections},
            grounded=[dict(g) for g in s.grounded],
            version=s.version,
        )


async def get_recent_actions(session_id: str) -> list[str]:
    async with _lock:
        st = _sessions.get(session_id)
        if not st or not st.recent_action_labels:
            return []
        return list(st.recent_action_labels)


async def append_recent_actions(session_id: str, labels: list[str], max_items: int = 5) -> None:
    if not labels:
        return
    async with _lock:
        st = _sessions.setdefault(session_id, _SessionBrainState())
        assert st.recent_action_labels is not None
        st.recent_action_labels.extend(labels)
        st.recent_action_labels = st.recent_action_labels[-max_items:]


async def get_memory_for_prompt(session_id: str) -> dict[str, Any]:
    """Session continuity fields for the agent user message (read under lock)."""
    async with _lock:
        st = _sessions.get(session_id)
        if not st:
            return {
                "task_anchor": "",
                "turn_log_excerpt": "",
                "seconds_since_last_step": None,
                "last_issued_action_labels": [],
                "recent_action_labels": [],
            }
        sec: float | None = None
        if st.last_step_monotonic > 0.0:
            sec = max(0.0, time.monotonic() - st.last_step_monotonic)
        k = _memory_turns()
        turns = st.turn_log[-k:] if k else []
        excerpt = _format_turn_log_excerpt(turns, _memory_max_chars())
        return {
            "task_anchor": st.task_anchor,
            "turn_log_excerpt": excerpt,
            "seconds_since_last_step": sec,
            "last_issued_action_labels": list(st.last_issued_action_labels),
            "recent_action_labels": list(st.recent_action_labels or []),
        }


def _format_turn_log_excerpt(turns: list[dict[str, Any]], max_chars: int) -> str:
    if not turns or max_chars <= 0:
        return ""
    lines: list[str] = []
    for t in turns:
        say = str(t.get("say", "")).strip()
        acts = t.get("actions")
        if not isinstance(acts, list):
            acts = []
        anch = str(t.get("anchor", "")).strip()
        lines.append(
            json.dumps(
                {"say": say, "actions": acts, "anchor_after": anch},
                separators=(",", ":"),
            )
        )
    blob = "\n".join(lines)
    if len(blob) <= max_chars:
        return blob
    return blob[-max_chars:]


async def update_memory_after_agent_success(
    session_id: str,
    *,
    say: str,
    action_labels: list[str],
    task_anchor_model: str,
    recent_max: int = 5,
) -> str:
    """
    Merge tri-state task_anchor, append turn log, update timing and last-issued labels.
    Returns the stored task_anchor after merge (for API response).
    """
    async with _lock:
        st = _sessions.setdefault(session_id, _SessionBrainState())
        st.task_anchor = merge_task_anchor_from_model(st.task_anchor, task_anchor_model)
        st.turn_log.append(
            {
                "say": say,
                "actions": list(action_labels),
                "anchor": st.task_anchor,
            }
        )
        kt = _memory_turns()
        if kt > 0:
            st.turn_log = st.turn_log[-kt:]
        else:
            st.turn_log = []
        st.last_step_monotonic = time.monotonic()
        st.last_issued_action_labels = list(action_labels)
        assert st.recent_action_labels is not None
        st.recent_action_labels.extend(action_labels)
        st.recent_action_labels = st.recent_action_labels[-recent_max:]
        return st.task_anchor


async def try_begin_agent_work(session_id: str, min_interval_ms: float) -> tuple[bool, str]:
    """
    One in-flight agent call per session; min interval between successful completions.
    Returns (allowed, reason) with reason in {"ok", "busy", "throttle"}.
    """
    async with _lock:
        st = _sessions.setdefault(session_id, _SessionBrainState())
        if st.agent_busy:
            return False, "busy"
        now = _now_ms()
        if now - st.last_agent_ms < min_interval_ms:
            return False, "throttle"
        st.agent_busy = True
        return True, "ok"


async def finish_agent_work(session_id: str, *, success: bool) -> None:
    async with _lock:
        st = _sessions.get(session_id)
        if not st:
            return
        st.agent_busy = False
        if success:
            st.last_agent_ms = _now_ms()
