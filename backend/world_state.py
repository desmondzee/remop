"""
Per-session LATEST_STATE for decoupled vision (fast) vs agent (slow).
Snapshot copy-on-read under a short asyncio lock; never hold across Gemini.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
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
