"""
Per-session LATEST_STATE for decoupled vision (fast) vs agent (slow).
Snapshot copy-on-read under a short asyncio lock; never hold across Gemini.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from agent_tools import Action, action_args_dict
from voice_gate import VoiceGateResult


def _held_object_max_len() -> int:
    return max(16, min(256, int(os.environ.get("AGENT_HELD_OBJECT_MAX_LEN", "128"))))


def apply_inferred_held_after_actions(current: str, actions: list[Action]) -> str:
    """
    Walk this tick's actions in order: place/drop clear; pick_up sets target if present.
    pick_up without target leaves previous held unchanged.
    """
    held = (current or "").strip()
    cap = _held_object_max_len()
    for a in actions:
        if a.name in ("place", "drop"):
            held = ""
        elif a.name == "pick_up":
            t = action_args_dict(a).get("target")
            if isinstance(t, str) and t.strip():
                held = t.strip()[:cap]
    return held[:cap] if held else ""


def _task_anchor_max_len() -> int:
    return max(8, min(512, int(os.environ.get("AGENT_TASK_ANCHOR_MAX_LEN", "128"))))


def _memory_turns() -> int:
    return max(0, min(50, int(os.environ.get("AGENT_MEMORY_TURNS", "5"))))


def _memory_max_chars() -> int:
    return max(0, min(20000, int(os.environ.get("AGENT_MEMORY_MAX_CHARS", "2000"))))


def sanitize_task_anchor_text(model_out: str) -> str:
    """
    Keep alphanumerics and ASCII spaces; collapse whitespace.
    Empty after sanitize means no usable anchor (caller treats as keep-current).
    """
    s = (model_out or "").strip()
    s = re.sub(r"[^a-zA-Z0-9 ]+", " ", s)
    s = " ".join(s.split())
    return s.strip()


def merge_task_anchor_from_model(current: str, model_out: str) -> str:
    """
    Tri-state task_anchor from the model:
    - "" or whitespace -> keep current
    - CLEAR (case-insensitive; outer quotes stripped) -> session anchor cleared
    - else -> replace with sanitized truncated text; punctuation-only -> keep current
    """
    s = (model_out or "").strip()
    while len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        s = s[1:-1].strip()
    if not s:
        return current
    if s.upper() == "CLEAR":
        return ""
    sanitized = sanitize_task_anchor_text(s)
    if not sanitized:
        return current
    cap = _task_anchor_max_len()
    return sanitized[:cap] if cap else sanitized


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
    voice_last_speak_text: str = ""
    voice_last_speak_monotonic: float = 0.0
    voice_last_motor_fingerprint: str = ""
    inferred_held_object: str = ""

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
                "voice_last_speak_text": "",
                "voice_last_speak_monotonic": 0.0,
                "voice_last_motor_fingerprint": "",
                "inferred_held_object": "",
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
            "voice_last_speak_text": st.voice_last_speak_text,
            "voice_last_speak_monotonic": st.voice_last_speak_monotonic,
            "voice_last_motor_fingerprint": st.voice_last_motor_fingerprint,
            "inferred_held_object": st.inferred_held_object,
        }


def _format_turn_log_excerpt(turns: list[dict[str, Any]], max_chars: int) -> str:
    if not turns or max_chars <= 0:
        return ""

    def _serialize_turn(t: dict[str, Any]) -> str:
        thought = str(t.get("thought", "") or t.get("say", "")).strip()
        instruction = str(t.get("instruction", "")).strip()
        acts = t.get("actions")
        if not isinstance(acts, list):
            acts = []
        anch = str(t.get("anchor", "")).strip()
        return json.dumps(
            {
                "thought": thought,
                "instruction": instruction,
                "actions": acts,
                "anchor_after": anch,
            },
            separators=(",", ":"),
        )

    lines = [_serialize_turn(t) for t in turns]
    blob = "\n".join(lines)
    if len(blob) <= max_chars:
        return blob

    chosen_rev: list[str] = []
    total = 0
    for s in reversed(lines):
        add_len = len(s) + (1 if chosen_rev else 0)
        if total + add_len > max_chars:
            break
        chosen_rev.append(s)
        total += add_len
    if chosen_rev:
        return "\n".join(reversed(chosen_rev))

    last_turn = turns[-1]
    thought = str(last_turn.get("thought", "") or last_turn.get("say", "")).strip()
    instruction = str(last_turn.get("instruction", "") or "").strip()
    acts = last_turn.get("actions")
    if not isinstance(acts, list):
        acts = []
    anch = str(last_turn.get("anchor", "")).strip()
    while True:
        line = json.dumps(
            {
                "thought": thought,
                "instruction": instruction,
                "actions": acts,
                "anchor_after": anch,
            },
            separators=(",", ":"),
        )
        if len(line) <= max_chars:
            return line
        if len(thought) > 48:
            thought = thought[: len(thought) - 120].rstrip() + "…"
        elif len(instruction) > 24:
            instruction = instruction[: len(instruction) - 60].rstrip() + "…"
        else:
            return json.dumps(
                {
                    "thought": "…",
                    "instruction": instruction[:80],
                    "actions": acts,
                    "anchor_after": anch,
                },
                separators=(",", ":"),
            )[:max_chars]


async def update_memory_after_agent_success(
    session_id: str,
    *,
    thought: str,
    instruction: str,
    action_labels: list[str],
    task_anchor_model: str,
    actions: list[Action] | None = None,
    recent_max: int = 5,
) -> tuple[str, str]:
    """
    Merge tri-state task_anchor, append turn log, update timing, inferred held object, last-issued labels.
    Returns (stored task_anchor, inferred_held_object) for API response.
    """
    async with _lock:
        st = _sessions.setdefault(session_id, _SessionBrainState())
        st.task_anchor = merge_task_anchor_from_model(st.task_anchor, task_anchor_model)
        if actions is not None:
            st.inferred_held_object = apply_inferred_held_after_actions(
                st.inferred_held_object, actions
            )
        st.turn_log.append(
            {
                "thought": thought,
                "instruction": instruction,
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
        return st.task_anchor, st.inferred_held_object


async def commit_voice_gate_result(session_id: str, result: VoiceGateResult) -> None:
    """Persist voice layer when the gate commits a speak (resets dwell clock)."""
    if not result.payload.should_speak:
        return
    async with _lock:
        st = _sessions.get(session_id)
        if not st:
            return
        p = result.payload
        st.voice_last_speak_text = p.speak
        st.voice_last_speak_monotonic = time.monotonic()
        if result.commit_reason == "motor" and result.motor_fingerprint:
            st.voice_last_motor_fingerprint = result.motor_fingerprint


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
