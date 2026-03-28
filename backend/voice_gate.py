"""
Emission gate between planner ticks and TTS: state + time only (no string similarity).
See plan: empty fingerprint wait-trap, dwell monotonic reset, is_tts_playing co-gate.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from agent_tools import Action

CommitReason = Literal["anchor", "motor", "dwell"]


def _voice_min_dwell_sec() -> float:
    try:
        return max(0.5, float(os.environ.get("VOICE_MIN_DWELL_SEC", "5")))
    except ValueError:
        return 5.0


def motor_fingerprint_for_actions(actions: list[Any]) -> str:
    """Canonical fingerprint: all non-wait actions, sorted by name+args."""
    parts: list[str] = []
    for a in actions:
        name = getattr(a, "name", None)
        if name == "wait" or name is None:
            continue
        raw = (getattr(a, "args_json", None) or "").strip() or "{}"
        parts.append(f"{name}:{raw}")
    parts.sort()
    return "|".join(parts)


def _derive_phase(
    anchor_after: str,
    fingerprint: str,
    actions: list[Any],
) -> str:
    if not (anchor_after or "").strip():
        return "idle"
    only_wait = bool(actions) and all(getattr(a, "name", None) == "wait" for a in actions)
    if fingerprint:
        return "motion_pending"
    if only_wait:
        return "waiting_scene"
    return "anchored"


@dataclass(frozen=True)
class VoicePayload:
    """JSON-safe fields for the client."""

    speak: str
    should_speak: bool
    phase: str
    supersede: bool


@dataclass(frozen=True)
class VoiceGateResult:
    payload: VoicePayload
    commit_reason: CommitReason | None
    motor_fingerprint: str


def compute_voice_gate(
    *,
    anchor_before: str,
    anchor_after: str,
    actions: list[Any],
    say: str,
    voice_last_speak_text: str,
    voice_last_speak_monotonic: float,
    voice_last_motor_fingerprint: str,
    now_monotonic: float | None = None,
    is_tts_playing: bool = False,
) -> VoiceGateResult:
    now = time.monotonic() if now_monotonic is None else now_monotonic
    fp = motor_fingerprint_for_actions(actions)
    phase = _derive_phase(anchor_after, fp, actions)

    raw_say = (say or "").strip()
    echo = (voice_last_speak_text or "").strip() or raw_say

    commit_reason: CommitReason | None = None
    supersede = False
    should_speak = False

    anchor_changed = anchor_before != anchor_after

    if anchor_changed:
        should_speak = True
        supersede = True
        commit_reason = "anchor"
    elif fp and fp != voice_last_motor_fingerprint:
        should_speak = True
        supersede = False
        commit_reason = "motor"
    elif not fp:
        pass
    else:
        pass

    if not should_speak:
        dwell_sec = _voice_min_dwell_sec()
        if (
            anchor_before == anchor_after
            and voice_last_speak_monotonic > 0.0
            and (now - voice_last_speak_monotonic) >= dwell_sec
        ):
            should_speak = True
            supersede = False
            commit_reason = "dwell"

    if is_tts_playing and should_speak and not supersede:
        should_speak = False
        commit_reason = None

    speak_out = raw_say if raw_say else echo
    if not should_speak:
        speak_out = echo if echo else raw_say

    payload = VoicePayload(
        speak=speak_out,
        should_speak=should_speak,
        phase=phase,
        supersede=supersede if should_speak else False,
    )
    return VoiceGateResult(
        payload=payload,
        commit_reason=commit_reason,
        motor_fingerprint=fp,
    )
