"""Single-shot Gemini call with structured JSON output (AgentResponse)."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

from google import genai
from google.genai import types

from agent_tools import (
    ACTION_REGISTRY_PROMPT,
    MOTION_ACTION_NAMES,
    AgentResponse,
    action_args_dict,
)
from gemini_session_log import append_gemini_log

_client: genai.Client | None = None


def _serialize_gemini_response(resp: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        t = getattr(resp, "text", None)
        if t is not None:
            out["text"] = t
    except Exception:
        out["text"] = None
    try:
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            if hasattr(um, "model_dump"):
                out["usage_metadata"] = um.model_dump()
            else:
                out["usage_metadata"] = str(um)
    except Exception:
        pass
    try:
        pf = getattr(resp, "prompt_feedback", None)
        if pf is not None:
            out["prompt_feedback"] = str(pf)
    except Exception:
        pass
    return out

DEFAULT_HIGH_LEVEL_INTENTION = """You are a curious and helpful high-level planner for a household tidying assistant (like a home robot).
Your ongoing intention is to keep living spaces orderly: notice clutter such as shoes, bottles, clothes, or loose objects on the floor or messy surfaces;
guide the human to pick items up when it makes sense, pair or group related things (e.g. shoes together), and place them neatly in sensible locations (shelves, bins, tables).
Work step by step; prefer safe, practical moves the human can actually perform."""

CONTINUITY_AND_ANCHOR_RULES = """
## Task anchor and detections
- The session may have a stored task_anchor (one-line subgoal). In your JSON, task_anchor "" means keep the stored anchor unchanged; the exact token CLEAR clears it for free exploration; any other short string replaces it.
- When you output a NEW anchor (not CLEAR), you MUST use exact object class names as they appear in the grounded list (field "class"), e.g. if you see cup, write about the cup—not "beverage" or "drink"—so perception filtering stays aligned.
- If the stored task_anchor refers to an object class that is NOT in the current grounded detection list, do not rely on wait alone. Either re-acquire the scene (look_around, turn, scan) OR set task_anchor to CLEAR and choose a useful next step from what is visible. Do not spin or spam wait for objects you cannot see.
"""

PHYSICAL_EVIDENCE_AND_HOLDING = """
## Physical evidence, thought, instruction, and tools
- Tools are requests to the human, not facts. Do not use past tense in thought for pick_up, place, drop, or motion unless the current image and grounding support completion (e.g. object gone from floor, clear interaction, or last outcome / human feedback says so).
- Right after pick_up in actions, keep issuing pick_up in subsequent ticks until the image or feedback shows progress; use wait only when the scene is genuinely settling (see intent persistence below).
- Choose placement yourself: use place with target and near drawn from visible grounded classes. Do not ask the human an open-ended "where should I put it?" in thought unless last outcome explicitly requires it.
- If the user message includes an inferred held target matching a detection class, assume that detection may be the object in hand in first-person view. Do NOT issue pick_up again for that same held class; proceed toward place using other visible classes as near targets.
- If last outcome / human feedback contradicts what you see (e.g. they dropped the item), align thought and actions; use CLEAR or a new anchor if needed.
- Intent persistence: If you issue a physical command (move_forward, move_backward, turn_left, turn_right, pick_up, place, drop, look_around), you MUST keep the same primary motor intent in actions on subsequent ticks until the image or last_outcome shows completion or clear failure. Do not drop that command for wait or look_around alone on the very next tick while the human may still be executing it—otherwise downstream speech may never reflect the skipped step. Re-issue the same tool until evidence; use wait when motion is still in progress per cooldown hints.
"""

PERCEPTION_AND_SCHEMA_BLOCK = f"""You see one still frame (WebP) plus a JSON list of grounded objects. Fields use normalized image coordinates cx, cy in [0,1].
cz is a rough step-equivalent forward distance from monocular depth (calibrated constant K), NOT metric LIDAR—use only for ordering near vs far and coarse approach.

{ACTION_REGISTRY_PROMPT}

Output valid JSON only matching the schema: thought, instruction, actions, and task_anchor.
- thought: operator dashboard only (status, reasoning, conversational allowed). Not read aloud.
- instruction: text-to-speech only when non-empty; second-person imperatives; empty when waiting with nothing new to say aloud. Follow the instruction field rules in the schema (banned words, no questions).
- task_anchor: do not output punctuation-only or whitespace-only anchors; use "" to keep, CLEAR to clear, or a short object class name.
Be concise. Prefer one clear next step. Safety first."""


def _high_level_intention() -> str:
    raw = os.environ.get("HIGH_LEVEL_INTENTION_PROMPT", "").strip()
    return raw if raw else DEFAULT_HIGH_LEVEL_INTENTION


def _system_instruction() -> str:
    return (
        "## Mission\n"
        + _high_level_intention()
        + "\n\n## Role and perception\n"
        "You coordinate a human who is your hands and eyes in the real world from a first-person camera.\n\n"
        + CONTINUITY_AND_ANCHOR_RULES
        + "\n"
        + PHYSICAL_EVIDENCE_AND_HOLDING
        + "\n"
        + PERCEPTION_AND_SCHEMA_BLOCK
    )


def _get_client() -> genai.Client:
    global _client
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    if _client is None:
        _client = genai.Client(api_key=key)
    return _client


def _max_output_tokens() -> int:
    return max(32, min(512, int(os.environ.get("AGENT_MAX_OUTPUT_TOKENS", "256"))))


def _model_id() -> str:
    return os.environ.get("GEMINI_AGENT_MODEL", "gemini-2.5-flash-lite").strip()


def _motion_cooldown_hint_sec() -> float:
    try:
        return max(0.0, float(os.environ.get("AGENT_MOTION_COOLDOWN_HINT_SEC", "2")))
    except ValueError:
        return 2.0


def _action_label_base(label: str) -> str:
    s = (label or "").strip()
    if "(" in s:
        return s.split("(", 1)[0].strip()
    return s


def _user_text(
    grounded: list[dict[str, Any]],
    *,
    recent_actions: list[str],
    goal: str | None,
    last_outcome: str | None,
    task_anchor: str,
    turn_log_excerpt: str,
    seconds_since_last_step: float | None,
    last_issued_action_labels: list[str],
    inferred_held_object: str = "",
) -> str:
    parts = [
        "Grounded objects (JSON array):",
        json.dumps(grounded, separators=(",", ":")),
    ]
    if task_anchor:
        parts.append(f"Stored task_anchor (session): {task_anchor}")
    else:
        parts.append("Stored task_anchor (session): (none — free exploration)")
    ih = (inferred_held_object or "").strip()
    if ih:
        parts.append(
            f'Session tool-inference: last requested holding target: "{ih}". '
            "Verify in the image before claiming the human is holding it; infer cleared after place or drop."
        )
    else:
        parts.append(
            "Session tool-inference: no object inferred as held (or cleared after place/drop)."
        )
    if turn_log_excerpt:
        parts.append("Recent turn log (newest last, one JSON object per line):\n" + turn_log_excerpt)
    if recent_actions:
        parts.append(
            "Recent action labels (rolling): "
            + json.dumps(recent_actions)
            + " Re-issuing the SAME physical tool across ticks until the image or last_outcome shows progress is allowed and encouraged (intent persistence). "
            "Avoid spamming different conflicting motions (e.g. rapid place vs pick_up flip-flop). If stuck, use look_around, reorient, wait, or CLEAR the anchor per rules."
        )
    if last_issued_action_labels:
        parts.append("Last step action labels: " + json.dumps(last_issued_action_labels))
    cd = _motion_cooldown_hint_sec()
    if (
        seconds_since_last_step is not None
        and cd > 0
        and seconds_since_last_step < cd
        and last_issued_action_labels
    ):
        bases = [_action_label_base(x) for x in last_issued_action_labels]
        if any(b in MOTION_ACTION_NAMES for b in bases):
            parts.append(
                f"Only {seconds_since_last_step:.2f}s since last agent step; last step used motion. "
                "If execution is likely still in progress, use wait with args_json \"{}\" or a brief thought line; "
                "do not repeat the same motion tool unless the view clearly shows it failed."
            )
    if goal:
        parts.append(f"Current goal: {goal}")
    if last_outcome:
        parts.append(
            f"Last outcome / human feedback: {last_outcome} "
            "(Trust this over your assumptions if it contradicts the scene.)"
        )
    parts.append(
        "Respond with JSON: one physical next step for the human and the matching actions. "
        "Use thought for operator-facing status; use instruction only for a short imperative spoken line, or \"\" if nothing new to say aloud."
    )
    return "\n".join(parts)


def run_agent_sync(
    frame_webp: bytes,
    grounded: list[dict[str, Any]],
    *,
    recent_actions: list[str],
    goal: str | None = None,
    last_outcome: str | None = None,
    task_anchor: str = "",
    turn_log_excerpt: str = "",
    seconds_since_last_step: float | None = None,
    last_issued_action_labels: list[str] | None = None,
    inferred_held_object: str = "",
    session_id: str = "",
) -> AgentResponse:
    client = _get_client()
    last_issued = list(last_issued_action_labels or [])
    user = _user_text(
        grounded,
        recent_actions=recent_actions,
        goal=goal,
        last_outcome=last_outcome,
        task_anchor=task_anchor,
        turn_log_excerpt=turn_log_excerpt,
        seconds_since_last_step=seconds_since_last_step,
        last_issued_action_labels=last_issued,
        inferred_held_object=inferred_held_object,
    )
    image_part = types.Part.from_bytes(data=frame_webp, mime_type="image/webp")
    text_part = types.Part.from_text(text=user)
    contents = [
        types.Content(role="user", parts=[image_part, text_part]),
    ]
    config = types.GenerateContentConfig(
        system_instruction=_system_instruction(),
        response_mime_type="application/json",
        response_schema=AgentResponse,
        max_output_tokens=_max_output_tokens(),
    )
    model = _model_id()
    log_sid = (session_id or "").strip()
    base_entry: dict[str, Any] = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "user_text": user,
        "frame_webp_bytes": len(frame_webp),
        "grounded": grounded,
        "recent_actions": recent_actions,
        "goal": goal,
        "last_outcome": last_outcome,
        "task_anchor": task_anchor,
        "turn_log_excerpt": turn_log_excerpt,
        "seconds_since_last_step": seconds_since_last_step,
        "last_issued_action_labels": last_issued,
        "inferred_held_object": inferred_held_object,
        "max_output_tokens": _max_output_tokens(),
    }
    logged = False
    try:
        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        base_entry["gemini_response"] = _serialize_gemini_response(resp)
        if resp.parsed is not None and isinstance(resp.parsed, AgentResponse):
            base_entry["agent_response"] = resp.parsed.model_dump()
            if log_sid:
                append_gemini_log(log_sid, base_entry)
                logged = True
            return resp.parsed
        raw = (resp.text or "").strip()
        if not raw:
            base_entry["error"] = "empty Gemini response"
            if log_sid:
                append_gemini_log(log_sid, base_entry)
                logged = True
            raise RuntimeError("empty Gemini response")
        data = json.loads(raw)
        parsed = AgentResponse.model_validate(data)
        base_entry["agent_response"] = parsed.model_dump()
        if log_sid:
            append_gemini_log(log_sid, base_entry)
            logged = True
        return parsed
    except Exception as e:
        if not logged:
            base_entry["error"] = repr(e)
            if log_sid:
                append_gemini_log(log_sid, base_entry)
        raise


async def run_agent(
    frame_webp: bytes,
    grounded: list[dict[str, Any]],
    *,
    recent_actions: list[str],
    goal: str | None = None,
    last_outcome: str | None = None,
    task_anchor: str = "",
    turn_log_excerpt: str = "",
    seconds_since_last_step: float | None = None,
    last_issued_action_labels: list[str] | None = None,
    inferred_held_object: str = "",
    session_id: str = "",
) -> AgentResponse:
    return await asyncio.to_thread(
        run_agent_sync,
        frame_webp,
        grounded,
        recent_actions=recent_actions,
        goal=goal,
        last_outcome=last_outcome,
        task_anchor=task_anchor,
        turn_log_excerpt=turn_log_excerpt,
        seconds_since_last_step=seconds_since_last_step,
        last_issued_action_labels=last_issued_action_labels,
        inferred_held_object=inferred_held_object,
        session_id=session_id,
    )


def format_action_label(a: Any) -> str:
    from agent_tools import Action

    if isinstance(a, Action):
        name = a.name
        args = action_args_dict(a)
    else:
        name = getattr(a, "name", "?")
        args = getattr(a, "args", {}) or {}
    if not args:
        return str(name)
    return f"{name}({json.dumps(args, separators=(',', ':'))})"
