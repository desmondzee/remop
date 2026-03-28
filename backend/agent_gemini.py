"""Single-shot Gemini call with structured JSON output (AgentResponse)."""

from __future__ import annotations

import asyncio
import functools
import json
import os
import re
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
from executors import AGENT
from gemini_session_log import append_gemini_log

_client: genai.Client | None = None

_THOUGHT_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _clamp_operator_thought(text: str, *, max_sentences: int = 2) -> str:
    """Keep operator/log thought short: at most max_sentences (default 2)."""
    t = (text or "").strip()
    if not t or max_sentences < 1:
        return t
    parts = [p.strip() for p in _THOUGHT_SENTENCE_SPLIT.split(t) if p.strip()]
    if len(parts) <= max_sentences:
        return t
    return " ".join(parts[:max_sentences]).strip()


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
- The JSON list can include objects from earlier frames in the window; always **verify against the WebP** whether something is still relevant in the current view.
"""

PHYSICAL_EVIDENCE_AND_HOLDING = """
## Physical evidence, thought, instruction, and tools
- Tools are requests, not facts. Use past tense in thought for pick/place/drop/motion only when the frame supports it: hand–object contact **seen in the WebP**, last_outcome, or human feedback—not because a class disappeared from the JSON list (it often does not). The detector JSON never includes hands or grasp; do not expect a "hand" label there. An empty grounded list is not proof the anchor object is gone; never claim a class is "still detected" if it is not in the JSON—use the image, vision-inferred contact, and last_outcome instead.
- Pick_up (first-person): after you requested pick_up, **inspect the WebP with your vision** (multimodal)—not the JSON—for a hand grasping or clearly touching the intended object; that is pick-up progress. Grounding may still list that class while it is in hand; do not treat "still in JSON" as failure. With contact visible in the image, stop repeating pick_up (instruction "" when continuing) and move toward place with a grounded "near". With no visible contact and no other progress, keep pick_up in actions per intent persistence; leave instruction "" when repeating (Silence is Golden).
- Holding: combine the session inferred-held line with **vision on the WebP**. If inferred held matches a class still in detections, that row may be the item in hand—do not pick_up that class again until place or drop. Same-class objects on a surface **without** contact/grasp visible in the image are not proof of holding; task_anchor alone is not proof.
- Place only when a valid "near" class is in the current grounded list; if not, look_around or move_forward (cz) first—do not spam the same invalid place.
- If last_outcome contradicts the scene, realign; use CLEAR or a new anchor if needed.
- Intent persistence: keep the same primary motor action across ticks until completion, clear failure, or hand/grasp **visible in the WebP** for pick_up—not merely "object vanished from grounding." Do not swap to wait or look_around alone on the very next tick while the human may still be executing the prior step; use wait when motion is settling per cooldown hints.
"""

PERCEPTION_AND_SCHEMA_BLOCK = f"""## Multimodal input (read this carefully)
You receive **one current still frame** (WebP) and a **separate JSON list** of detector outputs. Treat these as **two different sensors**:
- **WebP image (primary):** Use your vision on this frame for layout, occlusion, surfaces, clutter, lighting, and **whether the human’s hands are touching or holding something**. The image is the authoritative source for what is *visibly* in front of the camera *right now*.
- **JSON list (secondary):** Bounding-box class names and coarse positions only. It does **not** show hands, grasps, material, or many small objects; it can be incomplete, noisy, or include stale entries from earlier moments in the accumulation window. **Do not** plan from the JSON alone—cross-check every important claim against the WebP.

Fields use normalized image coordinates cx, cy in [0,1].
cz is a rough step-equivalent forward distance from monocular depth (calibrated constant K), NOT metric LIDAR—use only for ordering near vs far and coarse approach.

{ACTION_REGISTRY_PROMPT}

Output valid JSON only matching the schema: thought, instruction, actions, and task_anchor.
- thought: operator / log only (not TTS). One short sentence ideal; two sentences maximum. No paragraph-style narration.
- instruction: the ONLY spoken line for this tick; second-person imperatives; follow schema (banned words, no questions). Empty "" means complete silence—use that during wait or when repeating the same ongoing motion.
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
        "You coordinate a human who is your hands and eyes in the real world from a first-person camera. "
        "The WebP frame is the live view—inspect it every turn; the JSON list is a helper, not a substitute for looking.\n\n"
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
    grounded_from_accumulator: bool = False,
) -> str:
    if grounded_from_accumulator:
        parts0 = [
            "Grounded objects (JSON array) — **union across all detector passes since your last completed agent step** "
            "(same approximate class+position keeps highest confidence; capped). "
            "Reconcile with the WebP; entries may be stale if the scene moved.",
        ]
    else:
        parts0 = [
            "Grounded objects (JSON array) — **latest vision frame only** (accumulator empty, e.g. first step after engage). "
            "Still prefer verifying layout and hands in the WebP.",
        ]
    parts = [
        *parts0,
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
            "Detection JSON does not include hands—judge grasp or contact only from the WebP frame. "
            "Infer cleared after place or drop."
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
            + " Re-issuing the SAME physical tool across ticks until completion, clear failure, last_outcome, or "
            "(for pick_up) hand–object grasp/contact **visible in the WebP** (vision, not detection JSON) is allowed and encouraged (intent persistence). "
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
                "If execution is likely still in progress, use wait with args_json \"{}\" or one short thought sentence; "
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
        "Base your understanding of the **current** scene primarily on **looking at the WebP**; use the JSON as hints only. "
        "thought: one sentence for the operator log if possible, two at most (not spoken). "
        "instruction: only when the human should hear speech this tick—"
        "otherwise leave instruction empty \"\" for silence (no server-derived speech from actions)."
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
    grounded_from_accumulator: bool = False,
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
        grounded_from_accumulator=grounded_from_accumulator,
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
        "grounded_from_accumulator": grounded_from_accumulator,
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
            out = resp.parsed.model_copy(
                update={"thought": _clamp_operator_thought(resp.parsed.thought)}
            )
            base_entry["agent_response"] = out.model_dump()
            if log_sid:
                append_gemini_log(log_sid, base_entry)
                logged = True
            return out
        raw = (resp.text or "").strip()
        if not raw:
            base_entry["error"] = "empty Gemini response"
            if log_sid:
                append_gemini_log(log_sid, base_entry)
                logged = True
            raise RuntimeError("empty Gemini response")
        data = json.loads(raw)
        parsed = AgentResponse.model_validate(data)
        out = parsed.model_copy(update={"thought": _clamp_operator_thought(parsed.thought)})
        base_entry["agent_response"] = out.model_dump()
        if log_sid:
            append_gemini_log(log_sid, base_entry)
            logged = True
        return out
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
    grounded_from_accumulator: bool = False,
) -> AgentResponse:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        AGENT,
        functools.partial(
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
            grounded_from_accumulator=grounded_from_accumulator,
        ),
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
