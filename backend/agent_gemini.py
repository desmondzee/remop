"""Single-shot Gemini call with structured JSON output (AgentResponse)."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from google import genai
from google.genai import types

from agent_tools import (
    ACTION_REGISTRY_PROMPT,
    MOTION_ACTION_NAMES,
    AgentResponse,
    action_args_dict,
)

_client: genai.Client | None = None

DEFAULT_HIGH_LEVEL_INTENTION = """You are the high-level planner for a household tidying assistant (like a home robot).
Your ongoing intention is to keep living spaces orderly: notice clutter such as shoes, bottles, clothes, or loose objects on the floor or messy surfaces;
guide the human to pick items up when it makes sense, pair or group related things (e.g. shoes together), and place them neatly in sensible locations (shelves, bins, tables).
Work step by step; prefer safe, practical moves the human can actually perform."""

CONTINUITY_AND_ANCHOR_RULES = """
## Task anchor and detections
- The session may have a stored task_anchor (one-line subgoal). In your JSON, task_anchor "" means keep the stored anchor unchanged; the exact token CLEAR clears it for free exploration; any other short string replaces it.
- When you output a NEW anchor (not CLEAR), you MUST use exact object class names as they appear in the grounded list (field "class"), e.g. if you see cup, write about the cup—not "beverage" or "drink"—so perception filtering stays aligned.
- If the stored task_anchor refers to an object class that is NOT in the current grounded detection list, do not rely on wait alone. Either re-acquire the scene (look_around, turn, scan) OR set task_anchor to CLEAR and choose a useful next step from what is visible. Do not spin or spam wait for objects you cannot see.
"""

PERCEPTION_AND_SCHEMA_BLOCK = f"""You see one still frame (WebP) plus a JSON list of grounded objects. Fields use normalized image coordinates cx, cy in [0,1].
cz is a rough step-equivalent forward distance from monocular depth (calibrated constant K), NOT metric LIDAR—use only for ordering near vs far and coarse approach.

{ACTION_REGISTRY_PROMPT}

Output valid JSON only matching the schema: say, actions, and task_anchor.
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
) -> str:
    parts = [
        "Grounded objects (JSON array):",
        json.dumps(grounded, separators=(",", ":")),
    ]
    if task_anchor:
        parts.append(f"Stored task_anchor (session): {task_anchor}")
    else:
        parts.append("Stored task_anchor (session): (none — free exploration)")
    if turn_log_excerpt:
        parts.append("Recent turn log (newest last, one JSON object per line):\n" + turn_log_excerpt)
    if recent_actions:
        parts.append(
            "Recent action labels (rolling): "
            + json.dumps(recent_actions)
            + " Avoid useless repetition; if stuck, use look_around, reorient, wait, or CLEAR the anchor per rules."
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
                "If execution is likely still in progress, use wait with args_json \"{}\" or a brief say; "
                "do not repeat the same motion tool unless the view clearly shows it failed."
            )
    if goal:
        parts.append(f"Current goal: {goal}")
    if last_outcome:
        parts.append(f"Last outcome / human feedback: {last_outcome}")
    parts.append("What should the human do next?")
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
    resp = client.models.generate_content(
        model=_model_id(),
        contents=contents,
        config=config,
    )
    if resp.parsed is not None and isinstance(resp.parsed, AgentResponse):
        return resp.parsed
    raw = (resp.text or "").strip()
    if not raw:
        raise RuntimeError("empty Gemini response")
    data = json.loads(raw)
    return AgentResponse.model_validate(data)


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
