"""Single-shot Gemini call with structured JSON output (AgentResponse)."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from google import genai
from google.genai import types

from agent_tools import ACTION_REGISTRY_PROMPT, AgentResponse

_client: genai.Client | None = None

DEFAULT_HIGH_LEVEL_INTENTION = """You are the high-level planner for a household tidying assistant (like a home robot).
Your ongoing intention is to keep living spaces orderly: notice clutter such as shoes, bottles, clothes, or loose objects on the floor or messy surfaces;
guide the human to pick items up when it makes sense, pair or group related things (e.g. shoes together), and place them neatly in sensible locations (shelves, bins, tables).
Work step by step; prefer safe, practical moves the human can actually perform."""

PERCEPTION_AND_SCHEMA_BLOCK = f"""You see one still frame (WebP) plus a JSON list of grounded objects. Fields use normalized image coordinates cx, cy in [0,1].
cz is a rough step-equivalent forward distance from monocular depth (calibrated constant K), NOT metric LIDAR—use only for ordering near vs far and coarse approach.

{ACTION_REGISTRY_PROMPT}

Output valid JSON only matching the schema: a short say string and a list of actions.
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


def _user_text(
    grounded: list[dict[str, Any]],
    recent_actions: list[str],
    goal: str | None,
    last_outcome: str | None,
) -> str:
    parts = [
        "Grounded objects (JSON array):",
        json.dumps(grounded, separators=(",", ":")),
    ]
    if recent_actions:
        parts.append(
            "Recent actions taken: "
            + json.dumps(recent_actions)
            + " Do not repeat recent actions uselessly; if stuck, use look_around or wait."
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
) -> AgentResponse:
    client = _get_client()
    user = _user_text(grounded, recent_actions, goal, last_outcome)
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
) -> AgentResponse:
    return await asyncio.to_thread(
        run_agent_sync,
        frame_webp,
        grounded,
        recent_actions=recent_actions,
        goal=goal,
        last_outcome=last_outcome,
    )


def format_action_label(a: Any) -> str:
    from agent_tools import Action

    if isinstance(a, Action):
        name = a.name
        args = a.args
    else:
        name = getattr(a, "name", "?")
        args = getattr(a, "args", {}) or {}
    if not args:
        return str(name)
    return f"{name}({json.dumps(args, separators=(',', ':'))})"
