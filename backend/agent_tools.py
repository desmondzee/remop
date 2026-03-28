"""
Structured agent output schema and action registry for Gemini (response_schema).

Gemini structured output does not allow JSON Schema additionalProperties; avoid dict[str, Any]
on Pydantic models (it emits additionalProperties). Use args_json: str for parameters instead.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

ActionName = Literal[
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "pick_up",
    "drop",
    "place",
    "look_around",
    "wait",
]


class Action(BaseModel):
    name: ActionName
    args_json: str = Field(
        default="{}",
        description='Parameters as one JSON object encoded in a string only, e.g. {} or {"degrees":30} or {"target":"cup"}. No markdown, no trailing commentary.',
    )


_TASK_ANCHOR_DESC = (
    "Session task headline. Output empty string \"\" to keep the current anchor unchanged. "
    'Output exactly CLEAR (no quotes in the value) to clear the anchor and return to free exploration. '
    "Any other short string replaces the anchor; use exact object class names from the grounded list "
    '(e.g. "cup"), not synonyms like "drink".'
)


class AgentResponse(BaseModel):
    say: str = Field(description="Short human-facing instruction (one sentence).")
    actions: list[Action] = Field(default_factory=list)
    task_anchor: str = Field(
        default="",
        description=_TASK_ANCHOR_DESC,
        max_length=512,
    )


def action_args_dict(action: Action) -> dict[str, Any]:
    raw = (action.args_json or "").strip() or "{}"
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}


def action_to_api_dict(action: Action) -> dict[str, Any]:
    """API shape expected by the frontend: name + args object."""
    return {"name": action.name, "args": action_args_dict(action)}


MOTION_ACTION_NAMES: frozenset[str] = frozenset(
    {"move_forward", "move_backward", "turn_left", "turn_right"}
)


ACTION_REGISTRY_PROMPT = """
Each action uses args_json: a single string containing a JSON object (not arbitrary keys in the schema).
Examples:
- move_forward: args_json "{\\"steps\\":1}"
- move_backward: args_json "{\\"steps\\":1}"
- turn_left: args_json "{\\"degrees\\":30}"
- turn_right: args_json "{\\"degrees\\":30}"
- pick_up: args_json "{\\"target\\":\\"cup\\"}"
- drop: args_json "{}"
- place: args_json "{\\"target\\":\\"bottle\\",\\"near\\":\\"bin\\"}"
- look_around: args_json "{}"
- wait: args_json "{}" or "{\\"seconds\\":2}"
Use "{}" when there are no parameters.

wait: Hold your position when a physical action is still in progress or the scene is settling (e.g. after motion).
Do not use wait instead of look_around, turning, or CLEAR when the anchored object is missing from the detection list.

task_anchor field:
- "" = keep current session anchor.
- CLEAR = clear anchor (free exploration).
- Other text = new anchor (use exact class names from grounded objects).
"""
