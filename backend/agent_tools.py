"""
Structured agent output schema and action registry for Gemini (response_schema).

Gemini structured output does not allow JSON Schema additionalProperties; avoid dict[str, Any]
on Pydantic models (it emits additionalProperties). Use args_json: str for parameters instead.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

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

_THOUGHT_DESC = (
    "Operator dashboard line: status, reasoning, or wait-state copy. NOT read aloud. "
    "Conversational tone allowed. Do not duplicate the instruction field here unless you have extra context."
)

_INSTRUCTION_DESC = (
    "Preferred line for text-to-speech when non-empty. Disembodied director: second-person imperatives only "
    "(e.g. Turn left thirty degrees, Hold steady, Pick up the cup, Look around). "
    "BANNED in this field: the words I, me, my, we, let's, please, could you; question marks; "
    "narration of what you will do (e.g. I will look around). "
    "You may leave this empty when actions still carry the directive—the server will announce each tool in order."
)


class AgentResponse(BaseModel):
    thought: str = Field(default="", description=_THOUGHT_DESC)
    instruction: str = Field(default="", description=_INSTRUCTION_DESC)
    actions: list[Action] = Field(default_factory=list)
    task_anchor: str = Field(
        default="",
        description=_TASK_ANCHOR_DESC,
        max_length=512,
    )

    @model_validator(mode="before")
    @classmethod
    def _legacy_say_key(cls, data: Any) -> Any:
        if isinstance(data, dict) and "say" in data and "thought" not in data:
            out = {k: v for k, v in data.items() if k != "say"}
            out["thought"] = data.get("say", "") or ""
            return out
        return data

    @model_validator(mode="after")
    def _thought_default_from_instruction(self) -> "AgentResponse":
        t = (self.thought or "").strip()
        ins = (self.instruction or "").strip()
        if not t and ins:
            return self.model_copy(update={"thought": ins})
        return self


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


def _tts_phrase_for_action(action: Action) -> str:
    """Short imperative for TTS when the model omits instruction (one phrase per tool)."""
    args = action_args_dict(action)
    n = action.name

    def _int_key(k: str, default: int = 1) -> int:
        v = args.get(k)
        if isinstance(v, bool) or v is None:
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    if n == "move_forward":
        steps = max(1, _int_key("steps", 1))
        return "Step forward." if steps == 1 else f"Take {steps} steps forward."
    if n == "move_backward":
        steps = max(1, _int_key("steps", 1))
        return "Step backward." if steps == 1 else f"Take {steps} steps backward."
    if n == "turn_left":
        deg = _int_key("degrees", 30)
        return f"Turn left {deg} degrees."
    if n == "turn_right":
        deg = _int_key("degrees", 30)
        return f"Turn right {deg} degrees."
    if n == "pick_up":
        t = args.get("target")
        if isinstance(t, str) and t.strip():
            return f"Pick up the {t.strip()}."
        return "Pick up the object."
    if n == "drop":
        return "Put it down."
    if n == "place":
        tgt = args.get("target")
        near = args.get("near")
        ts = tgt.strip() if isinstance(tgt, str) and tgt.strip() else "it"
        if isinstance(near, str) and near.strip():
            return f"Place the {ts} near the {near.strip()}."
        return f"Place the {ts}."
    if n == "look_around":
        return "Look around."
    if n == "wait":
        sec = args.get("seconds")
        if isinstance(sec, (int, float)) and sec > 0:
            return "Hold steady."
        return "Hold steady."
    return ""


def tts_line_from_actions(actions: list[Action]) -> str:
    """
    Build a single TTS string from the action list (in order).
    Used when instruction is empty so every tool call is still vocalized.
    """
    if not actions:
        return ""
    parts = [_tts_phrase_for_action(a) for a in actions]
    parts = [p for p in parts if p]
    return ". ".join(parts).strip()


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
- pick_up: args_json "{\\"target\\":\\"cup\\"}" — request the human to grasp; target MUST match a visible grounded "class" string. This is a request, not confirmation that the object is already in hand.
- drop: args_json "{}" — release whatever is held; clears inferred holding state.
- place: args_json "{\\"target\\":\\"shoe\\",\\"near\\":\\"table\\"}" — put the currently held object near a surface or container. "near" MUST be an exact class string from the current grounded list (e.g. table, bin, shelf), not vague words like "away" or "somewhere".
- look_around: args_json "{}"
- wait: args_json "{}" or "{\\"seconds\\":2}"
Use "{}" when there are no parameters.

wait: Hold your position when a physical action is still in progress or the scene is settling (e.g. after motion).
Do not use wait instead of look_around, turning, or CLEAR when the anchored object is missing from the detection list.

pick_up vs holding: If the session line says you are already inferring a held target and that class still appears in detections, treat that detection as likely the object in hand (first-person view). Do NOT issue another pick_up for that same held class until place or drop has cleared it.

task_anchor field:
- "" = keep current session anchor.
- CLEAR = clear anchor (free exploration).
- Other text = new anchor (use exact class names from grounded objects).

JSON fields:
- thought: dashboard / operator only (not TTS).
- instruction: preferred spoken line; if empty, the server still speaks a short line derived from each action in order.
"""

