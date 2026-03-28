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
        description='Parameters as one JSON object encoded in a string only, e.g. {} or {"degrees":30} or {"target":"cup","sector":"left"}. No markdown, no trailing commentary.',
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
    "The ONLY spoken line to the human for this tick. Disembodied director: second-person imperatives only "
    "(e.g. Turn left thirty degrees, Hold steady, Pick up the cup on the left, Look around). "
    "BANNED in this field: the words I, me, my, we, let's, please, could you; question marks; "
    "narration of what you will do (e.g. I will look around). "
    "If you output an empty string \"\", the human hears nothing this tick—leave empty during wait or when "
    "repeating an ongoing motion so the assistant does not babble."
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
        sec = args.get("sector")
        if isinstance(t, str) and t.strip():
            ts = t.strip()
            if isinstance(sec, str) and sec.strip():
                return f"Pick up the {ts} on the {sec.strip()}."
            return f"Pick up the {ts}."
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
    Optional helper for tests or tooling; the live agent path uses model instruction only (Silence is Golden).
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
- move_forward: args_json "{\\"steps\\":1}" — walk forward. Use to approach a target when its grounded "cz" is too small (object too far to interact); prefer look_around or turns to scan before walking if the scene is unknown.
- move_backward: args_json "{\\"steps\\":1}"
- turn_left: args_json "{\\"degrees\\":30}"
- turn_right: args_json "{\\"degrees\\":30}"
- pick_up: args_json "{\\"target\\":\\"cup\\",\\"sector\\":\\"left\\"}" — request the human to grasp. "target" MUST match a visible grounded "class" string. If multiple objects share the same class, you MUST include "sector" (left/center/right) copied from the grounded object you mean. This is a request, not confirmation the object is in hand. "sector" does not change the session held-object class (still the same target class).
- drop: args_json "{}" — release whatever is held; clears inferred holding state.
- place: args_json "{\\"target\\":\\"shoe\\",\\"near\\":\\"table\\"}" — put the currently held object near a surface or container. "near" MUST be an exact class string from the current grounded list (e.g. table, bin, shelf), not vague words like "away" or "somewhere". If "near" is NOT in the current grounded list, you CANNOT use place—use look_around or move_forward (per cz) until a valid surface/container appears in the list. Do not repeat the same place when "near" is missing or invalid.
- look_around: args_json "{}" — rotate in place to scan the room. Do not use this for locomotion.
- wait: args_json "{}" or "{\\"seconds\\":2}"
Use "{}" when there are no parameters.

wait: Hold your position when a physical action is still in progress or the scene is settling (e.g. after motion).
Do not use wait instead of look_around, turning, or CLEAR when the anchored object is missing from the detection list.

pick_up vs holding: Detections often keep listing the object after grasp. In first person, a visible hand grasping or contacting that object is the main cue pick_up is succeeding. If the session infers a held target and that class still appears in grounding, treat it as likely in hand—do not pick_up that class again until place or drop.

task_anchor field:
- "" = keep current session anchor.
- CLEAR = clear anchor (free exploration).
- Other text = new anchor (use exact class names from grounded objects).

JSON fields:
- thought: dashboard / operator only (not TTS).
- instruction: the ONLY line spoken to the human this tick. If empty "", there is no speech—leave empty during wait or when repeating an ongoing motion. When pick_up includes "sector", your instruction must match (e.g. Pick up the bottle on the left).
"""

