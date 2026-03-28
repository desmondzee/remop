"""
Structured agent output schema and action registry for Gemini (response_schema).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

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
    args: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    say: str = Field(description="Short human-facing instruction (one sentence).")
    actions: list[Action] = Field(default_factory=list)


ACTION_REGISTRY_PROMPT = """
Valid action names and typical args (JSON object per action):
- move_forward: {"steps": number} optional, ~0.5m per step
- move_backward: {"steps": number}
- turn_left: {"degrees": number}
- turn_right: {"degrees": number}
- pick_up: {"target": string} class or description from grounded objects
- drop: {}
- place: {"target": string, "near": string} optional
- look_around: {}
- wait: {"seconds": number} optional
Use empty args {} when no parameters apply.
"""
