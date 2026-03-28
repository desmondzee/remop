"""
Reorder/truncate grounded detections for the LLM when a task_anchor is active.
Urgent classes always appear first (safety). Peripheral rest capped by env.
"""

from __future__ import annotations

import os
from typing import Any

URGENT_CLASSES: frozenset[str] = frozenset({"person", "cat", "dog"})


def _anchor_top_match() -> int:
    return max(0, int(os.environ.get("AGENT_ANCHOR_TOP_MATCH", "3")))


def _anchor_top_rest() -> int:
    return max(0, int(os.environ.get("AGENT_ANCHOR_TOP_REST", "2")))


def _class_key(g: dict[str, Any]) -> str:
    return str(g.get("class", "")).lower()


def _conf(g: dict[str, Any]) -> float:
    try:
        return float(g.get("conf", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _anchor_tokens(anchor_lower: str) -> set[str]:
    return {t for t in anchor_lower.split() if len(t) >= 2}


def _detection_matches_anchor(g: dict[str, Any], tokens: set[str]) -> bool:
    if not tokens:
        return False
    c = _class_key(g)
    if not c:
        return False
    for t in tokens:
        if t in c or c in t:
            return True
    return False


def _detection_matches_held(g: dict[str, Any], held_lower: str) -> bool:
    if not held_lower:
        return False
    c = _class_key(g)
    if not c:
        return False
    return c == held_lower or held_lower in c or c in held_lower


def _ground_merge_key(g: dict[str, Any]) -> tuple[Any, ...]:
    """Stable key for unioning the same physical object across frames."""
    c = str(g.get("class", "") or "").strip().lower()
    try:
        cx = round(float(g.get("cx", 0.0)), 3)
        cy = round(float(g.get("cy", 0.0)), 3)
        cz = round(float(g.get("cz", 0.0)), 3)
    except (TypeError, ValueError):
        cx, cy, cz = 0.0, 0.0, 0.0
    return (c, cx, cy, cz)


def merge_into_grounded_accumulator(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    max_items: int,
) -> list[dict[str, Any]]:
    """
    Union detector rows across frames: same class + rounded (cx,cy,cz) keeps the
    highest-confidence row. Then cap length by confidence (retain strongest signal).
    """
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for g in existing + incoming:
        k = _ground_merge_key(g)
        prev = best.get(k)
        if prev is None or _conf(g) > _conf(prev):
            best[k] = dict(g)
    merged = list(best.values())
    merged.sort(key=_conf, reverse=True)
    if max_items > 0 and len(merged) > max_items:
        merged = merged[:max_items]
    return merged


def _dedupe_preserve_order(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for g in items:
        key = (g.get("class"), g.get("cx"), g.get("cy"), g.get("cz"))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(g))
    return out


def prioritize_grounded_for_model(
    grounded: list[dict[str, Any]],
    task_anchor: str,
    *,
    inferred_held_object: str = "",
    top_match: int | None = None,
    top_rest: int | None = None,
) -> list[dict[str, Any]]:
    """
    If task_anchor is empty, return a shallow copy of grounded (no reorder).
    Else: all urgent-class detections first (by conf), then up to top_match
    anchor-aligned detections, then up to top_rest other detections (by conf).
    When inferred_held_object is set, prefer non-held classes in the rest slots
    so placement anchors (table, bin) surface before the held item blob.
    """
    gcopy = [dict(x) for x in grounded]
    anchor = (task_anchor or "").strip().lower()
    if not anchor:
        return gcopy

    tm = _anchor_top_match() if top_match is None else top_match
    tr = _anchor_top_rest() if top_rest is None else top_rest
    tokens = _anchor_tokens(anchor)

    urgent: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for item in gcopy:
        if _class_key(item) in URGENT_CLASSES:
            urgent.append(item)
        else:
            other.append(item)

    urgent.sort(key=_conf, reverse=True)

    matches = [x for x in other if _detection_matches_anchor(x, tokens)]
    non_matches = [x for x in other if not _detection_matches_anchor(x, tokens)]
    matches.sort(key=_conf, reverse=True)
    non_matches.sort(key=_conf, reverse=True)

    held_l = (inferred_held_object or "").strip().lower()
    if held_l and tr > 0:
        rest_other = [x for x in non_matches if not _detection_matches_held(x, held_l)]
        rest_heldish = [x for x in non_matches if _detection_matches_held(x, held_l)]
        rest_other.sort(key=_conf, reverse=True)
        rest_heldish.sort(key=_conf, reverse=True)
        rest_pool = rest_other + rest_heldish
        rest_sel = rest_pool[:tr]
    else:
        rest_sel = non_matches[:tr]

    selected = urgent + matches[:tm] + rest_sel
    return _dedupe_preserve_order(selected)
