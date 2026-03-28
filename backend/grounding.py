"""
Map detector JSON + MiDaS rel_depth into compact grounded objects for the LLM.
cz = rel_depth / K with static K (env); no per-frame min/max normalization.
"""

from __future__ import annotations

import os
from typing import Any


def _horizontal_sector(cx: float, left: float, right: float) -> str:
    if cx < left:
        return "left"
    if cx > right:
        return "right"
    return "center"


def _vertical_band(cy: float, low: float, high: float) -> str:
    if cy < low:
        return "upper"
    if cy > high:
        return "lower"
    return "middle"


def _depth_scale_k() -> float:
    raw = os.environ.get("AGENT_DEPTH_SCALE_K", "1.0").strip()
    try:
        k = float(raw)
    except ValueError:
        return 1.0
    return k if k > 1e-9 else 1.0


def _class_allowlist() -> set[str] | None:
    raw = os.environ.get("AGENT_CLASS_ALLOWLIST", "").strip()
    if not raw:
        return None
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


def _sector_thresholds() -> tuple[float, float, float, float]:
    """left cx max, right cx min, upper cy max, lower cy min."""
    lx = float(os.environ.get("AGENT_SECTOR_LEFT_MAX", "0.33"))
    rx = float(os.environ.get("AGENT_SECTOR_RIGHT_MIN", "0.66"))
    uy = float(os.environ.get("AGENT_SECTOR_UPPER_MAX", "0.33"))
    ly = float(os.environ.get("AGENT_SECTOR_LOWER_MIN", "0.66"))
    return lx, rx, uy, ly


def _min_detection_conf() -> float:
    raw = os.environ.get("AGENT_DETECTION_MIN_CONF", "0.3").strip()
    try:
        v = float(raw)
    except ValueError:
        return 0.3
    return max(0.0, min(1.0, v))


def ground_detections_payload(
    payload: dict[str, Any],
    *,
    k: float | None = None,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """
    Build minimal objects: class, cx, cy, cz, sector, v_band, conf.
    """
    if k is None:
        k = _depth_scale_k()
    if top_n is None:
        top_n = max(1, int(os.environ.get("AGENT_TOP_N_DETECTIONS", "20")))
    allow = _class_allowlist()
    lx, rx, uy, ly = _sector_thresholds()

    dets: list[dict[str, Any]] = list(payload.get("detections") or [])
    min_cf = _min_detection_conf()
    dets = [d for d in dets if float(d.get("conf") or 0.0) >= min_cf]
    dets.sort(key=lambda d: float(d.get("conf") or 0.0), reverse=True)

    out: list[dict[str, Any]] = []
    for d in dets:
        if len(out) >= top_n:
            break
        label = str(d.get("label", ""))
        if allow is not None and label.lower() not in allow:
            continue
        cx = float(d.get("cx", 0.5))
        cy = float(d.get("cy", 0.5))
        rel_depth = float(d.get("rel_depth", 0.0))
        cz = rel_depth / k
        out.append(
            {
                "class": label,
                "cx": round(cx, 3),
                "cy": round(cy, 3),
                "cz": round(cz, 3),
                "sector": _horizontal_sector(cx, lx, rx),
                "v_band": _vertical_band(cy, uy, ly),
                "conf": round(float(d.get("conf", 0.0)), 3),
            }
        )
    return out
