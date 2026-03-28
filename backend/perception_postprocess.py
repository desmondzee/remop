"""CPU-side downscale + WebP for agent path (run after GPU infer lock released)."""

from __future__ import annotations

import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from grounding import ground_detections_payload


def _max_edge() -> int:
    return max(64, int(os.environ.get("AGENT_IMAGE_MAX_EDGE", "512")))


def _webp_quality() -> int:
    return max(1, min(100, int(os.environ.get("AGENT_WEBP_QUALITY", "60"))))


def frame_to_agent_webp(frame_bgr: np.ndarray) -> bytes:
    """Resize so longest edge == max_edge (if larger); encode WebP."""
    max_edge = _max_edge()
    h, w = frame_bgr.shape[:2]
    scale = max_edge / max(h, w)
    if scale < 1.0:
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        small = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        small = frame_bgr
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(np.asarray(rgb))
    buf = BytesIO()
    im.save(buf, format="WEBP", quality=_webp_quality())
    return buf.getvalue()


def postprocess_frame_for_agent(
    frame_bgr: np.ndarray,
    detections_payload: dict,
) -> tuple[bytes, list[dict]]:
    webp = frame_to_agent_webp(frame_bgr)
    grounded = ground_detections_payload(detections_payload)
    return webp, grounded
