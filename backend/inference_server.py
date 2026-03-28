"""
FastAPI WebSocket server: WebP or JPEG frames in, JSON detections + depth out.
Per-session LATEST_STATE for agent (decoupled from Gemini latency).
Run: uvicorn inference_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import os
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from agent_gemini import format_action_label, run_agent
from perception_postprocess import postprocess_frame_for_agent
from vision_pipeline import VisionPipeline, pick_device
from world_state import (
    append_recent_actions,
    copy_snapshot,
    finish_agent_work,
    get_recent_actions,
    publish_latest,
    try_begin_agent_work,
)

try:
    from turbojpeg import TurboJPEG

    _turbo = TurboJPEG()
except Exception:
    _turbo = None

JPEG_SOI = b"\xff\xd8"
WEBP_MARKER = b"WEBP"

FMT_JPEG = 1
FMT_WEBP = 2


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Return BGR uint8 image."""
    if len(data) < 2:
        raise ValueError("empty image")
    if data[0] in (FMT_JPEG, FMT_WEBP):
        fmt = data[0]
        payload = data[1:]
        if fmt == FMT_JPEG:
            return _decode_jpeg(payload)
        return _decode_webp(payload)
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == WEBP_MARKER:
        return _decode_webp(data)
    if data[:2] == JPEG_SOI:
        return _decode_jpeg(data)
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("could not decode image bytes")
    return arr


def _decode_jpeg(buf: bytes) -> np.ndarray:
    if _turbo is not None:
        try:
            rgb = _turbo.decode(buf)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    arr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("JPEG decode failed")
    return arr


def _decode_webp(buf: bytes) -> np.ndarray:
    im = Image.open(BytesIO(buf)).convert("RGB")
    rgb = np.asarray(im)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _parse_origins() -> list[str]:
    raw = os.environ.get("INFERENCE_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]


def _agent_min_interval_ms() -> float:
    return max(0.0, float(os.environ.get("AGENT_MIN_INTERVAL_MS", "500")))


def _session_id(raw: str | None) -> str:
    s = (raw or "").strip()
    return s if s else "default"


app = FastAPI(title="remop inference")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_infer_lock = asyncio.Lock()
MODEL_PRESETS: dict[str, str] = {
    "oiv7": os.environ.get("YOLO_MODEL_OIV7")
    or os.environ.get("YOLO_MODEL", "yolov8n-oiv7.pt"),
    "coco": os.environ.get("YOLO_MODEL_COCO", "yolov8n.pt"),
}
_pipelines: dict[str, VisionPipeline] = {}


def _normalize_model_query(raw: str | None) -> str:
    key = (raw or "oiv7").strip().lower()
    return key if key in MODEL_PRESETS else "oiv7"


def get_pipeline_for_preset(preset: str) -> VisionPipeline:
    key = _normalize_model_query(preset)
    path = MODEL_PRESETS[key]
    if path not in _pipelines:
        _pipelines[path] = VisionPipeline(path, device=pick_device())
    return _pipelines[path]


async def _send_json_safe(ws: WebSocket, payload: dict) -> bool:
    try:
        await ws.send_json(payload)
        return True
    except Exception:
        return False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/agent/step")
async def agent_step(
    session_id: str | None = Query(
        None,
        description="Same id as WebSocket ?session_id= (default: default)",
    ),
    goal: str | None = Query(None),
    last_outcome: str | None = Query(None),
) -> dict:
    sid = _session_id(session_id)
    snap = await copy_snapshot(sid)
    if snap is None:
        raise HTTPException(
            status_code=409,
            detail="No perception state for this session yet; send frames on /ws/infer first.",
        )
    allowed, reason = await try_begin_agent_work(sid, _agent_min_interval_ms())
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Agent rate limited or busy: {reason}",
        )
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        await finish_agent_work(sid, success=False)
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured on the server.",
        )

    recent = await get_recent_actions(sid)
    try:
        result = await run_agent(
            snap.frame_webp,
            snap.grounded,
            recent_actions=recent,
            goal=goal,
            last_outcome=last_outcome,
        )
    except Exception as e:
        await finish_agent_work(sid, success=False)
        raise HTTPException(status_code=502, detail=f"Gemini error: {e}") from e

    labels = [format_action_label(a) for a in result.actions]
    await append_recent_actions(sid, labels)
    await finish_agent_work(sid, success=True)
    return {
        "say": result.say,
        "actions": [a.model_dump() for a in result.actions],
        "state_version": snap.version,
    }


@app.websocket("/ws/infer")
async def ws_infer(ws: WebSocket) -> None:
    await ws.accept()
    model_preset = _normalize_model_query(ws.query_params.get("model"))
    sid = _session_id(ws.query_params.get("session_id"))
    loop = asyncio.get_event_loop()
    try:
        while True:
            data = await ws.receive_bytes()
            try:
                frame = await loop.run_in_executor(None, decode_image_bytes, data)
            except Exception as e:
                if not await _send_json_safe(
                    ws,
                    {"error": str(e), "w": 0, "h": 0, "detections": []},
                ):
                    return
                continue

            async with _infer_lock:
                try:
                    pipeline = get_pipeline_for_preset(model_preset)
                    out = await loop.run_in_executor(None, pipeline.infer, frame)
                except Exception as e:
                    if not await _send_json_safe(
                        ws,
                        {"error": str(e), "w": 0, "h": 0, "detections": []},
                    ):
                        return
                    continue

            if not await _send_json_safe(ws, out):
                return

            def _cpu() -> tuple[bytes, list]:
                return postprocess_frame_for_agent(frame, out)

            try:
                webp_bytes, grounded = await loop.run_in_executor(None, _cpu)
                await publish_latest(sid, webp_bytes, out, grounded)
            except Exception:
                pass
    except WebSocketDisconnect:
        return


@app.on_event("startup")
async def startup() -> None:
    lp = asyncio.get_event_loop()
    await lp.run_in_executor(None, get_pipeline_for_preset, "oiv7")
