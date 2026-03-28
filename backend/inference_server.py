"""
FastAPI WebSocket server: WebP or JPEG frames in, JSON detections + depth out.
Per-session LATEST_STATE for agent (decoupled from Gemini latency).
Run: uvicorn inference_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Load backend/.env so vars work without exporting in the shell (see .env.example).
load_dotenv(Path(__file__).resolve().parent / ".env")

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from agent_attention import prioritize_grounded_for_model
from agent_gemini import format_action_label, run_agent
from agent_tools import action_to_api_dict
from executors import VISION, shutdown_pools
from perception_postprocess import postprocess_frame_for_agent
from vision_pipeline import VisionPipeline, pick_device
from voice_gate import compute_voice_gate
from world_state import (
    commit_voice_gate_result,
    copy_snapshot,
    finish_agent_work,
    get_memory_for_prompt,
    publish_latest,
    try_begin_agent_work,
    update_memory_after_agent_success,
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


@app.get("/v1/depth_preview")
async def depth_preview(
    session_id: str | None = Query(
        None,
        description="Same id as WebSocket ?session_id= (default: default)",
    ),
    v: int | None = Query(
        None,
        description="Cache-bust version from client (e.g. infer frame counter).",
    ),
) -> Response:
    """
    Latest MiDaS depth as JPEG bytes from session state (same source as WebSocket depth_jpeg_b64).
    Use for direct <img src> when JSON-in-WebSocket is undesirable.
    """
    del v  # cache-bust only; no server-side use
    sid = _session_id(session_id)
    snap = await copy_snapshot(sid)
    if snap is None:
        raise HTTPException(
            status_code=404,
            detail="No perception state for this session yet; stream frames on /ws/infer first.",
        )
    raw_b64 = snap.detections.get("depth_jpeg_b64")
    if not isinstance(raw_b64, str) or not raw_b64.strip():
        raise HTTPException(
            status_code=404,
            detail="Depth preview not available (set INCLUDE_DEPTH_PREVIEW=1 on the inference server).",
        )
    try:
        raw = base64.b64decode(raw_b64)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Invalid depth preview data: {e}") from e
    if not raw:
        raise HTTPException(status_code=404, detail="Empty depth preview.")
    return Response(
        content=raw,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.post("/v1/agent/step")
async def agent_step(
    session_id: str | None = Query(
        None,
        description="Same id as WebSocket ?session_id= (default: default)",
    ),
    goal: str | None = Query(None),
    last_outcome: str | None = Query(None),
    payload: Optional[dict[str, Any]] = Body(default=None),
) -> dict:
    is_tts_playing = bool(payload.get("is_tts_playing")) if isinstance(payload, dict) else False
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

    memory = await get_memory_for_prompt(sid)
    effective_anchor = str(memory.get("task_anchor") or "")
    inferred_held = str(memory.get("inferred_held_object") or "")
    grounded_for_model = prioritize_grounded_for_model(
        snap.grounded,
        effective_anchor,
        inferred_held_object=inferred_held,
    )
    recent = list(memory.get("recent_action_labels") or [])
    sec_since = memory.get("seconds_since_last_step")
    if sec_since is not None and not isinstance(sec_since, (int, float)):
        sec_since = None
    last_issued = list(memory.get("last_issued_action_labels") or [])
    turn_excerpt = str(memory.get("turn_log_excerpt") or "")
    vlst = str(memory.get("voice_last_speak_text") or "")
    vlsm = float(memory.get("voice_last_speak_monotonic") or 0.0)
    vlmf = str(memory.get("voice_last_motor_fingerprint") or "")

    try:
        result = await run_agent(
            snap.frame_webp,
            grounded_for_model,
            recent_actions=recent,
            goal=goal,
            last_outcome=last_outcome,
            task_anchor=effective_anchor,
            turn_log_excerpt=turn_excerpt,
            seconds_since_last_step=sec_since,
            last_issued_action_labels=last_issued,
            inferred_held_object=inferred_held,
            session_id=sid,
        )
    except Exception as e:
        await finish_agent_work(sid, success=False)
        raise HTTPException(status_code=502, detail=f"Gemini error: {e}") from e

    labels = [format_action_label(a) for a in result.actions]
    model_instruction = (result.instruction or "").strip()
    tts_line = model_instruction
    dashboard_say = (result.thought or "").strip() or model_instruction
    stored_anchor, inferred_held_out = await update_memory_after_agent_success(
        sid,
        thought=(result.thought or "").strip(),
        instruction=tts_line,
        action_labels=labels,
        task_anchor_model=result.task_anchor or "",
        actions=result.actions,
    )
    voice_res = compute_voice_gate(
        anchor_before=effective_anchor,
        anchor_after=stored_anchor,
        actions=result.actions,
        tts_line=tts_line,
        voice_last_speak_text=vlst,
        voice_last_speak_monotonic=vlsm,
        voice_last_motor_fingerprint=vlmf,
        is_tts_playing=is_tts_playing,
    )
    await commit_voice_gate_result(sid, voice_res)
    await finish_agent_work(sid, success=True)
    vp = voice_res.payload
    return {
        "say": dashboard_say,
        "instruction": result.instruction or "",
        "spoken_line": tts_line,
        "actions": [action_to_api_dict(a) for a in result.actions],
        "state_version": snap.version,
        "task_anchor": stored_anchor,
        "inferred_held_object": inferred_held_out,
        "voice": {
            "speak": vp.speak,
            "should_speak": vp.should_speak,
            "phase": vp.phase,
            "supersede": vp.supersede,
        },
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
                frame = await loop.run_in_executor(VISION, decode_image_bytes, data)
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
                    out = await loop.run_in_executor(VISION, pipeline.infer, frame)
                except Exception as e:
                    if not await _send_json_safe(
                        ws,
                        {"error": str(e), "w": 0, "h": 0, "detections": []},
                    ):
                        return
                    continue

            def _cpu() -> tuple[bytes, list]:
                return postprocess_frame_for_agent(frame, out)

            # Publish LATEST_STATE before notifying the client so POST /v1/agent/step
            # never races a 409 (client used to infer JSON before publish_latest finished).
            try:
                webp_bytes, grounded = await loop.run_in_executor(VISION, _cpu)
                await publish_latest(sid, webp_bytes, out, grounded)
            except Exception:
                pass

            if not await _send_json_safe(ws, out):
                return
    except WebSocketDisconnect:
        return


@app.on_event("startup")
async def startup() -> None:
    lp = asyncio.get_event_loop()
    await lp.run_in_executor(VISION, get_pipeline_for_preset, "oiv7")


@app.on_event("shutdown")
def shutdown() -> None:
    shutdown_pools()
