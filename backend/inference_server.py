"""
FastAPI WebSocket server: WebP or JPEG frames in, JSON detections + depth out.
Run: uvicorn inference_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import os
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from vision_pipeline import VisionPipeline, pick_device

try:
    from turbojpeg import TurboJPEG

    _turbo = TurboJPEG()
except Exception:
    _turbo = None

JPEG_SOI = b"\xff\xd8"
WEBP_MARKER = b"WEBP"

# Optional leading byte: 0x01 = JPEG, 0x02 = WebP (client may send for clarity)
FMT_JPEG = 1
FMT_WEBP = 2


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Return BGR uint8 image."""
    if len(data) < 2:
        raise ValueError("empty image")
    offset = 0
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


app = FastAPI(title="remop inference")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: VisionPipeline | None = None
_infer_lock = asyncio.Lock()


def get_pipeline() -> VisionPipeline:
    global _pipeline
    if _pipeline is None:
        model = os.environ.get("YOLO_MODEL", "yolo26n.pt")
        _pipeline = VisionPipeline(model, device=pick_device())
    return _pipeline


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/infer")
async def ws_infer(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_event_loop()
    try:
        while True:
            data = await ws.receive_bytes()
            async with _infer_lock:
                try:
                    frame = await loop.run_in_executor(None, decode_image_bytes, data)
                    pipeline = get_pipeline()
                    out = await loop.run_in_executor(None, pipeline.infer, frame)
                    await ws.send_json(out)
                except Exception as e:
                    await ws.send_json(
                        {"error": str(e), "w": 0, "h": 0, "detections": []}
                    )
    except WebSocketDisconnect:
        return


@app.on_event("startup")
async def startup() -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_pipeline)
