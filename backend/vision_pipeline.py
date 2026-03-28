"""
Shared Ultralytics YOLO detection + MiDaS 3.1 inference for CLI webcam and FastAPI WebSocket server.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from ultralytics import YOLO

MIDAS_HUB_REPO = "isl-org/MiDaS:v3_1"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_midas(device: torch.device) -> tuple[torch.nn.Module, object]:
    kwargs = {"trust_repo": True}
    model = torch.hub.load(MIDAS_HUB_REPO, "DPT_SwinV2_T_256", pretrained=True, **kwargs)
    model.to(device)
    model.eval()
    tfm = torch.hub.load(MIDAS_HUB_REPO, "transforms", **kwargs)
    transform = tfm.swin256_transform
    return model, transform


@torch.inference_mode()
def infer_depth_map(
    frame_bgr: np.ndarray,
    model: torch.nn.Module,
    transform,
    device: torch.device,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    batch = transform(rgb).to(device)
    pred = model(batch)
    if isinstance(pred, (list, tuple)):
        pred = pred[-1]
    pred = pred.squeeze()
    if pred.ndim != 2:
        pred = pred.reshape(pred.shape[-2], pred.shape[-1])
    pred_np = pred.detach().float().cpu().numpy()
    return cv2.resize(pred_np, (w, h), interpolation=cv2.INTER_LINEAR)


def depth_to_colormap_bgr(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    dmin, dmax = float(d.min()), float(d.max())
    if dmax - dmin < 1e-8:
        norm = np.zeros_like(d, dtype=np.uint8)
    else:
        norm = ((d - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)


def _detections_payload(result, depth: np.ndarray, w: int, h: int) -> dict[str, object]:
    payload: dict[str, object] = {"w": w, "h": h, "detections": []}
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return payload
    names = result.names
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    dets: list[dict[str, object]] = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = xyxy[i]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        u = int(np.clip(round(cx), 0, w - 1))
        v = int(np.clip(round(cy), 0, h - 1))
        ci = int(cls[i])
        label = names[ci] if isinstance(names, dict) else names[ci]
        dets.append(
            {
                "label": label,
                "conf": float(conf[i]),
                "x1": float(x1 / w),
                "y1": float(y1 / h),
                "x2": float(x2 / w),
                "y2": float(y2 / h),
                "cx": float(cx / w),
                "cy": float(cy / h),
                "rel_depth": float(depth[v, u]),
            }
        )
    payload["detections"] = dets
    return payload


class VisionPipeline:
    """Runs YOLO and MiDaS in parallel when both are needed; optional MiDaS stride."""

    def __init__(self, yolo_model_path: str, device: torch.device | None = None) -> None:
        self.device = device if device is not None else pick_device()
        self.yolo = YOLO(yolo_model_path)
        self.midas_model, self.midas_transform = load_midas(self.device)
        self._midas_every_n = max(1, int(os.environ.get("MIDAS_EVERY_N", "1")))
        self._frame_idx = 0
        self._last_depth: np.ndarray | None = None

    def _predict_yolo(self, frame_bgr: np.ndarray):
        return self.yolo.predict(frame_bgr, verbose=False)[0]

    def _predict_midas(self, frame_bgr: np.ndarray) -> np.ndarray:
        depth = infer_depth_map(frame_bgr, self.midas_model, self.midas_transform, self.device)
        self._last_depth = depth
        return depth

    def _run_models(self, frame_bgr: np.ndarray) -> tuple[object, np.ndarray]:
        h, w = frame_bgr.shape[:2]
        self._frame_idx += 1
        skip_midas = (
            self._midas_every_n > 1
            and (self._frame_idx % self._midas_every_n) != 0
            and self._last_depth is not None
            and self._last_depth.shape[0] == h
            and self._last_depth.shape[1] == w
        )
        if skip_midas:
            result = self._predict_yolo(frame_bgr)
            depth = self._last_depth
            assert depth is not None
            return result, depth
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_y = ex.submit(self._predict_yolo, frame_bgr)
            fut_m = ex.submit(self._predict_midas, frame_bgr)
            return fut_y.result(), fut_m.result()

    def infer(self, frame_bgr: np.ndarray) -> dict[str, object]:
        h, w = frame_bgr.shape[:2]
        result, depth = self._run_models(frame_bgr)
        return _detections_payload(result, depth, w, h)

    def infer_with_result(self, frame_bgr: np.ndarray) -> tuple[dict[str, object], object, np.ndarray]:
        """For local OpenCV UI: JSON payload + ultralytics result + depth map."""
        h, w = frame_bgr.shape[:2]
        result, depth = self._run_models(frame_bgr)
        return _detections_payload(result, depth, w, h), result, depth

    def infer_sequential(self, frame_bgr: np.ndarray) -> dict[str, object]:
        """Single-threaded path for debugging / parity checks."""
        h, w = frame_bgr.shape[:2]
        result = self._predict_yolo(frame_bgr)
        depth = self._predict_midas(frame_bgr)
        return _detections_payload(result, depth, w, h)
