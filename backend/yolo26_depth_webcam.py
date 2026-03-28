#!/usr/bin/env python3
"""
Real-time webcam: Ultralytics YOLO26 detection plus MiDaS 3.1 DPT Swin2 Tiny 256 depth
sampled at each box center. Relative depth is disparity-style (larger ≈ closer).

Usage:
  cd backend && pip install -r requirements.txt
  python yolo26_depth_webcam.py [--model yolo26n.pt] [--source 0] [--no-depth-panel]

MiDaS is loaded from PyTorch Hub (isl-org/MiDaS:v3_1) with DPT_SwinV2_T_256; timm is pinned in
requirements.txt so checkpoint and Swin backbone shapes stay aligned.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# PyTorch Hub: pin MiDaS v3.1 tag so hub code matches dpt_swin2_tiny_256 weights (master can diverge).
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO26 + MiDaS 3.1 webcam depth at detections.")
    p.add_argument("--model", default="yolo26n.pt", help="Ultralytics YOLO26 weights or YAML.")
    p.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or video file path.",
    )
    p.add_argument(
        "--no-depth-panel",
        action="store_true",
        help="Do not show side-by-side depth colormap (RGB + boxes only).",
    )
    return p.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    src: str | int
    if source.isdigit():
        src = int(source)
    else:
        src = source
    cap = cv2.VideoCapture(src)
    return cap


def main() -> int:
    args = parse_args()
    device = pick_device()
    print(f"Using device: {device}", file=sys.stderr)

    cap = open_capture(args.source)
    if not cap.isOpened():
        print(f"Error: could not open video source {args.source!r}", file=sys.stderr)
        return 1

    try:
        yolo = YOLO(args.model)
    except Exception as e:
        print(f"Error loading YOLO model: {e}", file=sys.stderr)
        cap.release()
        return 1

    try:
        midas_model, midas_transform = load_midas(device)
    except Exception as e:
        print(
            f"Error loading MiDaS from hub ({MIDAS_HUB_REPO}). "
            f"Ensure timm is compatible (see requirements). Cause: {e}",
            file=sys.stderr,
        )
        cap.release()
        return 1

    window = "YOLO26 + MiDaS depth"
    last_print = time.monotonic()
    print_interval_s = 0.5

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("End of stream or read failure.", file=sys.stderr)
                break

            h, w = frame.shape[:2]
            results = yolo.predict(frame, verbose=False)
            result = results[0]

            try:
                depth = infer_depth_map(frame, midas_model, midas_transform, device)
            except Exception as e:
                print(f"MiDaS inference error: {e}", file=sys.stderr)
                depth = np.zeros((h, w), dtype=np.float32)

            vis = result.plot()

            if not args.no_depth_panel:
                depth_vis = depth_to_colormap_bgr(depth)
                if depth_vis.shape[:2] != vis.shape[:2]:
                    depth_vis = cv2.resize(
                        depth_vis,
                        (vis.shape[1], vis.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                display = np.hstack([vis, depth_vis])
            else:
                display = vis

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            now = time.monotonic()
            if now - last_print >= print_interval_s:
                last_print = now
                ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    print(f"[{ts}] (no detections)")
                else:
                    names = result.names
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls = boxes.cls.cpu().numpy().astype(int)
                    conf = boxes.conf.cpu().numpy()
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = xyxy[i]
                        u = int(round((x1 + x2) / 2.0))
                        v = int(round((y1 + y2) / 2.0))
                        u = int(np.clip(u, 0, w - 1))
                        v = int(np.clip(v, 0, h - 1))
                        ci = int(cls[i])
                        cname = names[ci] if isinstance(names, dict) else names[ci]
                        cf = float(conf[i])
                        rel = float(depth[v, u])
                        print(
                            f"[{ts}] Class: {cname} | Conf: {cf:.2f} | "
                            f"Center: (X:{u}, Y:{v}) | Rel_Depth: {rel:.4f}"
                        )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
