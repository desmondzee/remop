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

from vision_pipeline import VisionPipeline, depth_to_colormap_bgr, pick_device


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
        pipeline = VisionPipeline(args.model, device=device)
    except Exception as e:
        print(f"Error loading models: {e}", file=sys.stderr)
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

            try:
                _payload, result, depth = pipeline.infer_with_result(frame)
            except Exception as e:
                print(f"Inference error: {e}", file=sys.stderr)
                continue

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
                    h, w = frame.shape[:2]
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
