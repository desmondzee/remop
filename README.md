# remop

Monorepo for the remop project: a Next.js frontend and a Python backend for perception experiments (Ultralytics YOLO26 plus MiDaS monocular depth).

## Repository layout

| Path | Description |
|------|-------------|
| [`frontend-remop/`](frontend-remop/) | Next.js 16 app (React 19, Tailwind CSS 4) — browser camera, WebSocket client, canvas overlay |
| [`backend/`](backend/) | Python: OpenCV webcam CLI (`yolo26_depth_webcam.py`), shared `vision_pipeline.py`, FastAPI WebSocket server (`inference_server.py`) |

## Browser + inference (recommended dev flow)

You need **two processes**: the Python inference server and the Next.js dev server.

### 1. Backend (FastAPI + WebSocket)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Always use the venv’s interpreter (avoids global uvicorn without torch):
python -m uvicorn inference_server:app --host 0.0.0.0 --port 8000
```

macOS/Linux shortcut (uses `.venv/bin/python` if present):

```bash
cd backend && chmod +x serve.sh && ./serve.sh
```

**Troubleshooting `ModuleNotFoundError: No module named 'torch'`:** You started a **`uvicorn` executable from outside the venv** (e.g. `/Library/Frameworks/.../bin/uvicorn`). Use **`python -m uvicorn`** after `source .venv/bin/activate`, or `./serve.sh`. With **uv**, install into the same env: `uv pip install -r requirements.txt` while the venv is activated (or `uv pip install -r requirements.txt --python .venv/bin/python`).

- **Health:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Frames:** WebSocket `ws://127.0.0.1:8000/ws/infer` — binary **WebP** or **JPEG** (optional leading byte: `0x02` WebP, `0x01` JPEG); response JSON `{ w, h, detections[] }` with normalized boxes and `rel_depth`.
- **CORS:** set `INFERENCE_CORS_ORIGINS` (comma-separated) if the app is not on `localhost:3000`.
- **Model:** `YOLO_MODEL` env overrides weights (default `yolo26n.pt`).
- **MiDaS stride:** `MIDAS_EVERY_N=2` reuses the previous depth map on alternating frames (faster, slightly stale depth).

**PyTurboJPEG** speeds up **JPEG** decode. It expects **libjpeg-turbo** on the system (e.g. macOS: `brew install jpeg-turbo`). If TurboJPEG is missing, the server falls back to OpenCV. **WebP** frames are decoded with **Pillow**.

### 2. Frontend (Next.js)

```bash
cd frontend-remop
cp .env.example .env.local   # optional: set NEXT_PUBLIC_INFERENCE_WS_URL
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Use **Start camera** (required for Safari / user-gesture rules). The UI sends **WebP** when supported, otherwise **JPEG**, using ping-pong pacing with `requestAnimationFrame`.

Use **`wss://`** and **HTTPS** in production so the camera API works outside localhost.

## Local OpenCV demo (no browser)

```bash
cd backend
source .venv/bin/activate
python yolo26_depth_webcam.py
```

Press `q` to quit. Flags: `--model`, `--source`, `--no-depth-panel`.

**Note:** First run downloads YOLO26 and MiDaS weights; `timm` is pinned in `requirements.txt` for MiDaS v3.1 compatibility.

## Optional: lower YOLO latency on Mac

Exporting the detector to **CoreML** or **ONNX** can reduce latency on Apple Silicon; see the Ultralytics [export docs](https://docs.ultralytics.com/modes/export/) (`yolo export model=yolo26n.pt format=coreml` / `format=onnx`). The stock server loads `.pt` weights.

## License

This project is licensed under the MIT License—see [LICENSE](LICENSE).
