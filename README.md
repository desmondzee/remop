# remop

Monorepo for the remop project: a Next.js frontend and a Python backend for perception experiments (Ultralytics YOLO plus MiDaS monocular depth).

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
cp .env.example .env   # optional: edit .env for GEMINI_API_KEY, CORS, etc. (loaded on startup; no need to export in the shell)
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
- **Session:** Append **`?session_id=<uuid>`** (same id the frontend uses) so perception writes per-session **`LATEST_STATE`** for the agent. The browser also sends **`?model=oiv7`** or **`?model=coco`** as before.
- **Agent (Gemini):** `POST /v1/agent/step?session_id=<uuid>` — no second image upload. Body is optional JSON (empty POST is valid): `{ "is_tts_playing": true }` so the server can suppress non-urgent speech while audio is playing. Optional query params: `goal`, `last_outcome`. Requires **`GEMINI_API_KEY`**. Returns `{ say, actions, state_version, task_anchor, inferred_held_object, voice }` (`inferred_held_object` is derived from recent pick_up/place/drop tools; verify visually). Raw **`say`** is the model line each tick; **`voice`** is the emission gate for future TTS: **`speak`**, **`should_speak`**, **`phase`**, **`supersede`**. If **`should_speak`** and not **`supersede`**, queue after current audio; if **`supersede`**, barge-in (pause, clear queue, play now)—used when **`task_anchor`** changes (e.g. CLEAR). Env **`VOICE_MIN_DWELL_SEC`** (default `5`): periodic status line when anchor+motor unchanged. **409** / **429** / **503** as before.
- **Config:** Copy [`backend/.env.example`](backend/.env.example) to **`backend/.env`**. The server loads it automatically from the backend directory (via `python-dotenv`). Shell exports still override `.env` if set.
- **Agent env:** `GEMINI_AGENT_MODEL` (default `gemini-2.5-flash-lite`), `AGENT_MIN_INTERVAL_MS` (default `500`), `AGENT_IMAGE_MAX_EDGE` (default `512`), `AGENT_WEBP_QUALITY` (default `60`), **`AGENT_DEPTH_SCALE_K`** (static scale for `cz = rel_depth / K` — calibrate once; default `1.0`), optional `AGENT_TOP_N_DETECTIONS`, `AGENT_CLASS_ALLOWLIST` (comma-separated lowercase class names). **`HIGH_LEVEL_INTENTION_PROMPT`** — optional override for the household-tidying mission block in the Gemini system prompt (see `backend/agent_gemini.py` default if unset). **Memory / continuity:** `AGENT_MEMORY_TURNS`, `AGENT_MEMORY_MAX_CHARS`, `AGENT_MOTION_COOLDOWN_HINT_SEC`, `AGENT_TASK_ANCHOR_MAX_LEN`, `AGENT_HELD_OBJECT_MAX_LEN` (inferred held target from tools). **Voice gate:** `VOICE_MIN_DWELL_SEC`. **Perception:** `AGENT_DETECTION_MIN_CONF` (default `0.3`) filters low-confidence boxes before grounding. **Blinders** (when `task_anchor` is set): `AGENT_ANCHOR_TOP_MATCH`, `AGENT_ANCHOR_TOP_REST` (default `2`); urgent classes `person`/`cat`/`dog` always stay visible (`backend/agent_attention.py`). Gate logic: `backend/voice_gate.py`.
- **CORS:** set `INFERENCE_CORS_ORIGINS` in `.env` (comma-separated) if the app is not on `localhost:3000`.
- **Model:** The browser sends **`?model=oiv7`** or **`?model=coco`** on the inference WebSocket (see the Detector control in the UI). Weights: **oiv7** → `yolov8n-oiv7.pt` ([Open Images v7](https://docs.ultralytics.com/datasets/detect/open-images-v7/), 601 classes); **coco** → `yolov8n.pt` (MS COCO, 80 classes). Override paths with **`YOLO_MODEL_OIV7`**, **`YOLO_MODEL_COCO`**, or legacy **`YOLO_MODEL`** (same as oiv7 when `YOLO_MODEL_OIV7` is unset). **Objects365** has an official [dataset YAML](https://docs.ultralytics.com/datasets/detect/objects365/) for training, but no published `*-o365.pt` on Ultralytics assets.
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

**Agent TTS (instructions only):** Spoken audio uses **`voice.speak`** only when **`voice.should_speak`** is true (never raw **`say`**). **Start camera** unlocks audio on iOS/Safari: a silent Web Speech utterance plus **`AudioContext.resume()`** for optional neural playback. Default engine is **browser (Web Speech)**; choose **Neural (Kokoro)** in the UI for on-device synthesis (first load downloads model weights; WebGPU when available, else WASM). Env: [`frontend-remop/.env.example`](frontend-remop/.env.example) — `NEXT_PUBLIC_TTS_ENGINE`, `NEXT_PUBLIC_KOKORO_VOICE`. Kokoro currently runs on the **main thread** after dynamic import; heavy loads may briefly affect the camera loop.

Use **`wss://`** and **HTTPS** in production so the camera API works outside localhost.

## Local OpenCV demo (no browser)

```bash
cd backend
source .venv/bin/activate
python yolo26_depth_webcam.py
```

Press `q` to quit. Flags: `--model`, `--source`, `--no-depth-panel`.

**Note:** First run downloads YOLO and MiDaS weights; `timm` is pinned in `requirements.txt` for MiDaS v3.1 compatibility.

## Optional: lower YOLO latency on Mac

Exporting the detector to **CoreML** or **ONNX** can reduce latency on Apple Silicon; see the Ultralytics [export docs](https://docs.ultralytics.com/modes/export/) (`yolo export model=yolov8n-oiv7.pt format=coreml` / `format=onnx`). The stock server loads `.pt` weights.

## License

This project is licensed under the MIT License—see [LICENSE](LICENSE).
