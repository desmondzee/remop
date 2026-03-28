# remop

Monorepo for **live camera perception** (Ultralytics YOLO + MiDaS monocular depth) exposed over a **FastAPI WebSocket**, plus a **Next.js** client that draws detections and optionally runs a **Gemini-based “household tidying” agent** that proposes human-executable actions and gated text-to-speech.

This document is a **technical reference** for schemas, in-memory state, concurrency, environment variables, and how each module fits together.

---

## 1. High-level architecture

```mermaid
flowchart LR
  subgraph browser [Browser / Next.js]
    Cam[Camera + canvas capture]
    WS[WebSocket client]
    HTTP[fetch POST agent/step]
    TTS[Web Speech or OpenAI TTS]
  end
  subgraph backend [Python backend]
    Dec[decode_image_bytes]
    VP[VisionPipeline YOLO + MiDaS]
    PP[postprocess_frame_for_agent]
    WSrv[/ws/infer]
    Pub[publish_latest LATEST_STATE]
    Agent[run_agent Gemini]
    Gate[compute_voice_gate]
    Step[/v1/agent/step]
  end
  Cam -->|binary WebP/JPEG + fmt byte| WS
  WS --> WSrv
  WSrv --> Dec --> VP --> PP
  PP --> Pub
  Step --> Agent
  Pub --> Step
  Step --> Gate
  HTTP --> Step
  Gate --> HTTP
  HTTP --> TTS
```

**Separation of concerns**

- **Vision path** is tuned for **low latency**: decode → YOLO + depth → JSON to the browser; in parallel, CPU postprocess builds **WebP + grounded JSON** and atomically publishes **`LATEST_STATE`** for the session.
- **Agent path** is **slow** (Gemini round-trip). It **never blocks** the WebSocket loop beyond copying a snapshot; rate limiting and “busy” flags prevent overlapping calls per session.
- **TTS** is a **client concern**; the server only returns a **`voice`** object and accepts **`is_tts_playing`** so the voice gate can defer non-urgent speech while audio plays.

---

## 2. Repository layout

| Path | Role |
|------|------|
| [`frontend-remop/`](frontend-remop/) | Next.js app: camera, WebSocket streaming, agent polling, overlay canvas, TTS (`app/components/CameraOverlay.tsx`, `app/lib/agentTts.ts`, `app/api/tts/route.ts`). |
| [`backend/inference_server.py`](backend/inference_server.py) | FastAPI app: `GET /health`, `WebSocket /ws/infer`, `POST /v1/agent/step`. |
| [`backend/vision_pipeline.py`](backend/vision_pipeline.py) | Shared **YOLO** + **MiDaS 3.1** (`DPT_SwinV2_T_256` from `isl-org/MiDaS:v3_1`); builds the **detections JSON** for the wire. |
| [`backend/perception_postprocess.py`](backend/perception_postprocess.py) | After GPU work: resize frame → **WebP** for Gemini; run **grounding**. |
| [`backend/grounding.py`](backend/grounding.py) | Map raw detections → compact **`grounded`** list for the LLM (`cz`, sectors, filters). |
| [`backend/agent_attention.py`](backend/agent_attention.py) | When **`task_anchor`** is set, **reorder/cap** grounded objects (safety + focus). |
| [`backend/agent_tools.py`](backend/agent_tools.py) | **Pydantic** `AgentResponse` / `Action` schema, action→API mapping, default TTS phrases from actions. |
| [`backend/agent_gemini.py`](backend/agent_gemini.py) | **Google GenAI** `generate_content` with **`response_schema=AgentResponse`**, system prompt, user text assembly, **JSONL logging** hook. |
| [`backend/world_state.py`](backend/world_state.py) | Per-session **`LATEST_STATE`**, **turn log**, **task_anchor**, **inferred held object**, agent throttle/busy, voice-gate memory. |
| [`backend/voice_gate.py`](backend/voice_gate.py) | **`compute_voice_gate`**: when to speak, **supersede** vs queue, phases, co-gating with **`is_tts_playing`**. |
| [`backend/executors.py`](backend/executors.py) | Dedicated **`ThreadPoolExecutor`** pools: **`VISION`** vs **`AGENT`**. |
| [`backend/gemini_session_log.py`](backend/gemini_session_log.py) | Append **JSON Lines** under `backend/logs/` per session. |
| [`backend/yolo26_depth_webcam.py`](backend/yolo26_depth_webcam.py) | Standalone OpenCV demo using the same `VisionPipeline`. |

---

## 3. WebSocket: `/ws/infer`

### 3.1 Connection URL

- Path: **`/ws/infer`**
- Query parameters (read in [`inference_server.py`](backend/inference_server.py)):
  - **`model`**: `oiv7` (default) or `coco` → selects YOLO weights preset.
  - **`session_id`**: opaque string; default server-side session is **`"default"`** if missing or blank. Must match **`POST /v1/agent/step?session_id=`** so **`LATEST_STATE`** lines up.

### 3.2 Client → server: binary frame format

The server accepts **raw WebP/JPEG bytes** or an **optional leading format byte**:

| First byte | Meaning |
|------------|---------|
| `0x01` | Remaining bytes are **JPEG**. |
| `0x02` | Remaining bytes are **WebP**. |
| (absent) | Auto-detect: RIFF+`WEBP` → WebP; `FF D8` → JPEG; else OpenCV `imdecode`. |

The Next.js client sends **`[fmt_u8][image_bytes]`** ([`CameraOverlay.tsx`](frontend-remop/app/components/CameraOverlay.tsx): `FMT_JPEG = 1`, `FMT_WEBP = 2`).

Decode implementation: **`decode_image_bytes`** in [`inference_server.py`](backend/inference_server.py) (TurboJPEG for JPEG when available; Pillow for WebP).

### 3.3 Inference concurrency

- A process-wide **`asyncio.Lock()`** (`_infer_lock`) ensures **one GPU pipeline `infer()` at a time** across all WebSocket clients (avoids GPU contention).
- Decode and CPU postprocess run in the **`VISION`** thread pool.

### 3.4 Per-frame server pipeline (order matters)

1. Receive bytes → decode to **BGR `numpy`**.
2. Under `_infer_lock`: **`VisionPipeline.infer(frame)`** → wire payload `out` (see §4).
3. **Without** holding `_infer_lock`: **`postprocess_frame_for_agent(frame, out)`** → `(webp_bytes, grounded)`.
4. **`await publish_latest(session_id, webp_bytes, out, grounded)`** — updates **`LATEST_STATE`** *before* replying on the socket (avoids client races where agent POST returns 409).
5. **`await ws.send_json(out)`** — same structure as §4.

---

## 4. Wire JSON: inference response (`out`)

Produced by **`_detections_payload`** in [`vision_pipeline.py`](backend/vision_pipeline.py). Sent to the browser as the WebSocket **text JSON** message.

```ts
// Conceptual TypeScript shape (matches frontend InferResponse)
type InferResponse = {
  w: number;
  h: number;
  detections: Detection[];
  error?: string; // only on decode/pipeline failure paths
};

type Detection = {
  label: string;      // class name from YOLO
  conf: number;       // 0..1
  x1: number; y1: number; x2: number; y2: number;  // normalized 0..1
  cx: number; cy: number;                          // box center, normalized
  rel_depth: number;  // MiDaS scalar at integer pixel (u,v) of center
};
```

**Depth semantics:** MiDaS outputs are used **as a relative ordering signal** (see YOLO webcam docstring: larger values ≈ closer in this setup). They are **not** metric depth. The agent prompt describes **`cz`** as a rough step-equivalent after scaling (§5).

On failure, the server may send **`{ error, w: 0, h: 0, detections: [] }`**.

---

## 5. Grounded objects (LLM-facing)

Built by **`ground_detections_payload`** in [`grounding.py`](backend/grounding.py) from the same **`out`** dict the client receives (not from WebP).

Each element:

```json
{
  "class": "cup",
  "cx": 0.512,
  "cy": 0.403,
  "cz": 1.234,
  "sector": "left" | "center" | "right",
  "v_band": "upper" | "middle" | "lower",
  "conf": 0.87
}
```

- **`cz`** = **`rel_depth / K`** where **`K`** is **`AGENT_DEPTH_SCALE_K`** (default `1.0`). There is **no** per-frame min/max normalization — calibration is **your responsibility**.
- **`sector` / `v_band`** use configurable thresholds: **`AGENT_SECTOR_*`** env vars (defaults split at ~1/3 and ~2/3 of normalized x/y).
- **Filtering:** drop detections with **`conf < AGENT_DETECTION_MIN_CONF`** (default `0.3`), sort by conf descending, keep up to **`AGENT_TOP_N_DETECTIONS`** (default `20`).
- **Optional allowlist:** **`AGENT_CLASS_ALLOWLIST`** — comma-separated **lowercase** names; if set, only those classes pass.

### 5.1 Attention / “blinders” (`task_anchor` non-empty)

**`prioritize_grounded_for_model`** in [`agent_attention.py`](backend/agent_attention.py):

- If **`task_anchor`** is empty → returns a **copy** of full grounded list (no reorder).
- Else:
  1. **`person`**, **`cat`**, **`dog`** always appear **first** (by conf), regardless of anchor.
  2. Up to **`AGENT_ANCHOR_TOP_MATCH`** detections whose class **matches anchor tokens** (substring/token overlap on class string).
  3. Up to **`AGENT_ANCHOR_TOP_REST`** other detections (by conf), with a twist: if **`inferred_held_object`** is set, **prefer non-held classes first** in the “rest” slots so surfaces (`table`, `bin`, …) surface before the held blob.

Deduping preserves order on `(class, cx, cy, cz)`.

---

## 6. Session state: `world_state.py`

All session data lives in **`_sessions: dict[str, _SessionBrainState]`** under a single **`asyncio.Lock()`** (`_lock`). There is **no** disk persistence of session brain state (except Gemini logs).

### 6.1 `LatestSnapshot` (published by vision, read by agent)

| Field | Type | Meaning |
|-------|------|---------|
| `frame_webp` | `bytes` | Resized **WebP** of the **full-resolution** frame (for Gemini multimodal input). |
| `detections` | `dict` | **Copy** of the wire payload (`w`, `h`, `detections[]`) at publish time. |
| `grounded` | `list[dict]` | Grounded objects list (§5) at publish time. |
| `version` | `int` | Monotonic **per session** counter incremented on each successful publish. |

### 6.2 `_SessionBrainState` (memory + coordination)

| Field | Purpose |
|-------|---------|
| `latest` | Current **`LatestSnapshot`** or `None` until first successful infer. |
| `recent_action_labels` | Rolling labels (formatted `name` or `name({...json})`), last **5** by default. |
| `last_agent_ms` | Wall for **rate limit** after successful agent completion. |
| `agent_busy` | **True** while a Gemini call is in flight for this session. |
| `task_anchor` | Session subgoal string; merged from model each successful step (§7). |
| `turn_log` | List of dicts: `thought`, `instruction`, `actions` (labels), `anchor` (post-merge). Trimmed to **`AGENT_MEMORY_TURNS`** newest entries. |
| `last_step_monotonic` | Used to compute **`seconds_since_last_step`** for the prompt. |
| `last_issued_action_labels` | Exact labels from the **previous** successful step. |
| `voice_last_speak_text` | Last **committed** TTS line (for echo / dwell dedupe). |
| `voice_last_speak_monotonic` | When that line was committed. |
| `voice_last_motor_fingerprint` | Canonical fingerprint of **non-wait** actions from last **motor** commit (§9). |
| `inferred_held_object` | Server-side inference from **pick_up / place / drop** (§6.3). |

### 6.3 Inferred “held” object

**`apply_inferred_held_after_actions`** walks **this tick’s** `actions` in order:

- **`place`** or **`drop`** → clears held.
- **`pick_up`** → if `args.target` is a non-empty string, sets held to that string (truncated to **`AGENT_HELD_OBJECT_MAX_LEN`**).

This is **not** vision-based truth; the prompt tells the model to **verify visually**.

### 6.4 `task_anchor` merge rules

**`merge_task_anchor_from_model`** (`world_state.py`):

- Model output **`""`** → **keep** current anchor.
- **`CLEAR`** (case-insensitive; outer quotes stripped) → **clear** anchor.
- Otherwise → **sanitize** (alphanumerics + ASCII spaces only, collapse whitespace); if empty after sanitize, **keep** current; else replace, truncated to **`AGENT_TASK_ANCHOR_MAX_LEN`**.

### 6.5 Turn log excerpt for Gemini

**`get_memory_for_prompt`** builds:

- **`turn_log_excerpt`**: last **`AGENT_MEMORY_TURNS`** turns, each as one **JSON object per line** (`thought`, `instruction`, `actions`, `anchor_after`), total chars capped by **`AGENT_MEMORY_MAX_CHARS`** (tail-truncated if needed).
- **`seconds_since_last_step`**: `time.monotonic() - last_step_monotonic` if a prior step exists.

### 6.6 Agent admission control

**`try_begin_agent_work`:**

- Rejects if **`agent_busy`** → HTTP **429** reason `busy`.
- Rejects if **`now_ms - last_agent_ms < AGENT_MIN_INTERVAL_MS`** → **429** reason `throttle`.
- Else sets **`agent_busy = True`**.

**`finish_agent_work`:** clears busy; on **success** updates **`last_agent_ms`**.

---

## 7. Agent: Gemini

### 7.1 Entry point

**`POST /v1/agent/step`** ([`inference_server.py`](backend/inference_server.py)):

1. **`copy_snapshot(session_id)`** — if `None` → **409** (“No perception state… send frames first”).
2. **`try_begin_agent_work`** — may **429**.
3. Missing **`GEMINI_API_KEY`** → **503**, busy cleared.
4. **`get_memory_for_prompt`**, **`prioritize_grounded_for_model`** on **`snap.grounded`**.
5. **`run_agent(snap.frame_webp, grounded_for_model, …)`** in **`AGENT`** thread pool ([`executors.py`](backend/executors.py)).
6. On success: **`update_memory_after_agent_success`**, **`compute_voice_gate`**, **`commit_voice_gate_result`**, **`finish_agent_work(success=True)`**.

Query params: **`goal`**, **`last_outcome`** (optional strings, forwarded into the user message).

Body (JSON, optional): **`{ "is_tts_playing": boolean }`** — forwarded to **`compute_voice_gate`**.

### 7.2 Model configuration

| Env | Default | Role |
|-----|---------|------|
| `GEMINI_API_KEY` | — | Required for agent. |
| `GEMINI_AGENT_MODEL` | `gemini-2.5-flash-lite` | Model id. |
| `AGENT_MAX_OUTPUT_TOKENS` | `256` | Clamped 32..512. |
| `HIGH_LEVEL_INTENTION_PROMPT` | (built-in tidying text) | Overrides mission paragraph. |

The **system instruction** is assembled in **`_system_instruction()`** in [`agent_gemini.py`](backend/agent_gemini.py): mission, role, **anchor/holding rules**, **`ACTION_REGISTRY_PROMPT`**, and JSON-only output requirement.

### 7.3 User message content (text side)

Includes, in order (when applicable):

- **`Grounded objects (JSON array):`** — JSON dump of **attention-filtered** grounded list.
- **`Stored task_anchor`**
- **`Session tool-inference`** — inferred held object line.
- **`Recent turn log`** — excerpt.
- **`Recent action labels`** + **`Last step action labels`**
- **Motion cooldown hint** — if within **`AGENT_MOTION_COOLDOWN_HINT_SEC`** of last step **and** last labels included a **motion** action (`move_*`, `turn_*`), nudge toward **`wait`** / not repeating motion without evidence.
- **`Current goal`**, **`Last outcome / human feedback`**
- Closing instruction to respond with JSON.

Image side: **`types.Part.from_bytes(..., mime_type="image/webp")`**.

### 7.4 Structured output schema (`AgentResponse`)

Defined in [`agent_tools.py`](backend/agent_tools.py) (Pydantic, used as **`response_schema`**):

| Field | Type | Notes |
|-------|------|-------|
| `thought` | `str` | Dashboard / operator only; **not** read aloud by design. |
| `instruction` | `str` | Preferred TTS line when non-empty; constrained in prompt (second-person imperatives; banned words). |
| `actions` | `list[Action]` | Each action: **`name`** + **`args_json`** string containing a **JSON object**. |
| `task_anchor` | `str` | Tri-state: `""` keep, `CLEAR` clear, else new anchor (§6.4). |

**Legacy:** if the model returns **`say`** instead of **`thought`**, a validator maps **`say` → thought**.

**`ActionName`**: `move_forward`, `move_backward`, `turn_left`, `turn_right`, `pick_up`, `drop`, `place`, `look_around`, `wait`.

**`args_json` examples** (from prompt/registry):

- `move_forward` / `move_backward`: `{"steps": 1}`
- `turn_left` / `turn_right`: `{"degrees": 30}`
- `pick_up`: `{"target": "<class string>"}` — must match grounded **`class`**
- `place`: `{"target": "...", "near": "<class string>"}` — **`near`** must be grounded class
- `drop`: `{}`
- `look_around`: `{}`
- `wait`: `{}` or `{"seconds": 2}`

### 7.5 API response: `POST /v1/agent/step`

Success JSON ([`inference_server.py`](backend/inference_server.py)):

| Key | Meaning |
|-----|---------|
| `say` | Dashboard line: **`thought`** if non-empty else **spoken_line** fallback. |
| `instruction` | Raw model **`instruction`**. |
| `spoken_line` | **`instruction`** if set, else **`tts_line_from_actions`**. |
| `actions` | `[{ "name", "args" }]` via **`action_to_api_dict`** (parsed `args_json`). |
| `state_version` | **`snap.version`** at time of read (perception snapshot used). |
| `task_anchor` | Stored anchor after merge. |
| `inferred_held_object` | After applying this tick’s actions. |
| `voice` | `{ speak, should_speak, phase, supersede }` (§9). |

HTTP errors:

- **409** — no `LATEST_STATE` for session.
- **429** — busy or throttle.
- **503** — no API key.
- **502** — Gemini failure (`Gemini error: …`).

### 7.6 Session logging (`backend/logs/`)

**`append_gemini_log(session_id, entry)`** writes **JSON Lines** (`.jsonl`). Filename pattern: **`gemini_{sanitized_session}_{utc_timestamp}.jsonl`**.

Each line may include: UTC timestamp, model id, full **`user_text`**, **`frame_webp_bytes`** length, grounded snapshot, memory fields, **`gemini_response`** metadata, parsed **`agent_response`**, or **`error`**.

---

## 8. Voice gate (`voice_gate.py`)

**`compute_voice_gate`** decides **`should_speak`**, **`supersede`**, and **`phase`** using **anchor transitions**, **motor fingerprint**, **dwell**, **min interval**, and **`is_tts_playing`**.

### 8.1 Motor fingerprint

**`motor_fingerprint_for_actions`**: for every action **except** `wait`, canonicalize `args_json` (sorted keys JSON) and build sorted strings `name:{...}` joined by `|`. Empty if only waits.

### 8.2 Phase (`voice.phase`)

| Condition | Phase |
|-----------|--------|
| No anchor after merge | `idle` |
| Non-empty fingerprint | `motion_pending` |
| All actions are `wait` | `waiting_scene` |
| Else | `anchored` |

### 8.3 When `should_speak` becomes true

1. **Anchor changed** (`anchor_before != anchor_after`) → **`should_speak`**, **`supersede = True`**, commit reason **`anchor`**.
2. Else if fingerprint **non-empty** and **≠** `voice_last_motor_fingerprint` → **`should_speak`**, **`supersede = False`**, commit **`motor`**.
3. Else if **dwell**: same anchor, **`voice_last_speak_monotonic > 0`**, and elapsed ≥ **`VOICE_MIN_DWELL_SEC`** → **`should_speak`** (unless normalized TTS line equals last spoken line — then suppressed).
4. **`is_tts_playing`** and **`should_speak`** and **not** **`supersede`** → **force `should_speak = False`** (urgent barge-in still allowed via supersede).

### 8.4 Min interval

**`VOICE_MIN_INTERVAL_SEC`** (default `2`): if **`should_speak`**, not supersede, and last speak was too recent → suppress.

### 8.5 `speak` field

The **`speak`** string in the payload is the **line intended for TTS** when gating allows; when not speaking, the gate may echo **`voice_last_speak_text`** for UI stability.

### 8.6 Persistence on commit

**`commit_voice_gate_result`** (in `world_state.py`) runs only when **`should_speak`**; updates **`voice_last_speak_text`**, **`voice_last_speak_monotonic`**, and on **`commit_reason == "motor"`** stores **`voice_last_motor_fingerprint`**.

---

## 9. Frontend (Next.js)

### 9.1 Dynamic import

**[`CameraOverlayMount.tsx`](frontend-remop/app/components/CameraOverlayMount.tsx)** loads **`CameraOverlay`** with **`next/dynamic` { ssr: false }** so browser-only APIs stay client-only.

### 9.2 Session id

On mount, **`crypto.randomUUID()`** (fallback `sid-{Date.now()}`) — same id appended to **`/ws/infer`** and **`/v1/agent/step`**.

### 9.3 WebSocket URL

**`NEXT_PUBLIC_INFERENCE_WS_URL`** (default `ws://127.0.0.1:8000/ws/infer`). Query params added at connect: **`model`**, **`session_id`**.

### 9.4 Capture loop

**`requestAnimationFrame`** drives **`tick()`** when streaming:

- Skips when **`document.visibilityState === 'hidden'`**.
- **Ping-pong** with **`busyRef`**: one frame in flight at a time.
- Canvas **`toBlob`** **WebP** at quality `0.5` if supported, else **JPEG** `0.72`.
- Prepends **`FMT_WEBP` / `FMT_JPEG`** byte.

**`inferReady`**: set **`true`** after first JSON message with **`w > 0`** (used to avoid agent POST before state exists).

### 9.5 Agent polling

When **`streaming && sessionId && inferReady`**:

- HTTP base: **`NEXT_PUBLIC_AGENT_HTTP_URL`** or derived from WebSocket URL (**`ws`→`http`**, **`wss`→`https`**).
- **`POST .../v1/agent/step?session_id=`** every **`750ms`** (`gapMs`), with at most one request in flight (`agentStepInFlightRef`).
- Body: **`{ is_tts_playing }`** from **`subscribeTtsPlaying`** / **`getIsTtsPlaying()`** in [`agentTts.ts`](frontend-remop/app/lib/agentTts.ts).
- **429** is ignored silently (throttle).
- TTS: if **`voice.should_speak`**, uses **`voice.speak`** with **`speakInstruction(line, { supersede })`**; non-supersede path delays slightly (**`CLIENT_TTS_MIN_GAP_MS = 250`**) to stagger from server **`VOICE_MIN_INTERVAL_SEC`**.

### 9.6 Audio unlock

**`unlockAudioFromUserGesture()`** and **`resumeAudioContextAfterUserGesture()`** run on **Start camera** (same user gesture) for Safari/iOS: silent Web Speech utterance + **`AudioContext.resume()`**.

### 9.7 TTS engines

| Engine | Behavior |
|--------|----------|
| **`browser`** (default) | **`speechSynthesis`** queue; **supersede** cancels queue and clears defer timer. |
| **`openai`** | **`POST /api/tts`** → OpenAI **`tts-1`**, **`response_format: mp3`**; playback via **HTMLAudioElement** first, **Web Audio** fallback. |

**`NEXT_PUBLIC_TTS_ENGINE`**, **`NEXT_PUBLIC_OPENAI_TTS_VOICE`**, server **`OPENAI_API_KEY`** for `/api/tts`. Optional **`NEXT_PUBLIC_TTS_DEBUG`**.

---

## 10. Environment variables (reference)

### 10.1 Backend (`backend/.env`)

See **[`backend/.env.example`](backend/.env.example)** for the canonical list. Additional / noteworthy:

| Variable | Default | Module |
|----------|---------|--------|
| `VISION_EXECUTOR_WORKERS` | `8` | `executors.py` |
| `AGENT_EXECUTOR_WORKERS` | `4` | `executors.py` |
| `VOICE_MIN_INTERVAL_SEC` | `2` | `voice_gate.py` (not in `.env.example`) |
| `YOLO_MODEL` | — | Legacy alias for oiv7 path when `YOLO_MODEL_OIV7` unset |
| `MIDAS_EVERY_N` | `1` | Reuse previous depth map every N frames |

### 10.2 Frontend (`frontend-remop/.env.local`)

See **[`frontend-remop/.env.example`](frontend-remop/.env.example)**.

---

## 11. Quick start (development)

You need **two processes**: Python inference server + Next.js dev server.

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # set GEMINI_API_KEY for agent; optional CORS, tuning
python -m uvicorn inference_server:app --host 0.0.0.0 --port 8000
```

Shortcut: **`./serve.sh`** (uses venv python if present).

**Health:** `http://127.0.0.1:8000/health`

**Troubleshooting `ModuleNotFoundError: No module named 'torch'`:** use **`python -m uvicorn`** inside the activated venv, not a global `uvicorn` binary.

**PyTurboJPEG** (optional JPEG speedup): requires **libjpeg-turbo** on the system; otherwise OpenCV decode is used.

### Frontend

```bash
cd frontend-remop
cp .env.example .env.local   # optional: NEXT_PUBLIC_INFERENCE_WS_URL, OPENAI_API_KEY
npm install
npm run dev
```

Open `http://localhost:3000`. Use **Start camera** (required for Safari / user-gesture rules).

Use **`wss://`** and **HTTPS** in production for camera APIs outside localhost; align **`INFERENCE_CORS_ORIGINS`**.

### Local OpenCV demo (no browser)

```bash
cd backend && source .venv/bin/activate && python yolo26_depth_webcam.py
```

First run downloads YOLO and MiDaS weights; **`timm`** is pinned for MiDaS v3.1 compatibility.

---

## 12. License

This project is licensed under the MIT License—see [LICENSE](LICENSE).
