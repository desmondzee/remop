/**
 * Agent instruction TTS: Web Speech (default) or OpenAI `tts-1` (cheapest) via `/api/tts`.
 * Keeps strong refs to SpeechSynthesisUtterance until onend/onerror (Chrome/Safari GC).
 * Defers speak() after cancel() to avoid engine race. Call unlockAudioFromUserGesture
 * from Start camera (same user gesture) for iOS/Safari and AudioContext playback.
 */

export type TtsEngine = "browser" | "openai";

/** OpenAI speech voices (all work with `tts-1`). */
export const OPENAI_TTS_VOICE_IDS = [
  "alloy",
  "ash",
  "ballad",
  "coral",
  "echo",
  "fable",
  "nova",
  "onyx",
  "sage",
  "shimmer",
  "verse",
] as const;

export type OpenaiTtsVoiceId = (typeof OPENAI_TTS_VOICE_IDS)[number];

const CANCEL_DEFER_MS = 50;

const OPENAI_TTS_VOICES_SET = new Set<string>(OPENAI_TTS_VOICE_IDS);

/** Console: `[remop:tts]` — on in development; set NEXT_PUBLIC_TTS_DEBUG=0 to mute, =1 to force in prod. */
const TTS_DEBUG =
  process.env.NEXT_PUBLIC_TTS_DEBUG !== "0" &&
  (process.env.NEXT_PUBLIC_TTS_DEBUG === "1" ||
    process.env.NODE_ENV === "development");

function ttsLog(...args: unknown[]) {
  if (!TTS_DEBUG) return;
  console.log("[remop:tts]", ...args);
}

/** Always-on logs for OpenAI `/api/tts` pipeline (filter DevTools by `[remop:tts:openai]`). */
const OAI = "[remop:tts:openai]";

function openaiLog(message: string, detail?: Record<string, unknown>) {
  if (detail === undefined) console.log(OAI, message);
  else console.log(OAI, message, detail);
}

function openaiWarn(message: string, detail?: unknown) {
  console.warn(OAI, message, detail);
}

function openaiErr(message: string, detail?: unknown) {
  console.error(OAI, message, detail);
}

function envEngine(): TtsEngine {
  return process.env.NEXT_PUBLIC_TTS_ENGINE === "openai"
    ? "openai"
    : "browser";
}

function envDefaultOpenaiVoice(): OpenaiTtsVoiceId {
  const v = process.env.NEXT_PUBLIC_OPENAI_TTS_VOICE?.trim().toLowerCase() ?? "";
  if (OPENAI_TTS_VOICES_SET.has(v)) return v as OpenaiTtsVoiceId;
  return "alloy";
}

// --- Web Speech: GC-safe utterance retention ---
const activeUtterances: SpeechSynthesisUtterance[] = [];

function removeUtterance(u: SpeechSynthesisUtterance) {
  const i = activeUtterances.indexOf(u);
  if (i >= 0) activeUtterances.splice(i, 1);
}

function clearActiveUtterances() {
  activeUtterances.length = 0;
}

// --- Playing flag (POST is_tts_playing) ---
let isPlayingFlag = false;
const subscribers = new Set<(v: boolean) => void>();
const idleSubscribers = new Set<(atMs: number) => void>();

function emitAgentTtsBecameIdle(atMs: number) {
  idleSubscribers.forEach((cb) => {
    if (typeof cb === "function") {
      try {
        cb(atMs);
      } catch {
        /* ignore */
      }
    }
  });
}

function emitPlaying(v: boolean) {
  if (isPlayingFlag === v) return;
  const wasPlaying = isPlayingFlag;
  isPlayingFlag = v;
  subscribers.forEach((cb) => {
    if (typeof cb === "function") {
      try {
        cb(v);
      } catch {
        /* ignore subscriber failures */
      }
    }
  });
  if (!v && wasPlaying && getTtsQueueDepth() === 0) {
    emitAgentTtsBecameIdle(Date.now());
  }
}

export function getIsTtsPlaying(): boolean {
  return isPlayingFlag;
}

export function subscribeTtsPlaying(cb: (v: boolean) => void): () => void {
  subscribers.add(cb);
  cb(isPlayingFlag);
  return () => subscribers.delete(cb);
}

/** Fires once when playback stops and the instruction queue is empty (both engines). */
export function subscribeAgentTtsBecameIdle(
  cb: (atMs: number) => void
): () => void {
  idleSubscribers.add(cb);
  return () => idleSubscribers.delete(cb);
}

// --- Engine & OpenAI voice ---
let engine: TtsEngine = envEngine();
let openaiVoice: OpenaiTtsVoiceId = envDefaultOpenaiVoice();
let openaiRotateVoices = false;
let openaiRotateIndex = 0;

export type OpenaiTtsStatus =
  | { status: "idle"; detail: string }
  | { status: "loading"; detail: string }
  | { status: "ready"; detail: string }
  | { status: "error"; detail: string };

let openaiStatus: OpenaiTtsStatus["status"] = "idle";
let openaiStatusDetail = "";
const openaiStatusSubscribers = new Set<(s: OpenaiTtsStatus) => void>();

function emitOpenaiStatus() {
  const payload: OpenaiTtsStatus = {
    status: openaiStatus,
    detail: openaiStatusDetail,
  };
  openaiStatusSubscribers.forEach((cb) => {
    if (typeof cb === "function") {
      try {
        cb(payload);
      } catch {
        /* ignore */
      }
    }
  });
}

export function subscribeOpenaiTtsStatus(
  cb: (s: OpenaiTtsStatus) => void
): () => void {
  openaiStatusSubscribers.add(cb);
  cb({ status: openaiStatus, detail: openaiStatusDetail });
  return () => openaiStatusSubscribers.delete(cb);
}

export function setTtsEngine(next: TtsEngine) {
  engine = next;
}

export function getTtsEngine(): TtsEngine {
  return engine;
}

export function setOpenaiTtsVoice(voice: string) {
  const v = voice.trim().toLowerCase();
  if (OPENAI_TTS_VOICES_SET.has(v)) {
    openaiVoice = v as OpenaiTtsVoiceId;
  }
}

export function getOpenaiTtsVoice(): OpenaiTtsVoiceId {
  return openaiVoice;
}

export function setOpenaiTtsRotateVoices(rotate: boolean) {
  openaiRotateVoices = rotate;
}

export function getOpenaiTtsRotateVoices(): boolean {
  return openaiRotateVoices;
}

function pickOpenaiVoiceForLine(): OpenaiTtsVoiceId {
  if (!openaiRotateVoices) return openaiVoice;
  const i = openaiRotateIndex++ % OPENAI_TTS_VOICE_IDS.length;
  return OPENAI_TTS_VOICE_IDS[i];
}

// --- AudioContext (OpenAI MP3 decode + playback) ---
let audioContext: AudioContext | null = null;
let unlockDone = false;

export function getSharedAudioContext(): AudioContext | null {
  return audioContext;
}

/** Some environments expose speechSynthesis without speak() or the Utterance constructor. */
function getBrowserSpeechApi(): {
  synth: SpeechSynthesis;
  Utterance: new (text: string) => SpeechSynthesisUtterance;
} | null {
  if (typeof window === "undefined") return null;
  const w = window as Window & {
    SpeechSynthesisUtterance?: new (text: string) => SpeechSynthesisUtterance;
  };
  const synth = w.speechSynthesis;
  const Utterance = w.SpeechSynthesisUtterance;
  if (
    !synth ||
    typeof synth.speak !== "function" ||
    typeof synth.cancel !== "function" ||
    typeof Utterance !== "function"
  ) {
    return null;
  }
  return { synth, Utterance };
}

// --- Browser queue ---
const browserQueue: string[] = [];
let browserSpeaking = false;
let browserDeferTimer: number | null = null;

function clearBrowserDefer() {
  if (browserDeferTimer !== null) {
    window.clearTimeout(browserDeferTimer);
    browserDeferTimer = null;
  }
}

function browserPumpPlayingState() {
  emitPlaying(browserSpeaking || browserQueue.length > 0 || browserDeferTimer !== null);
}

function speakNextBrowserUtterance() {
  const api = getBrowserSpeechApi();
  if (!api) {
    browserQueue.length = 0;
    browserSpeaking = false;
    clearBrowserDefer();
    emitPlaying(false);
    return;
  }
  if (browserSpeaking || browserQueue.length === 0) {
    browserPumpPlayingState();
    return;
  }
  const text = browserQueue.shift()!;
  browserSpeaking = true;
  browserPumpPlayingState();

  const u = new api.Utterance(text);
  u.lang = "en-US";
  u.rate = 1.05;
  u.pitch = 1;
  activeUtterances.push(u);

  u.onstart = () => {
    emitPlaying(true);
  };

  const finish = () => {
    removeUtterance(u);
    browserSpeaking = false;
    if (browserQueue.length === 0) {
      emitPlaying(false);
    } else {
      speakNextBrowserUtterance();
    }
  };

  u.onend = () => finish();
  u.onerror = () => finish();

  api.synth.speak(u);
}

function scheduleBrowserAfterCancel() {
  clearBrowserDefer();
  browserDeferTimer = window.setTimeout(() => {
    browserDeferTimer = null;
    speakNextBrowserUtterance();
  }, CANCEL_DEFER_MS);
  browserPumpPlayingState();
}

function browserSpeakInstruction(text: string, supersede: boolean) {
  const api = getBrowserSpeechApi();
  if (!api) return;

  if (supersede) {
    clearBrowserDefer();
    api.synth.cancel();
    clearActiveUtterances();
    browserSpeaking = false;
    browserQueue.length = 0;
    browserQueue.push(text);
    scheduleBrowserAfterCancel();
    return;
  }

  browserQueue.push(text);
  if (!browserSpeaking && !browserDeferTimer) {
    speakNextBrowserUtterance();
  } else {
    browserPumpPlayingState();
  }
}

// --- OpenAI queue ---
const openaiQueue: string[] = [];
let openaiGen = 0;
let openaiSerial = Promise.resolve();
let openaiCurrentSource: AudioBufferSourceNode | null = null;

async function ensureAudioContext(): Promise<AudioContext | null> {
  if (typeof window === "undefined") return null;
  if (!audioContext) {
    const Ctx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (!Ctx) {
      openaiErr("AudioContext constructor missing");
      return null;
    }
    audioContext = new Ctx();
    openaiLog("AudioContext created", {
      state: audioContext.state,
      sampleRate: audioContext.sampleRate,
    });
  }
  try {
    await audioContext.resume();
    openaiLog("AudioContext resume ok", { state: audioContext.state });
  } catch (e) {
    openaiWarn("AudioContext.resume() rejected", e);
  }
  return audioContext;
}

async function fetchOpenaiSpeechMp3(
  text: string,
  voice: OpenaiTtsVoiceId
): Promise<ArrayBuffer> {
  const t0 =
    typeof performance !== "undefined" ? performance.now() : 0;
  openaiLog("POST /api/tts (sending)", {
    voice,
    textLength: text.length,
    textPreview: text.slice(0, 120),
  });

  let res: Response;
  try {
    res = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, voice }),
    });
  } catch (e) {
    openaiErr("fetch failed before response (network / CORS / offline?)", e);
    throw e;
  }

  const elapsedMs =
    typeof performance !== "undefined"
      ? Math.round(performance.now() - t0)
      : 0;
  openaiLog("response headers", {
    ok: res.ok,
    status: res.status,
    statusText: res.statusText,
    contentType: res.headers.get("content-type"),
    elapsedMs,
  });

  if (!res.ok) {
    const raw = await res.text();
    let msg = raw.slice(0, 400) || res.statusText;
    try {
      const j = JSON.parse(raw) as { error?: string };
      if (typeof j.error === "string" && j.error) msg = j.error;
    } catch {
      /* use raw slice */
    }
    openaiErr("API error body", { status: res.status, msg });
    throw new Error(msg || `TTS HTTP ${res.status}`);
  }

  const buf = await res.arrayBuffer();
  logMp3PayloadPrefix(buf);
  openaiLog("audio MP3 received", { byteLength: buf.byteLength });
  return buf;
}

/** Detect JSON/HTML mistaken for audio or empty payload. */
function logMp3PayloadPrefix(buf: ArrayBuffer): void {
  const n = Math.min(16, buf.byteLength);
  const u8 = new Uint8Array(buf, 0, n);
  const hex = [...u8].map((b) => b.toString(16).padStart(2, "0")).join(" ");
  const head =
    n >= 3
      ? String.fromCharCode(u8[0], u8[1], u8[2], n >= 4 ? u8[3] : 32)
      : "";
  const frameSync =
    n >= 2 && u8[0] === 0xff && (u8[1] & 0xe0) === 0xe0;
  const id3 = n >= 3 && u8[0] === 0x49 && u8[1] === 0x44 && u8[2] === 0x33;
  const looksLikeMp3 = frameSync || id3;
  openaiLog("payload prefix (expect MP3 frame 0xFFEx or ID3)", {
    hex,
    headAscii: head.trim(),
    frameSync,
    id3Tag: id3,
    looksLikeMp3,
  });
}

/**
 * Native &lt;audio&gt; MP3 decode/output — usually more reliable than decodeAudioData for MPEG.
 * Must wait for load events before play(); immediate play() after setting src often stays silent.
 */
async function playMp3ViaAudioElement(arrayBuffer: ArrayBuffer): Promise<void> {
  const blob = new Blob([arrayBuffer], { type: "audio/mpeg" });
  const url = URL.createObjectURL(blob);
  const el = new Audio();
  el.volume = 1;
  el.muted = false;
  el.preload = "auto";
  el.setAttribute("playsinline", "true");

  const revoke = () => {
    try {
      URL.revokeObjectURL(url);
    } catch {
      /* ignore */
    }
  };

  try {
    await new Promise<void>((resolve, reject) => {
      const t = window.setTimeout(() => {
        reject(new Error("HTMLAudio load timeout (15s)"));
      }, 15_000);
      const done = () => window.clearTimeout(t);

      const onReady = () => {
        openaiLog("HTMLAudio can play", {
          duration: el.duration,
          readyState: el.readyState,
          networkState: el.networkState,
          paused: el.paused,
        });
      };

      el.addEventListener(
        "loadedmetadata",
        () => {
          onReady();
        },
        { once: true }
      );

      el.addEventListener(
        "canplay",
        () => {
          done();
          resolve();
        },
        { once: true }
      );

      el.addEventListener(
        "error",
        () => {
          done();
          const code = el.error?.code;
          const msg = el.error?.message;
          reject(
            new Error(
              `HTMLAudio load error${code != null ? ` code=${code}` : ""}${msg ? `: ${msg}` : ""}`
            )
          );
        },
        { once: true }
      );

      el.src = url;
      el.load();
    });

    if (!Number.isFinite(el.duration) || el.duration <= 0) {
      openaiWarn("HTMLAudio duration suspicious after load", {
        duration: el.duration,
      });
    }

    try {
      await el.play();
      openaiLog("HTMLAudio play() resolved", {
        currentTime: el.currentTime,
        paused: el.paused,
      });
    } catch (e) {
      openaiWarn("HTMLAudio play() rejected — retrying in DOM (some WebKit builds)", e);
      el.style.display = "none";
      el.setAttribute("aria-hidden", "true");
      document.body.appendChild(el);
      try {
        await el.play();
        openaiLog("HTMLAudio play() ok after appendChild", {
          paused: el.paused,
        });
      } catch (e2) {
        document.body.removeChild(el);
        throw e2;
      }
    }

    await new Promise<void>((resolve, reject) => {
      el.onended = () => {
        openaiLog("HTMLAudio onended");
        if (el.parentNode) el.parentNode.removeChild(el);
        revoke();
        resolve();
      };
      el.onerror = () => {
        if (el.parentNode) el.parentNode.removeChild(el);
        revoke();
        reject(new Error("HTMLAudioElement error during playback"));
      };
    });
  } catch (e) {
    revoke();
    throw e;
  }
}

async function playMp3BufferWebAudio(arrayBuffer: ArrayBuffer): Promise<void> {
  openaiLog("playMp3BufferWebAudio: start", {
    inputBytes: arrayBuffer.byteLength,
  });

  const ctx = await ensureAudioContext();
  if (!ctx) {
    openaiErr("playMp3BufferWebAudio: no AudioContext after ensure");
    return;
  }

  let audio: AudioBuffer;
  try {
    const copy = arrayBuffer.slice(0);
    openaiLog("decodeAudioData: calling…", { ctxState: ctx.state });
    audio = await ctx.decodeAudioData(copy);
    openaiLog("decodeAudioData: ok", {
      durationSec: audio.duration,
      channels: audio.numberOfChannels,
      sampleRate: audio.sampleRate,
      length: audio.length,
    });
  } catch (e) {
    openaiErr("decodeAudioData failed (bad MP3 or unsupported codec?)", e);
    ttsLog("decodeAudioData failed", e);
    return;
  }

  if (
    !Number.isFinite(audio.duration) ||
    audio.duration <= 0 ||
    audio.length === 0
  ) {
    openaiErr("decodeAudioData produced empty / invalid buffer", {
      duration: audio.duration,
      length: audio.length,
    });
    return;
  }

  try {
    await ctx.resume();
  } catch (e) {
    openaiWarn("playMp3BufferWebAudio: resume before start rejected", e);
  }
  openaiLog("playMp3BufferWebAudio: pre-start", {
    ctxState: ctx.state,
    currentTime: ctx.currentTime,
  });

  if (ctx.state !== "running") {
    openaiErr("Web Audio: context not running after resume", {
      state: ctx.state,
    });
    return;
  }

  const src = ctx.createBufferSource();
  src.buffer = audio;
  const gain = ctx.createGain();
  gain.gain.value = 1;
  src.connect(gain);
  gain.connect(ctx.destination);
  openaiCurrentSource = src;

  const startAt = ctx.currentTime;
  return new Promise((resolve) => {
    src.onended = () => {
      openaiLog("BufferSource onended (playback finished)");
      if (openaiCurrentSource === src) openaiCurrentSource = null;
      resolve();
    };
    try {
      src.start(startAt);
      openaiLog("BufferSource.start ok", {
        startAt,
        ctxState: ctx.state,
      });
    } catch (e) {
      openaiErr("BufferSource.start failed", e);
      if (openaiCurrentSource === src) openaiCurrentSource = null;
      resolve();
    }
  });
}

/** OpenAI returns MP3: prefer native &lt;audio&gt; (correct load + play order), then Web Audio. */
async function playMp3Buffer(arrayBuffer: ArrayBuffer): Promise<void> {
  openaiLog("playMp3Buffer: strategy = HTMLAudio first, then Web Audio", {
    bytes: arrayBuffer.byteLength,
  });
  try {
    await playMp3ViaAudioElement(arrayBuffer);
    openaiLog("playMp3Buffer: HTMLAudio path completed");
    return;
  } catch (e) {
    openaiWarn("playMp3Buffer: HTMLAudio failed, Web Audio fallback", e);
  }
  await playMp3BufferWebAudio(arrayBuffer);
}

async function drainOpenaiQueue(): Promise<void> {
  while (openaiQueue.length > 0) {
    const myGen = openaiGen;
    const line = openaiQueue[0];
    const voice = pickOpenaiVoiceForLine();

    openaiLog("drain: processing queue head", {
      gen: myGen,
      queueLen: openaiQueue.length,
      voice,
      lineLength: line.length,
    });

    openaiStatus = "loading";
    openaiStatusDetail = `Requesting (${voice})…`;
    emitOpenaiStatus();

    let buf: ArrayBuffer;
    try {
      buf = await fetchOpenaiSpeechMp3(line, voice);
    } catch (e) {
      openaiStatus = "error";
      openaiStatusDetail =
        e instanceof Error ? e.message : String(e);
      emitOpenaiStatus();
      openaiErr("drain: fetchOpenaiSpeechMp3 threw", e);
      ttsLog("openai TTS fetch failed", e);
      openaiQueue.shift();
      continue;
    }

    if (myGen !== openaiGen) {
      openaiLog("drain: skipped after fetch (superseded)", {
        myGen,
        openaiGen,
      });
      continue;
    }

    openaiStatus = "ready";
    openaiStatusDetail = `OpenAI tts-1 (${voice})`;
    emitOpenaiStatus();

    openaiQueue.shift();
    emitPlaying(true);
    openaiLog("drain: playing…");
    await playMp3Buffer(buf);
    if (myGen !== openaiGen) {
      openaiLog("drain: skipped after play (superseded)");
      continue;
    }
  }
  openaiLog("drain: queue empty, emitPlaying(false)");
  emitPlaying(false);
}

function scheduleOpenaiDrain() {
  openaiLog("scheduleOpenaiDrain: chaining drain");
  openaiSerial = openaiSerial.then(() => drainOpenaiQueue()).catch((e) => {
    openaiErr("drainOpenaiQueue unhandled rejection", e);
    ttsLog("openai drain error", e);
    openaiQueue.length = 0;
    openaiStatus = "error";
    openaiStatusDetail = e instanceof Error ? e.message : String(e);
    emitOpenaiStatus();
    emitPlaying(false);
  });
}

function openaiSpeakInstruction(text: string, supersede: boolean) {
  if (supersede) {
    openaiGen++;
    openaiLog("speak: supersede", { gen: openaiGen, textLength: text.length });
    try {
      openaiCurrentSource?.stop();
    } catch {
      /* already stopped */
    }
    openaiCurrentSource = null;
    openaiQueue.length = 0;
    openaiQueue.push(text);
  } else {
    openaiLog("speak: enqueue", { textLength: text.length });
    openaiQueue.push(text);
  }
  openaiLog("speak: queue state", {
    queueLen: openaiQueue.length,
    rotate: openaiRotateVoices,
    fixedVoice: openaiVoice,
  });
  scheduleOpenaiDrain();
}

/** Call from the Start camera click handler (after permission) so iOS/Safari allow audio. */
export function unlockAudioFromUserGesture(): void {
  if (typeof window === "undefined") return;

  if (!unlockDone) {
    unlockDone = true;
    try {
      const api = getBrowserSpeechApi();
      if (api) {
        const u = new api.Utterance(" ");
        u.volume = 0;
        activeUtterances.push(u);
        u.onend = u.onerror = () => {
          removeUtterance(u);
        };
        api.synth.speak(u);
      }
    } catch {
      /* ignore */
    }
  }

  if (!audioContext) {
    const Ctx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (Ctx) {
      audioContext = new Ctx();
    }
  }
  const ctx = audioContext;
  if (ctx) {
    void ctx.resume().catch(() => {});
    /* Same user-gesture tick: start a silent buffer so the destination graph stays eligible
       for playback after async work (e.g. OpenAI fetch ~1s). Without this, many browsers
       leave the context suspended or block BufferSource after the gesture expires. */
    try {
      const n = Math.max(128, Math.ceil(ctx.sampleRate * 0.05));
      const silent = ctx.createBuffer(1, n, ctx.sampleRate);
      const priming = ctx.createBufferSource();
      priming.buffer = silent;
      priming.connect(ctx.destination);
      priming.start(ctx.currentTime);
      ttsLog("unlockAudio: Web Audio primed (silent)", {
        samples: n,
        state: ctx.state,
      });
    } catch (e) {
      ttsLog("unlockAudio: Web Audio prime failed", e);
    }
  }
}

/**
 * Call from a click/tap handler with `await` **before** OpenAI TTS runs its fetch (~1s).
 * Pairs sync Web Audio priming (`unlockAudioFromUserGesture`) with an awaited
 * `AudioContext.resume()` so autoplay policies still see user activation.
 */
export async function resumeAudioContextAfterUserGesture(): Promise<void> {
  if (typeof window === "undefined") return;
  unlockAudioFromUserGesture();
  const ctx = await ensureAudioContext();
  if (!ctx) return;
  try {
    await ctx.resume();
  } catch (e) {
    openaiWarn("resumeAudioContextAfterUserGesture: resume rejected", e);
  }
  openaiLog("resumeAudioContextAfterUserGesture done", { state: ctx.state });
}

export function speakInstruction(
  text: string,
  opts: { supersede: boolean }
): void {
  const t = text.trim();
  if (!t) return;
  if (engine === "openai") {
    openaiLog("speakInstruction → openai", {
      supersede: opts.supersede,
      charCount: t.length,
    });
    openaiSpeakInstruction(t, opts.supersede);
  } else {
    browserSpeakInstruction(t, opts.supersede);
  }
}

/** Approximate queued / in-flight instruction count for debug UI. */
export function getTtsQueueDepth(): number {
  if (engine === "openai") {
    return openaiQueue.length + (openaiCurrentSource ? 1 : 0);
  }
  return browserQueue.length + (browserSpeaking ? 1 : 0);
}

/** Stop camera / teardown: cancel speech and clear queues so flags do not stick. */
export function resetAgentTts(): void {
  const api = getBrowserSpeechApi();
  if (api) {
    clearBrowserDefer();
    api.synth.cancel();
  }
  clearActiveUtterances();
  browserQueue.length = 0;
  browserSpeaking = false;

  openaiGen++;
  try {
    openaiCurrentSource?.stop();
  } catch {
    /* */
  }
  openaiCurrentSource = null;
  openaiQueue.length = 0;

  openaiStatus = "idle";
  openaiStatusDetail = "";
  emitOpenaiStatus();

  emitPlaying(false);
}
