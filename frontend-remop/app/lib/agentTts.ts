/**
 * Agent instruction TTS: Web Speech (default) or optional Kokoro (client-side).
 * Kokoro follows hexgrad/kokoro.js: Transformers.js + ONNX + phonemizer (eSpeak), with stream
 * polyfills in `kokoro/kokoroBrowserRuntime.ts` so WebKit can iterate gzip streams.
 * Keeps strong refs to SpeechSynthesisUtterance until onend/onerror (Chrome/Safari GC).
 * Defers speak() after cancel() to avoid engine race. Call unlockAudioFromUserGesture
 * from Start camera (same user gesture) for iOS/Safari.
 */

import {
  ensureKokoroStreamPolyfills,
  maskProcessNodeVersionForBrowserImport,
} from "./kokoro/kokoroBrowserRuntime";

export type TtsEngine = "browser" | "kokoro";

const CANCEL_DEFER_MS = 50;

/** Console: `[remop:tts]` — on in development; set NEXT_PUBLIC_TTS_DEBUG=0 to mute, =1 to force in prod. */
const TTS_DEBUG =
  process.env.NEXT_PUBLIC_TTS_DEBUG !== "0" &&
  (process.env.NEXT_PUBLIC_TTS_DEBUG === "1" ||
    process.env.NODE_ENV === "development");

function ttsLog(...args: unknown[]) {
  if (!TTS_DEBUG) return;
  console.log("[remop:tts]", ...args);
}

const KOKORO_MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";

function envEngine(): TtsEngine {
  return process.env.NEXT_PUBLIC_TTS_ENGINE === "kokoro"
    ? "kokoro"
    : "browser";
}

function envKokoroVoice(): string {
  return process.env.NEXT_PUBLIC_KOKORO_VOICE?.trim() || "af_heart";
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

function emitPlaying(v: boolean) {
  if (isPlayingFlag === v) return;
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
}

export function getIsTtsPlaying(): boolean {
  return isPlayingFlag;
}

export function subscribeTtsPlaying(cb: (v: boolean) => void): () => void {
  subscribers.add(cb);
  cb(isPlayingFlag);
  return () => subscribers.delete(cb);
}

// --- Engine & Kokoro ---
let engine: TtsEngine = envEngine();
const kokoroVoice = envKokoroVoice();
type LoadedKokoro = import("./kokoro/kokoroTts").KokoroTTS;
let kokoroModel: LoadedKokoro | null = null;
let kokoroLoadPromise: Promise<LoadedKokoro> | null = null;
let kokoroStatus: "idle" | "loading" | "ready" | "error" = "idle";
let kokoroStatusDetail = "";
const kokoroStatusSubscribers = new Set<
  (s: { status: typeof kokoroStatus; detail: string }) => void
>();

function emitKokoroStatus() {
  const payload = { status: kokoroStatus, detail: kokoroStatusDetail };
  kokoroStatusSubscribers.forEach((cb) => {
    if (typeof cb === "function") {
      try {
        cb(payload);
      } catch {
        /* ignore */
      }
    }
  });
}

export function subscribeKokoroStatus(
  cb: (s: { status: typeof kokoroStatus; detail: string }) => void
): () => void {
  kokoroStatusSubscribers.add(cb);
  cb({ status: kokoroStatus, detail: kokoroStatusDetail });
  return () => kokoroStatusSubscribers.delete(cb);
}

export function setTtsEngine(next: TtsEngine) {
  engine = next;
}

export function getTtsEngine(): TtsEngine {
  return engine;
}

// --- AudioContext (Kokoro playback + resume on user gesture) ---
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

// --- Kokoro queue ---
const kokoroQueue: string[] = [];
let kokoroGen = 0;
let kokoroSerial = Promise.resolve();
let kokoroCurrentSource: AudioBufferSourceNode | null = null;

async function ensureKokoro(): Promise<LoadedKokoro> {
  if (kokoroModel) return kokoroModel;
  if (kokoroLoadPromise) return kokoroLoadPromise;

  kokoroStatus = "loading";
  kokoroStatusDetail = "";
  emitKokoroStatus();

  kokoroLoadPromise = (async () => {
    ttsLog("ensureKokoro: stream polyfills + dynamic import");
    await ensureKokoroStreamPolyfills();
    const unmaskNode = maskProcessNodeVersionForBrowserImport();
    let KokoroTTS: typeof import("./kokoro/kokoroTts").KokoroTTS;
    try {
      ({ KokoroTTS } = await import("./kokoro/kokoroTts"));
    } finally {
      unmaskNode();
    }
    let device: "webgpu" | "wasm" = "wasm";
    if (typeof navigator !== "undefined" && "gpu" in navigator) {
      try {
        const adapter = await (
          navigator as Navigator & {
            gpu?: { requestAdapter: () => Promise<unknown> };
          }
        ).gpu?.requestAdapter?.();
        if (adapter) device = "webgpu";
      } catch {
        device = "wasm";
      }
    }
    ttsLog("ensureKokoro: from_pretrained", KOKORO_MODEL_ID, device);
    const model = await KokoroTTS.from_pretrained(KOKORO_MODEL_ID, {
      dtype: "q8",
      device,
    });
    if (typeof model?.generate !== "function") {
      throw new TypeError("KokoroTTS model has no generate()");
    }
    kokoroModel = model;
    kokoroStatus = "ready";
    kokoroStatusDetail = device === "webgpu" ? "WebGPU" : "WASM";
    emitKokoroStatus();
    ttsLog("ensureKokoro: ready", device);
    return model;
  })().catch((e) => {
    kokoroLoadPromise = null;
    kokoroStatus = "error";
    kokoroStatusDetail = e instanceof Error ? e.message : String(e);
    emitKokoroStatus();
    ttsLog("ensureKokoro: error", e);
    throw e;
  });

  return kokoroLoadPromise;
}

/**
 * Kokoro `generate()` finishes long after the click that unlocked audio; browsers often
 * leave AudioContext "suspended" until resume() runs in the same "activation" as playback.
 * Always await resume() here (and create the context if needed) so output is audible.
 */
async function playRawAudioToContext(
  raw: { audio: Float32Array; sampling_rate: number }
): Promise<void> {
  if (typeof window === "undefined") {
    ttsLog("playRaw: skip (no window)");
    return;
  }

  let ctx = audioContext;
  if (!ctx) {
    const Ctx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (!Ctx) {
      ttsLog("playRaw: no AudioContext constructor");
      return;
    }
    ctx = new Ctx();
    audioContext = ctx;
    ttsLog("playRaw: created AudioContext", "sampleRate", ctx.sampleRate);
  }

  const n = raw.audio?.length ?? 0;
  const rate = raw.sampling_rate;
  if (n === 0 || !Number.isFinite(rate) || rate <= 0) {
    ttsLog("playRaw: skip empty or bad rate", { n, rate, rawKeys: raw && Object.keys(raw) });
    return;
  }

  let resumed = ctx.state;
  try {
    await ctx.resume();
    resumed = ctx.state;
  } catch (e) {
    ttsLog("playRaw: resume() rejected", e, "state", ctx.state);
  }
  ttsLog("playRaw: context state after resume", resumed, "samples", n, "rate", rate);

  const samples = new Float32Array(raw.audio);

  let buffer: AudioBuffer;
  try {
    buffer = ctx.createBuffer(1, n, rate);
  } catch (e) {
    ttsLog("playRaw: createBuffer failed", e);
    return;
  }
  buffer.getChannelData(0).set(samples);

  const src = ctx.createBufferSource();
  src.buffer = buffer;
  const gain = ctx.createGain();
  gain.gain.value = 1;
  src.connect(gain);
  gain.connect(ctx.destination);
  kokoroCurrentSource = src;

  return new Promise((resolve) => {
    src.onended = () => {
      if (kokoroCurrentSource === src) kokoroCurrentSource = null;
      ttsLog("playRaw: onended");
      resolve();
    };
    try {
      src.start();
      ttsLog("playRaw: source.start() ok");
    } catch (e) {
      ttsLog("playRaw: source.start() threw", e);
      if (kokoroCurrentSource === src) kokoroCurrentSource = null;
      resolve();
    }
  });
}

async function drainKokoroQueue(): Promise<void> {
  while (kokoroQueue.length > 0) {
    const myGen = kokoroGen;
    const line = kokoroQueue[0];
    let model: LoadedKokoro;
    try {
      model = await ensureKokoro();
    } catch (e) {
      ttsLog("drain: ensureKokoro failed", e);
      kokoroQueue.length = 0;
      emitPlaying(false);
      return;
    }
    if (myGen !== kokoroGen) continue;

    // Let the browser run pending work (WebSocket send, rAF) before ONNX blocks the main thread.
    await new Promise<void>((r) => requestAnimationFrame(() => r()));

    ttsLog("drain: generate start", { len: line.length, voice: kokoroVoice });
    let raw: Awaited<ReturnType<LoadedKokoro["generate"]>>;
    try {
      raw = await model.generate(line, { voice: kokoroVoice, speed: 1.05 });
    } catch (e) {
      ttsLog("drain: generate failed", e);
      kokoroQueue.shift();
      continue;
    }
    if (myGen !== kokoroGen) continue;

    const alen = raw?.audio?.length ?? 0;
    const sr = raw?.sampling_rate;
    ttsLog("drain: generate done", { samples: alen, sampling_rate: sr });

    kokoroQueue.shift();
    emitPlaying(true);
    await playRawAudioToContext(raw);
    if (myGen !== kokoroGen) continue;
  }
  emitPlaying(false);
}

function scheduleKokoroDrain() {
  kokoroSerial = kokoroSerial.then(() => drainKokoroQueue()).catch((e) => {
    ttsLog("drain: unhandled rejection", e);
    kokoroQueue.length = 0;
    emitPlaying(false);
  });
}

function kokoroSpeakInstruction(text: string, supersede: boolean) {
  ttsLog("kokoroSpeakInstruction", { supersede, len: text.length, hasCtx: !!audioContext });
  if (supersede) {
    kokoroGen++;
    try {
      kokoroCurrentSource?.stop();
    } catch {
      /* already stopped */
    }
    kokoroCurrentSource = null;
    kokoroQueue.length = 0;
    kokoroQueue.push(text);
  } else {
    kokoroQueue.push(text);
  }
  scheduleKokoroDrain();
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
  void audioContext
    ?.resume()
    .then(() => {
      ttsLog("unlock: resume resolved", "state", audioContext?.state);
    })
    .catch((e) => {
      ttsLog("unlock: resume rejected", e, "state", audioContext?.state);
    });
}

/** Warm Kokoro weights after unlock (e.g. right after camera starts). */
export function prefetchKokoro(): void {
  if (typeof window === "undefined") return;
  void ensureKokoro().catch(() => {
    /* status already set */
  });
}

export function speakInstruction(
  text: string,
  opts: { supersede: boolean }
): void {
  const t = text.trim();
  if (!t) return;
  if (engine === "kokoro") {
    kokoroSpeakInstruction(t, opts.supersede);
  } else {
    browserSpeakInstruction(t, opts.supersede);
  }
}

/** Approximate queued / in-flight instruction count for debug UI. */
export function getTtsQueueDepth(): number {
  if (engine === "kokoro") {
    return kokoroQueue.length + (kokoroCurrentSource ? 1 : 0);
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

  kokoroGen++;
  try {
    kokoroCurrentSource?.stop();
  } catch {
    /* */
  }
  kokoroCurrentSource = null;
  kokoroQueue.length = 0;

  emitPlaying(false);
}
