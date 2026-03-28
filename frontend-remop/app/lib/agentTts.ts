/**
 * Agent instruction TTS: Web Speech (default) or optional Kokoro (client-side).
 * Keeps strong refs to SpeechSynthesisUtterance until onend/onerror (Chrome/Safari GC).
 * Defers speak() after cancel() to avoid engine race. Call unlockAudioFromUserGesture
 * from Start camera (same user gesture) for iOS/Safari.
 */

export type TtsEngine = "browser" | "kokoro";

const CANCEL_DEFER_MS = 50;

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
  for (const cb of subscribers) cb(v);
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
let kokoroModel: import("kokoro-js").KokoroTTS | null = null;
let kokoroLoadPromise: Promise<import("kokoro-js").KokoroTTS> | null = null;
let kokoroStatus: "idle" | "loading" | "ready" | "error" = "idle";
let kokoroStatusDetail = "";
const kokoroStatusSubscribers = new Set<
  (s: { status: typeof kokoroStatus; detail: string }) => void
>();

function emitKokoroStatus() {
  const payload = { status: kokoroStatus, detail: kokoroStatusDetail };
  for (const cb of kokoroStatusSubscribers) cb(payload);
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
  if (typeof window === "undefined" || !window.speechSynthesis) return;
  if (browserSpeaking || browserQueue.length === 0) {
    browserPumpPlayingState();
    return;
  }
  const text = browserQueue.shift()!;
  browserSpeaking = true;
  browserPumpPlayingState();

  const u = new SpeechSynthesisUtterance(text);
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

  window.speechSynthesis.speak(u);
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
  if (typeof window === "undefined" || !window.speechSynthesis) return;

  if (supersede) {
    clearBrowserDefer();
    window.speechSynthesis.cancel();
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

async function ensureKokoro(): Promise<import("kokoro-js").KokoroTTS> {
  if (kokoroModel) return kokoroModel;
  if (kokoroLoadPromise) return kokoroLoadPromise;

  kokoroStatus = "loading";
  kokoroStatusDetail = "";
  emitKokoroStatus();

  kokoroLoadPromise = (async () => {
    const { KokoroTTS } = await import("kokoro-js");
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
    const model = await KokoroTTS.from_pretrained(KOKORO_MODEL_ID, {
      dtype: "q8",
      device,
    });
    kokoroModel = model;
    kokoroStatus = "ready";
    kokoroStatusDetail = device === "webgpu" ? "WebGPU" : "WASM";
    emitKokoroStatus();
    return model;
  })().catch((e) => {
    kokoroLoadPromise = null;
    kokoroStatus = "error";
    kokoroStatusDetail = e instanceof Error ? e.message : String(e);
    emitKokoroStatus();
    throw e;
  });

  return kokoroLoadPromise;
}

function playRawAudioToContext(
  raw: { audio: Float32Array; sampling_rate: number }
): Promise<void> {
  const ctx = audioContext;
  if (!ctx) return Promise.resolve();

  const buffer = ctx.createBuffer(1, raw.audio.length, raw.sampling_rate);
  buffer.getChannelData(0).set(raw.audio);

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
      resolve();
    };
    try {
      src.start();
    } catch {
      if (kokoroCurrentSource === src) kokoroCurrentSource = null;
      resolve();
    }
  });
}

async function drainKokoroQueue(): Promise<void> {
  while (kokoroQueue.length > 0) {
    const myGen = kokoroGen;
    const line = kokoroQueue[0];
    let model: import("kokoro-js").KokoroTTS;
    try {
      model = await ensureKokoro();
    } catch {
      kokoroQueue.length = 0;
      emitPlaying(false);
      return;
    }
    if (myGen !== kokoroGen) continue;

    type GenOpts = { voice?: string; speed?: number };
    const raw = await (
      model.generate as (
        text: string,
        opts?: GenOpts
      ) => Promise<{ audio: Float32Array; sampling_rate: number }>
    )(line, { voice: kokoroVoice, speed: 1.05 });
    if (myGen !== kokoroGen) continue;

    kokoroQueue.shift();
    emitPlaying(true);
    await playRawAudioToContext(raw);
    if (myGen !== kokoroGen) continue;
  }
  emitPlaying(false);
}

function scheduleKokoroDrain() {
  kokoroSerial = kokoroSerial.then(() => drainKokoroQueue()).catch(() => {
    kokoroQueue.length = 0;
    emitPlaying(false);
  });
}

function kokoroSpeakInstruction(text: string, supersede: boolean) {
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
      const u = new SpeechSynthesisUtterance(" ");
      u.volume = 0;
      activeUtterances.push(u);
      u.onend = u.onerror = () => {
        removeUtterance(u);
      };
      window.speechSynthesis?.speak(u);
    } catch {
      /* ignore */
    }
  }

  if (!audioContext) {
    const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (Ctx) {
      audioContext = new Ctx();
    }
  }
  void audioContext?.resume();
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
  if (typeof window !== "undefined" && window.speechSynthesis) {
    clearBrowserDefer();
    window.speechSynthesis.cancel();
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
