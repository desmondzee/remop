"use client";

import { AppleIntelligenceGlow } from "@xuanhe/apple-intelligence-glow-react";
import { AnimatePresence, motion } from "motion/react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  getIsTtsPlaying,
  getOpenaiTtsRotateVoices,
  getOpenaiTtsVoice,
  getTtsEngine,
  getTtsQueueDepth,
  resetAgentTts,
  resumeAudioContextAfterUserGesture,
  setTtsEngine,
  speakInstruction,
  subscribeAgentTtsBecameIdle,
  subscribeOpenaiTtsStatus,
  subscribeTtsPlaying,
  unlockAudioFromUserGesture,
  type OpenaiTtsVoiceId,
  type TtsEngine,
} from "../lib/agentTts";
import {
  matchTaskAnchorToDetection,
  type AnchorMatchableDetection,
} from "../lib/taskAnchorMatch";

import { AgentDockOrbit } from "./AgentDockOrbit";
import { AgentRevealText } from "./AgentRevealText";
import {
  PerceptionPanel,
  type DetectorPresetId,
  type PerceptionPanelView,
  type SessionTaskLogEntry,
} from "./PerceptionPanel";
import { SettingsModal } from "./SettingsModal";
import { TwinBrandHud } from "./TwinBrandHud";
import { TwinStatusBar } from "./TwinStatusBar";

const DEFAULT_WS =
  process.env.NEXT_PUBLIC_INFERENCE_WS_URL ?? "ws://127.0.0.1:8000/ws/infer";

function inferHttpBaseFromWs(wsUrl: string): string {
  const override = process.env.NEXT_PUBLIC_INFERENCE_HTTP_URL?.trim();
  if (override) return override.replace(/\/$/, "");
  try {
    const u = new URL(wsUrl);
    u.protocol = u.protocol === "wss:" ? "https:" : "http:";
    u.pathname = "";
    u.search = "";
    u.hash = "";
    return u.toString().replace(/\/$/, "");
  } catch {
    return "http://127.0.0.1:8000";
  }
}

type AgentAction = { name: string; args: Record<string, unknown> };

type AgentVoice = {
  speak: string;
  should_speak: boolean;
  supersede: boolean;
  phase: string;
};

type AgentStepOk = {
  say: string;
  instruction?: string;
  /** Effective line sent to the voice gate / TTS (model instruction or tool-derived). */
  spoken_line?: string;
  actions: AgentAction[];
  state_version?: number;
  task_anchor?: string;
  inferred_held_object?: string;
  voice?: AgentVoice;
};

/** Soft spacing between non-supersede TTS schedules (server also enforces VOICE_MIN_INTERVAL_SEC). */
const CLIENT_TTS_MIN_GAP_MS = 250;
/** Re-speak the last tool-step line if the user has heard no TTS for this long (playback idle). */
const TOOL_REPEAT_SILENCE_MS = 4000;

function parseFastApiDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((x) =>
        typeof x === "object" && x !== null && "msg" in x
          ? String((x as { msg: string }).msg)
          : JSON.stringify(x)
      )
      .join("; ");
  }
  return JSON.stringify(detail);
}

const FMT_JPEG = 1;
const FMT_WEBP = 2;
const WEBP_QUALITY = 0.5;
const JPEG_QUALITY = 0.72;

type Detection = {
  label: string;
  conf: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  cx: number;
  cy: number;
  rel_depth: number;
};

type InferResponse = {
  w: number;
  h: number;
  detections: Detection[];
  error?: string;
  depth_jpeg_b64?: string;
  depth_preview_error?: string;
};

/**
 * Human-readable zone name (e.g. "Pacific Time"), not the raw IANA id.
 */
function friendlyTimeZoneLabel(): string {
  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const parts = new Intl.DateTimeFormat(undefined, {
      timeZone: tz,
      timeZoneName: "longGeneric",
    }).formatToParts(new Date());
    const name = parts.find((p) => p.type === "timeZoneName")?.value?.trim();
    if (name) return name;
    return tz.replaceAll("_", " ");
  } catch {
    return "—";
  }
}

function depthToAccent(relDepth: number): string {
  const t = Math.min(1, Math.max(0, relDepth / 85));
  const hue = 168 - t * 100;
  return `hsl(${hue} 82% 58%)`;
}

function drawTeslaCorners(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  color: string,
  arm: number
): void {
  const L = Math.max(10, Math.min(arm, w * 0.12, h * 0.12));
  ctx.strokeStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 7;
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  ctx.beginPath();
  ctx.moveTo(x, y + L);
  ctx.lineTo(x, y);
  ctx.lineTo(x + L, y);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(x + w - L, y);
  ctx.lineTo(x + w, y);
  ctx.lineTo(x + w, y + L);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(x, y + h - L);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x + L, y + h);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(x + w - L, y + h);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x + w, y + h - L);
  ctx.stroke();

  ctx.shadowBlur = 0;
}

function supportsWebpEncode(): Promise<boolean> {
  return new Promise((resolve) => {
    const c = document.createElement("canvas");
    c.width = 2;
    c.height = 2;
    c.toBlob(
      (b) => resolve(!!b && b.type === "image/webp"),
      "image/webp",
      0.5
    );
  });
}

function labelLooksLikeContinuity(label: string): boolean {
  const s = label.toLowerCase();
  return (
    s.includes("continuity") ||
    s.includes("iphone") ||
    s.includes("ipad")
  );
}

/** Pick a default device id: Continuity / iPhone-style label first, else first video input. */
function defaultVideoDeviceId(devices: MediaDeviceInfo[]): string {
  const v = devices.filter((d) => d.kind === "videoinput");
  const named = v.find((d) => d.label && labelLooksLikeContinuity(d.label));
  if (named?.deviceId) return named.deviceId;
  return v[0]?.deviceId ?? "";
}

function detectionToPerceptionTrack(d: {
  label: string;
  conf: number;
  rel_depth: number;
  cx: number;
}) {
  return {
    label: d.label,
    conf: d.conf,
    rel_depth: d.rel_depth,
    cx: d.cx,
  };
}

export default function CameraOverlay() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const rafRef = useRef(0);
  const busyRef = useRef(false);
  const useWebpRef = useRef(true);
  const streamRef = useRef<MediaStream | null>(null);
  const streamingRef = useRef(false);
  const reconnectDelayRef = useRef(1000);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const connectWsRef = useRef<() => void>(() => {});
  const switchingPresetRef = useRef(false);
  const prevDetectorPresetRef = useRef<DetectorPresetId>("oiv7");
  const agentStepInFlightRef = useRef(false);
  const inferReadyRef = useRef(false);
  const sessionIdRef = useRef<string | null>(null);
  /** True while Web Speech or OpenAI TTS is playing or queued; drives POST is_tts_playing. */
  const isTtsPlayingRef = useRef(false);
  const ttsPendingTimerRef = useRef<number | null>(null);
  const lastNonSupersedeTtsAtRef = useRef(0);
  /** Last successful infer payload; used to redraw overlay after canvas resize (setting w/h clears bitmap). */
  const latestInferRef = useRef<InferResponse | null>(null);
  const inferMsgCountRef = useRef(0);
  const prevSayForAnimRef = useRef<string | null>(null);
  const lastTaskLogSigRef = useRef<string>("");
  /** Timestamp from `subscribeAgentTtsBecameIdle` (0 = never idle since agent on). */
  const lastTtsIdleAtRef = useRef(0);
  /** Line last spoken for a step that included tool actions (for silence reminder). */
  const lastToolRelatedTtsLineRef = useRef("");

  const [streaming, setStreaming] = useState(false);
  const [status, setStatus] = useState("Standby");
  const [wsUrl, setWsUrl] = useState(DEFAULT_WS);
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");
  const [devicesLoaded, setDevicesLoaded] = useState(false);
  const [detectorPreset, setDetectorPreset] = useState<DetectorPresetId>("oiv7");
  /** Client-only (null on SSR/first paint) so hydration matches; set in useEffect. */
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [agentSay, setAgentSay] = useState("");
  const [agentInstruction, setAgentInstruction] = useState("");
  const [agentActions, setAgentActions] = useState<AgentAction[]>([]);
  const [agentVoice, setAgentVoice] = useState<AgentVoice | null>(null);
  const [agentTaskAnchor, setAgentTaskAnchor] = useState("");
  const [agentInferredHeld, setAgentInferredHeld] = useState("");
  const [ttsQueueLen, setTtsQueueLen] = useState(0);
  const [ttsEngine, setTtsEngineState] = useState<TtsEngine>(() => getTtsEngine());
  const [openaiVoice, setOpenaiVoiceState] = useState<OpenaiTtsVoiceId>(() =>
    getOpenaiTtsVoice()
  );
  const [openaiRotate, setOpenaiRotateState] = useState(() =>
    getOpenaiTtsRotateVoices()
  );
  const [openaiTtsLabel, setOpenaiTtsLabel] = useState("");
  const [agentNote, setAgentNote] = useState("");
  /** Backend LATEST_STATE exists only after at least one successful infer for this session. */
  const [inferReady, setInferReady] = useState(false);
  const [hudFrame, setHudFrame] = useState<InferResponse | null>(null);
  const [inferHz, setInferHz] = useState(0);
  const [wsConnected, setWsConnected] = useState(false);
  const [wsConnecting, setWsConnecting] = useState(false);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [settingsModalOpen, setSettingsModalOpen] = useState(false);
  const [depthPreviewUrl, setDepthPreviewUrl] = useState<string | null>(null);
  const [depthPreviewError, setDepthPreviewError] = useState<string | null>(null);
  const [depthFrameSeq, setDepthFrameSeq] = useState(0);
  const [perceptionPanelView, setPerceptionPanelView] =
    useState<PerceptionPanelView>("perception");
  const [failedDepthHttpSrc, setFailedDepthHttpSrc] = useState<string | null>(null);
  const [clockTick, setClockTick] = useState(() => Date.now());
  const [geoLabel, setGeoLabel] = useState(friendlyTimeZoneLabel);
  /** Voice / step agent: off by default after camera engages; press A to run. */
  const [agentEnabled, setAgentEnabled] = useState(false);
  /** Bumps when `agentSay` is replaced (not extended) so prior line exits upward. */
  const [sayAnimEpoch, setSayAnimEpoch] = useState(0);
  const [sessionTaskLog, setSessionTaskLog] = useState<SessionTaskLogEntry[]>([]);
  /** Task-anchor glow: keep last box while fading out (400ms) to mask detection jitter. */
  const [anchorGlowDisplay, setAnchorGlowDisplay] = useState<{
    det: AnchorMatchableDetection;
    fading: boolean;
  } | null>(null);

  const depthHttpSrc = useMemo(() => {
    if (!streaming || !sessionId || !inferReady) return null;
    const base = inferHttpBaseFromWs(wsUrl);
    return `${base}/v1/depth_preview?session_id=${encodeURIComponent(sessionId)}&v=${depthFrameSeq}`;
  }, [streaming, sessionId, inferReady, wsUrl, depthFrameSeq]);

  const depthImgSrc = useMemo(() => {
    if (depthHttpSrc && depthHttpSrc !== failedDepthHttpSrc) return depthHttpSrc;
    return depthPreviewUrl;
  }, [depthHttpSrc, depthPreviewUrl, failedDepthHttpSrc]);

  const clockParts = useMemo(() => {
    const d = new Date(clockTick);
    return {
      weekday: d.toLocaleDateString(undefined, { weekday: "short" }),
      dateLine: d.toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
      }),
      time: d.toLocaleTimeString(undefined, {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }),
    };
  }, [clockTick]);

  useEffect(() => {
    const id = window.setInterval(() => setClockTick(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const { latitude: lat, longitude: lon } = pos.coords;
        const ns = lat >= 0 ? "N" : "S";
        const ew = lon >= 0 ? "E" : "W";
        const coordLabel = `${Math.abs(lat).toFixed(3)}°${ns} · ${Math.abs(lon).toFixed(3)}°${ew}`;
        setGeoLabel(coordLabel);
        try {
          const r = await fetch(
            `/api/reverse-geocode?lat=${encodeURIComponent(String(lat))}&lon=${encodeURIComponent(String(lon))}`
          );
          if (!r.ok) return;
          const body = (await r.json()) as { label?: string | null };
          if (body.label?.trim()) setGeoLabel(body.label.trim());
        } catch {
          /* keep coordinates */
        }
      },
      () => {},
      { enableHighAccuracy: false, timeout: 12_000, maximumAge: 120_000 }
    );
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && settingsModalOpen) {
        e.preventDefault();
        setSettingsModalOpen(false);
        return;
      }
      if (e.key === "." && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setSettingsModalOpen((open) => !open);
        return;
      }
      if (e.key === "a" || e.key === "A") {
        const el = e.target;
        if (
          el instanceof HTMLInputElement ||
          el instanceof HTMLTextAreaElement ||
          el instanceof HTMLSelectElement
        ) {
          return;
        }
        if (el instanceof HTMLElement && el.isContentEditable) return;
        if (!streamingRef.current) return;
        e.preventDefault();
        setAgentEnabled((v) => !v);
        return;
      }
      if (e.key !== "/") return;
      const el = e.target;
      if (
        el instanceof HTMLInputElement ||
        el instanceof HTMLTextAreaElement ||
        el instanceof HTMLSelectElement
      ) {
        return;
      }
      if (el instanceof HTMLElement && el.isContentEditable) return;
      e.preventDefault();
      setRightPanelOpen((open) => !open);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [settingsModalOpen]);

  useEffect(() => {
    const s = agentSay;
    const prev = prevSayForAnimRef.current;
    if (s === prev) return;

    if (prev === null && s === "") {
      prevSayForAnimRef.current = "";
      return;
    }

    const extended =
      typeof prev === "string" && prev.length > 0 && s.startsWith(prev);
    if (!extended) {
      setSayAnimEpoch((e) => e + 1);
    }
    prevSayForAnimRef.current = s;
  }, [agentSay]);

  const clearReconnect = () => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  };

  const drawDetections = useCallback((data: InferResponse) => {
    const overlay = overlayRef.current;
    const video = videoRef.current;
    if (!overlay || !video) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    const ow = overlay.width;
    const oh = overlay.height;
    ctx.clearRect(0, 0, ow, oh);
    if (!data.detections?.length) return;

    ctx.font = "600 11px var(--font-geist-mono, ui-monospace), monospace";

    for (const d of data.detections) {
      const x = d.x1 * ow;
      const y = d.y1 * oh;
      const w = (d.x2 - d.x1) * ow;
      const h = (d.y2 - d.y1) * oh;
      const accent = depthToAccent(d.rel_depth);
      drawTeslaCorners(ctx, x, y, w, h, accent, 26);

      const label = `${d.label}  ${(d.conf * 100).toFixed(0)}%  ·  z ${d.rel_depth.toFixed(1)}`;
      const padX = 8;
      const tw = ctx.measureText(label).width;
      const pillW = tw + padX * 2;
      const pillH = 20;
      const py = Math.max(2, y - pillH - 4);

      ctx.fillStyle = "rgba(8, 10, 14, 0.72)";
      ctx.strokeStyle = "rgba(255,255,255,0.12)";
      ctx.lineWidth = 1;
      const r = 6;
      ctx.beginPath();
      ctx.roundRect(x, py, pillW, pillH, r);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = accent;
      ctx.shadowColor = accent;
      ctx.shadowBlur = 4;
      ctx.fillText(label, x + padX, py + 14);
      ctx.shadowBlur = 0;
    }
  }, []);

  useEffect(() => {
    const id =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `sid-${Date.now()}`;
    queueMicrotask(() => setSessionId(id));
  }, []);

  useEffect(() => {
    return subscribeTtsPlaying((v) => {
      isTtsPlayingRef.current = v;
    });
  }, []);

  useEffect(() => {
    return subscribeAgentTtsBecameIdle((t) => {
      lastTtsIdleAtRef.current = t;
    });
  }, []);

  useEffect(() => {
    if (!agentEnabled) {
      lastTtsIdleAtRef.current = 0;
      lastToolRelatedTtsLineRef.current = "";
    }
  }, [agentEnabled]);

  useEffect(() => {
    return subscribeOpenaiTtsStatus(({ status, detail }) => {
      if (status === "idle") setOpenaiTtsLabel("");
      else if (status === "loading") setOpenaiTtsLabel(detail);
      else if (status === "ready") setOpenaiTtsLabel(detail);
      else setOpenaiTtsLabel(`OpenAI TTS error: ${detail}`);
    });
  }, []);

  useEffect(() => {
    setTtsEngine(ttsEngine);
  }, [ttsEngine]);

  useEffect(() => {
    if (!streaming) return;
    const id = window.setInterval(() => {
      setTtsQueueLen(getTtsQueueDepth());
      isTtsPlayingRef.current = getIsTtsPlaying();
    }, 400);
    return () => window.clearInterval(id);
  }, [streaming]);

  useEffect(() => {
    const tick = window.setInterval(() => {
      setInferHz(inferMsgCountRef.current);
      inferMsgCountRef.current = 0;
    }, 1000);
    return () => window.clearInterval(tick);
  }, []);

  const connectWs = useCallback(() => {
    clearReconnect();
    if (!streamingRef.current || !sessionId) return;
    let urlToOpen = wsUrl;
    try {
      const u = new URL(wsUrl);
      u.searchParams.set("model", detectorPreset);
      u.searchParams.set("session_id", sessionId);
      urlToOpen = u.toString();
    } catch {
      /* keep wsUrl if not parseable */
    }
    try {
      const ws = new WebSocket(urlToOpen);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      setStatus("Linking perception stack…");
      setWsConnecting(true);
      setWsConnected(false);
      ws.onopen = () => {
        reconnectDelayRef.current = 1000;
        setInferReady(false);
        setWsConnecting(false);
        setWsConnected(true);
        setStatus("Perception live");
      };
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(String(ev.data)) as InferResponse;
          busyRef.current = false;
          inferMsgCountRef.current += 1;
          if (data.error) {
            latestInferRef.current = null;
            setStatus(`Server: ${data.error}`);
          } else if (data.w > 0) {
            latestInferRef.current = data;
            setInferReady(true);
            setDepthFrameSeq((n) => n + 1);
          }
          if (data.depth_jpeg_b64) {
            setDepthPreviewUrl(`data:image/jpeg;base64,${data.depth_jpeg_b64}`);
            setDepthPreviewError(null);
          } else {
            setDepthPreviewUrl(null);
            const err = data.depth_preview_error?.trim();
            setDepthPreviewError(err || null);
          }
          drawDetections(data);
          // Keep HUD / object count in sync with every server frame (no throttle).
          setHudFrame(data);
        } catch {
          busyRef.current = false;
        }
      };
      ws.onerror = () => {
        setStatus("Stream fault");
      };
      ws.onclose = () => {
        wsRef.current = null;
        busyRef.current = false;
        setWsConnecting(false);
        setWsConnected(false);
        if (!streamingRef.current) return;
        if (switchingPresetRef.current) {
          switchingPresetRef.current = false;
          connectWsRef.current();
          return;
        }
        setStatus("Reconnecting…");
        const delay = reconnectDelayRef.current;
        reconnectDelayRef.current = Math.min(delay * 2, 10_000);
        reconnectTimerRef.current = setTimeout(() => connectWsRef.current(), delay);
      };
    } catch {
      setStatus("Invalid stream URL");
    }
  }, [wsUrl, detectorPreset, sessionId, drawDetections]);

  useEffect(() => {
    connectWsRef.current = connectWs;
  }, [connectWs]);

  useEffect(() => {
    inferReadyRef.current = inferReady;
  }, [inferReady]);

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    if (!streaming) {
      prevDetectorPresetRef.current = detectorPreset;
      return;
    }
    if (prevDetectorPresetRef.current === detectorPreset) return;
    prevDetectorPresetRef.current = detectorPreset;
    switchingPresetRef.current = true;
    clearReconnect();
    wsRef.current?.close();
  }, [streaming, detectorPreset]);

  const refreshVideoDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const list = await navigator.mediaDevices.enumerateDevices();
      const vids = list.filter((d) => d.kind === "videoinput");
      setVideoDevices(vids);
      setDevicesLoaded(true);
      setSelectedDeviceId((prev) => {
        if (prev && vids.some((d) => d.deviceId === prev)) return prev;
        return defaultVideoDeviceId(vids);
      });
    } catch {
      setDevicesLoaded(true);
    }
  }, []);

  useEffect(() => {
    const md = navigator.mediaDevices;
    if (!md) return;
    const t = window.setTimeout(() => {
      void refreshVideoDevices();
    }, 0);
    md.addEventListener("devicechange", refreshVideoDevices);
    return () => {
      window.clearTimeout(t);
      md.removeEventListener("devicechange", refreshVideoDevices);
    };
  }, [refreshVideoDevices]);

  const stopCamera = useCallback(() => {
    streamingRef.current = false;
    setStreaming(false);
    setAgentSay("");
    setAgentInstruction("");
    setAgentActions([]);
    if (ttsPendingTimerRef.current) {
      window.clearTimeout(ttsPendingTimerRef.current);
      ttsPendingTimerRef.current = null;
    }
    setAgentNote("");
    setSessionTaskLog([]);
    lastTaskLogSigRef.current = "";
    setInferReady(false);
    resetAgentTts();
    setTtsQueueLen(0);
    setHudFrame(null);
    latestInferRef.current = null;
    setWsConnected(false);
    setWsConnecting(false);
    setDepthPreviewUrl(null);
    setDepthPreviewError(null);
    setDepthFrameSeq(0);
    setFailedDepthHttpSrc(null);
    setPerceptionPanelView("perception");
    clearReconnect();
    busyRef.current = false;
    cancelAnimationFrame(rafRef.current);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    const s = streamRef.current;
    if (s) {
      s.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    const v = videoRef.current;
    if (v) v.srcObject = null;
    const o = overlayRef.current;
    if (o) {
      const ctx = o.getContext("2d");
      ctx?.clearRect(0, 0, o.width, o.height);
    }
    setStatus("Standby");
    void refreshVideoDevices();
  }, [refreshVideoDevices]);

  const startCamera = useCallback(async () => {
    if (streamingRef.current) return;
    const webpOk = await supportsWebpEncode();
    useWebpRef.current = webpOk;
    try {
      await refreshVideoDevices();
      const video: MediaTrackConstraints = {
        width: { ideal: 640, max: 1280 },
        height: { ideal: 360, max: 720 },
      };
      if (selectedDeviceId) {
        video.deviceId = { exact: selectedDeviceId };
      } else {
        video.facingMode = "user";
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video,
        audio: false,
      });
      void refreshVideoDevices();
      streamRef.current = stream;
      const v = videoRef.current;
      if (!v) return;
      v.srcObject = stream;
      await v.play();
      streamingRef.current = true;
      setStreaming(true);
      unlockAudioFromUserGesture();
      void resumeAudioContextAfterUserGesture();
      setStatus(webpOk ? "Vision · WebP" : "Vision · JPEG");
    } catch (e) {
      setStatus(`Camera failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, [refreshVideoDevices, selectedDeviceId]);

  /** Open WebSocket when camera is on and session id is ready (avoids duplicate opens). */
  useEffect(() => {
    if (!streaming || !sessionId) return;
    const st = wsRef.current?.readyState;
    if (st === WebSocket.OPEN || st === WebSocket.CONNECTING) return;
    queueMicrotask(() => connectWs());
  }, [streaming, sessionId, connectWs]);

  const tick = useCallback(() => {
    if (document.visibilityState === "hidden") return;
    const video = videoRef.current;
    const cap = captureRef.current;
    const ws = wsRef.current;
    if (
      !streamingRef.current ||
      !video ||
      !cap ||
      !ws ||
      ws.readyState !== WebSocket.OPEN
    ) {
      return;
    }
    if (busyRef.current) return;
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (w === 0 || h === 0) return;

    busyRef.current = true;
    cap.width = w;
    cap.height = h;
    const ctx = cap.getContext("2d");
    if (!ctx) {
      busyRef.current = false;
      return;
    }
    ctx.drawImage(video, 0, 0, w, h);
    const mime = useWebpRef.current ? "image/webp" : "image/jpeg";
    const q = useWebpRef.current ? WEBP_QUALITY : JPEG_QUALITY;
    cap.toBlob(
      (blob) => {
        if (!blob) {
          if (useWebpRef.current) {
            useWebpRef.current = false;
          }
          busyRef.current = false;
          return;
        }
        const fmt = useWebpRef.current ? FMT_WEBP : FMT_JPEG;
        blob.arrayBuffer().then((buf) => {
          const cur = wsRef.current;
          if (!cur || cur.readyState !== WebSocket.OPEN) {
            busyRef.current = false;
            return;
          }
          const body = new Uint8Array(1 + buf.byteLength);
          body[0] = fmt;
          body.set(new Uint8Array(buf), 1);
          cur.send(body);
        });
      },
      mime,
      q
    );
  }, []);

  useEffect(() => {
    if (!streaming) return;
    const loop = () => {
      tick();
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [streaming, tick]);

  useEffect(() => {
    if (!streaming || !sessionId || !inferReady || !agentEnabled) return;
    const base = (
      process.env.NEXT_PUBLIC_AGENT_HTTP_URL ?? inferHttpBaseFromWs(wsUrl)
    ).replace(/\/$/, "");
    const gapMs = 750;
    let cancelled = false;
    let timeoutId: number | undefined;

    const stepOnce = async () => {
      if (cancelled || !streamingRef.current) return;
      const sid = sessionIdRef.current;
      if (!sid || !inferReadyRef.current) {
        timeoutId = window.setTimeout(stepOnce, 200);
        return;
      }
      if (agentStepInFlightRef.current) {
        timeoutId = window.setTimeout(stepOnce, 100);
        return;
      }
      agentStepInFlightRef.current = true;
      try {
        const r = await fetch(
          `${base}/v1/agent/step?session_id=${encodeURIComponent(sid)}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              is_tts_playing: isTtsPlayingRef.current,
            }),
          }
        );
        const text = await r.text();
        let body: unknown;
        try {
          body = JSON.parse(text) as unknown;
        } catch {
          body = null;
        }
        if (!r.ok) {
          const detail =
            body &&
            typeof body === "object" &&
            body !== null &&
            "detail" in body
              ? parseFastApiDetail((body as { detail: unknown }).detail)
              : text.slice(0, 160);
          if (r.status === 429) {
            return;
          }
          setAgentNote(`${r.status}: ${detail}`);
          return;
        }
        const ok = body as AgentStepOk;
        if (ok && typeof ok.say === "string") {
          setAgentSay(ok.say);
          setAgentInstruction(
            typeof ok.spoken_line === "string"
              ? ok.spoken_line
              : typeof ok.instruction === "string"
                ? ok.instruction
                : ""
          );
          const rawActions = Array.isArray(ok.actions) ? ok.actions : [];
          setAgentActions(rawActions);
          if (rawActions.length > 0) {
            const sig = rawActions
              .map((a) => `${a.name}:${JSON.stringify(a.args)}`)
              .join("|");
            if (sig !== lastTaskLogSigRef.current) {
              lastTaskLogSigRef.current = sig;
              const summary = rawActions.map((a) => a.name).join(" · ");
              const id = `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
              setSessionTaskLog((prev) =>
                [{ id, summary }, ...prev].slice(0, 15)
              );
            }
          }
          setAgentTaskAnchor(
            typeof ok.task_anchor === "string" ? ok.task_anchor : ""
          );
          setAgentInferredHeld(
            typeof ok.inferred_held_object === "string"
              ? ok.inferred_held_object
              : ""
          );
          const v = ok.voice;
          setAgentVoice(
            v &&
              typeof v.speak === "string" &&
              typeof v.should_speak === "boolean" &&
              typeof v.supersede === "boolean" &&
              typeof v.phase === "string"
              ? v
              : null
          );
          let spokeThisStep = false;
          if (v && typeof v.speak === "string" && v.should_speak) {
            const line = v.speak.trim();
            if (line) {
              if (rawActions.length > 0) {
                lastToolRelatedTtsLineRef.current = line;
              }
              if (ttsPendingTimerRef.current) {
                window.clearTimeout(ttsPendingTimerRef.current);
                ttsPendingTimerRef.current = null;
              }
              const supersede = v.supersede;
              const flush = () => {
                speakInstruction(line, { supersede });
                setTtsQueueLen(getTtsQueueDepth());
                if (!supersede) {
                  lastNonSupersedeTtsAtRef.current = Date.now();
                }
              };
              if (supersede) {
                flush();
              } else {
                const elapsed = Date.now() - lastNonSupersedeTtsAtRef.current;
                const delay = Math.max(0, CLIENT_TTS_MIN_GAP_MS - elapsed);
                ttsPendingTimerRef.current = window.setTimeout(() => {
                  ttsPendingTimerRef.current = null;
                  flush();
                }, delay);
              }
              spokeThisStep = true;
            }
          }
          if (
            !spokeThisStep &&
            lastToolRelatedTtsLineRef.current &&
            !getIsTtsPlaying() &&
            getTtsQueueDepth() === 0 &&
            lastTtsIdleAtRef.current > 0 &&
            Date.now() - lastTtsIdleAtRef.current >= TOOL_REPEAT_SILENCE_MS
          ) {
            const reminder = lastToolRelatedTtsLineRef.current;
            if (ttsPendingTimerRef.current) {
              window.clearTimeout(ttsPendingTimerRef.current);
              ttsPendingTimerRef.current = null;
            }
            const elapsedGap = Date.now() - lastNonSupersedeTtsAtRef.current;
            const delay = Math.max(0, CLIENT_TTS_MIN_GAP_MS - elapsedGap);
            const flushReminder = () => {
              speakInstruction(reminder, { supersede: false });
              setTtsQueueLen(getTtsQueueDepth());
              lastNonSupersedeTtsAtRef.current = Date.now();
            };
            if (delay === 0) {
              flushReminder();
            } else {
              ttsPendingTimerRef.current = window.setTimeout(() => {
                ttsPendingTimerRef.current = null;
                flushReminder();
              }, delay);
            }
          }
          setAgentNote("");
        }
      } catch (e) {
        setAgentNote(e instanceof Error ? e.message : String(e));
      } finally {
        agentStepInFlightRef.current = false;
        if (!cancelled) {
          timeoutId = window.setTimeout(stepOnce, gapMs);
        }
      }
    };

    void stepOnce();
    return () => {
      cancelled = true;
      if (timeoutId !== undefined) window.clearTimeout(timeoutId);
      if (ttsPendingTimerRef.current) {
        window.clearTimeout(ttsPendingTimerRef.current);
        ttsPendingTimerRef.current = null;
      }
    };
  }, [streaming, sessionId, wsUrl, inferReady, agentEnabled]);

  useEffect(() => {
    const onVis = () => {
      if (document.visibilityState === "hidden") {
        busyRef.current = false;
      }
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    const overlay = overlayRef.current;
    if (!video || !overlay) return;
    const applySizeAndRedraw = () => {
      const vw = video.clientWidth;
      const vh = video.clientHeight;
      if (vw <= 0 || vh <= 0) return;
      if (overlay.width === vw && overlay.height === vh) return;
      overlay.width = vw;
      overlay.height = vh;
      const snap = latestInferRef.current;
      if (snap && !snap.error && snap.w > 0) {
        drawDetections(snap);
      }
    };
    const ro = new ResizeObserver(applySizeAndRedraw);
    ro.observe(video);
    applySizeAndRedraw();
    return () => ro.disconnect();
  }, [streaming, drawDetections]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const detections = hudFrame?.detections ?? [];
  const taskAnchorMatch = useMemo(
    () =>
      agentTaskAnchor.trim()
        ? matchTaskAnchorToDetection(detections, agentTaskAnchor)
        : null,
    [agentTaskAnchor, detections]
  );

  useEffect(() => {
    if (taskAnchorMatch) {
      setAnchorGlowDisplay({ det: taskAnchorMatch, fading: false });
      return;
    }
    setAnchorGlowDisplay((prev) => (prev ? { ...prev, fading: true } : null));
  }, [taskAnchorMatch]);

  useEffect(() => {
    if (!anchorGlowDisplay?.fading) return;
    const t = window.setTimeout(() => setAnchorGlowDisplay(null), 400);
    return () => clearTimeout(t);
  }, [anchorGlowDisplay?.fading]);

  useEffect(() => {
    if (!streaming || !agentEnabled) {
      setAnchorGlowDisplay(null);
    }
  }, [streaming, agentEnabled]);

  const sortedByDepth = [...detections].sort((a, b) => a.rel_depth - b.rel_depth);
  const closest = sortedByDepth[0];
  const focusPerceptionTrack = useMemo(() => {
    if (taskAnchorMatch) return detectionToPerceptionTrack(taskAnchorMatch);
    if (closest) return detectionToPerceptionTrack(closest);
    return null;
  }, [taskAnchorMatch, closest]);
  const focusFromAnchor = Boolean(taskAnchorMatch);
  const meanConf =
    detections.length > 0
      ? detections.reduce((s, d) => s + d.conf, 0) / detections.length
      : 0;

  const wsState = wsConnected
    ? "LIVE"
    : wsConnecting
      ? "SYNC"
      : streaming
        ? "WAIT"
        : "OFF";

  return (
    <div className="flex h-full min-h-0 w-full min-w-0 flex-1 flex-col overflow-hidden">
      <section
        aria-label="Live camera and perception overlay"
        className="console-full-bleed console-viewport-scan twin-scanlines relative flex h-full min-h-0 w-full min-w-0 flex-1 flex-col overflow-hidden bg-black"
      >
        <div className="relative flex h-full min-h-0 w-full min-w-0 flex-1 flex-col overflow-hidden bg-black">
          <video
            ref={videoRef}
            className="absolute inset-0 z-0 h-full w-full min-h-0 object-cover"
            autoPlay
            playsInline
            muted
          />
          {perceptionPanelView === "depth" && depthImgSrc ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={depthImgSrc}
              alt=""
              aria-hidden
              decoding="async"
              className="pointer-events-none absolute inset-0 z-[1] h-full w-full min-h-0 object-cover"
              onError={() => {
                if (depthHttpSrc) setFailedDepthHttpSrc(depthHttpSrc);
              }}
            />
          ) : null}
          <canvas
            ref={overlayRef}
            className="pointer-events-none absolute inset-0 z-[2] h-full w-full min-h-0"
          />
          {streaming && agentEnabled && anchorGlowDisplay ? (
            <motion.div
              className="pointer-events-none absolute z-[3] box-border min-h-[2.5rem] min-w-[2.5rem]"
              initial={false}
              animate={{
                left: `${anchorGlowDisplay.det.x1 * 100}%`,
                top: `${anchorGlowDisplay.det.y1 * 100}%`,
                width: `${(anchorGlowDisplay.det.x2 - anchorGlowDisplay.det.x1) * 100}%`,
                height: `${(anchorGlowDisplay.det.y2 - anchorGlowDisplay.det.y1) * 100}%`,
                opacity: anchorGlowDisplay.fading ? 0 : 1,
              }}
              transition={{
                left: { type: "spring", stiffness: 220, damping: 28, mass: 0.55 },
                top: { type: "spring", stiffness: 220, damping: 28, mass: 0.55 },
                width: { type: "spring", stiffness: 220, damping: 28, mass: 0.55 },
                height: { type: "spring", stiffness: 220, damping: 28, mass: 0.55 },
                opacity: {
                  duration: anchorGlowDisplay.fading ? 0.4 : 0.14,
                  ease: [0.22, 1, 0.36, 1],
                },
              }}
            >
              <AppleIntelligenceGlow
                isActive
                state="focus"
                radius="1rem"
                className="remop-anchor-aie h-full w-full overflow-hidden rounded-2xl"
              >
                <div className="h-full w-full rounded-2xl" aria-hidden />
              </AppleIntelligenceGlow>
            </motion.div>
          ) : null}
          <div className="pointer-events-none absolute inset-0 z-[2] bg-gradient-to-t from-black/45 via-transparent to-black/15" />

          <TwinBrandHud
            weekday={clockParts.weekday}
            dateLine={clockParts.dateLine}
            time={clockParts.time}
            location={geoLabel}
          />

          <div className="relative z-20 flex min-h-0 min-w-0 flex-1 flex-col pointer-events-none">
            <div className="pointer-events-none flex min-h-0 min-w-0 flex-1 flex-col gap-3 overflow-x-clip px-4 pb-[calc(1.35rem+env(safe-area-inset-bottom))] pt-3 sm:px-5 sm:pb-[calc(1.5rem+env(safe-area-inset-bottom))] sm:pt-4 lg:flex-row lg:items-stretch lg:justify-end lg:gap-5 lg:px-6 lg:pt-5">
              <div className="relative flex min-h-0 min-w-0 flex-1 flex-col lg:block">
                <div
                  className="pointer-events-none absolute left-2 top-2 hidden h-8 w-8 rounded-tl border-l-2 border-t-2 border-[rgba(59,158,255,0.35)] shadow-[0_0_24px_rgba(59,158,255,0.07)] lg:block"
                  aria-hidden
                />
                <div
                  className="pointer-events-none absolute right-2 top-2 hidden h-8 w-8 rounded-tr border-r-2 border-t-2 border-[rgba(59,158,255,0.35)] shadow-[0_0_24px_rgba(59,158,255,0.07)] lg:block"
                  aria-hidden
                />
              </div>

              <div
                className={`shrink-0 overflow-hidden transition-[max-width,opacity,margin] duration-300 ease-out ${
                  rightPanelOpen
                    ? "ml-0 max-w-[min(308px,92vw)] opacity-100 lg:ml-2 lg:max-w-[min(308px,34vw)]"
                    : "pointer-events-none ml-0 max-w-0 opacity-0"
                }`}
                aria-hidden={!rightPanelOpen}
              >
                <aside
                  id="perception-panel"
                  className="pointer-events-auto flex max-h-[min(40vh,300px)] min-h-0 w-[min(300px,92vw)] flex-col sm:max-h-[min(46vh,380px)] lg:max-h-full lg:w-[min(300px,32vw)]"
                >
                <PerceptionPanel
                  panelView={perceptionPanelView}
                  onPanelViewChange={setPerceptionPanelView}
                  depthImgSrc={depthImgSrc}
                  focusTarget={focusPerceptionTrack}
                  focusFromAnchor={focusFromAnchor}
                  sessionTaskLog={sessionTaskLog}
                  depthToAccent={depthToAccent}
                  depthPreviewError={depthPreviewError}
                  streaming={streaming}
                />
                </aside>
              </div>
            </div>
          </div>

          <AnimatePresence mode="wait">
            {streaming ? (
              agentEnabled ? (
                <motion.div
                  key="agent-dock"
                  initial={{ opacity: 0, y: 36, scale: 0.94 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{
                    opacity: 0,
                    y: 22,
                    scale: 0.97,
                    transition: { duration: 0.36, ease: [0.22, 1, 0.36, 1] },
                  }}
                  transition={{
                    type: "spring",
                    stiffness: 260,
                    damping: 28,
                    mass: 0.92,
                  }}
                  className="pointer-events-none absolute inset-x-0 bottom-0 z-[100] flex justify-center overflow-visible px-3 sm:px-4"
                  style={{
                    paddingBottom:
                      "max(3.25rem, calc(env(safe-area-inset-bottom, 0px) + 2.75rem))",
                  }}
                >
                  <div className="pointer-events-auto relative isolate w-full min-w-0 max-w-xl sm:max-w-2xl">
                    <motion.div
                      layout
                      transition={{
                        type: "spring",
                        stiffness: 300,
                        damping: 28,
                        mass: 0.88,
                      }}
                      className="glass-panel relative z-[1] max-h-[min(38vh,320px)] w-full overflow-y-auto overscroll-contain rounded-[1.25rem] px-4 py-3.5 sm:px-5 sm:py-4"
                      style={{ willChange: "transform" }}
                    >
                      <motion.h2
                        className="mono-caps text-white/75"
                        initial={{ opacity: 0, y: -6 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{
                          delay: 0.06,
                          type: "spring",
                          stiffness: 380,
                          damping: 32,
                        }}
                      >
                        Agent
                      </motion.h2>
                      {agentNote ? (
                        <motion.p
                          initial={{ opacity: 0, y: -4 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{
                            type: "spring",
                            stiffness: 380,
                            damping: 34,
                          }}
                          className="mt-2 font-mono text-xs leading-relaxed text-[var(--tw-warn)]/85"
                        >
                          {agentNote}
                        </motion.p>
                      ) : null}
                      <div className="relative mt-2.5 min-h-[2.5rem] overflow-hidden">
                        <AnimatePresence initial={false} mode="popLayout">
                          <motion.div
                            key={sayAnimEpoch}
                            initial={{ opacity: 0, y: 18 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -36 }}
                            transition={{
                              type: "spring",
                              stiffness: 340,
                              damping: 32,
                              mass: 0.8,
                              opacity: { duration: 0.24 },
                            }}
                            className="text-base font-semibold leading-snug tracking-tight text-white sm:text-lg"
                            style={{ willChange: "transform" }}
                          >
                            <AgentRevealText text={agentSay} charPerSec={112} />
                          </motion.div>
                        </AnimatePresence>
                      </div>
                      <motion.div
                        layout
                        className="mt-3 border-t border-white/8 pt-3"
                        transition={{
                          type: "spring",
                          stiffness: 340,
                          damping: 32,
                        }}
                      >
                        <p className="font-mono text-[10px] leading-relaxed text-white/58 sm:text-[11px]">
                          <span className="text-white/65">Anchor</span>{" "}
                          {agentTaskAnchor ? `"${agentTaskAnchor}"` : "—"}
                          <span className="text-white/42"> · </span>
                          <span className="text-white/65">Held</span>{" "}
                          {agentInferredHeld ? `"${agentInferredHeld}"` : "—"}
                        </p>
                        {agentVoice ? (
                          <ul className="mt-2 space-y-1 font-mono text-[10px] text-white/58 sm:text-[11px]">
                            <li>
                              <span className="text-white/52">Phase</span>{" "}
                              <span className="text-[var(--tw-accent)]/80">{agentVoice.phase}</span>
                            </li>
                            <li>
                              <span className="text-white/52">TTS</span>{" "}
                              {agentInstruction.trim() || "—"}
                            </li>
                            <li>
                              <span className="text-white/52">Speak</span>{" "}
                              {agentVoice.speak || "—"}
                              <span className="text-white/42"> · </span>
                              <span className="text-[var(--tw-teal)]/70">queue ~{ttsQueueLen}</span>
                            </li>
                          </ul>
                        ) : null}
                      </motion.div>
                      <AnimatePresence initial={false}>
                        {agentActions.length > 0 ? (
                          <motion.ul
                            key="actions"
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: 4 }}
                            transition={{
                              type: "spring",
                              stiffness: 360,
                              damping: 34,
                            }}
                            className="mt-3 space-y-1.5 border-t border-white/8 pt-3 font-mono text-[11px] text-white/72 sm:text-[12px]"
                          >
                            {agentActions.map((a, i) => (
                              <motion.li
                                key={`${a.name}-${i}`}
                                layout
                                initial={{ opacity: 0, x: -14 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: -8 }}
                                transition={{
                                  type: "spring",
                                  stiffness: 400,
                                  damping: 30,
                                  delay: i * 0.045,
                                }}
                                className="rounded-lg border border-white/[0.08] bg-black/38 px-2.5 py-1.5"
                              >
                                <span className="font-semibold text-[var(--tw-teal)]/85">{a.name}</span>
                                {a.args && Object.keys(a.args).length > 0
                                  ? ` ${JSON.stringify(a.args)}`
                                  : ""}
                              </motion.li>
                            ))}
                          </motion.ul>
                        ) : null}
                      </AnimatePresence>
                    </motion.div>
                    <div
                      className="pointer-events-none absolute inset-0 z-[60] overflow-visible rounded-[1.25rem]"
                      aria-hidden
                    >
                      <AgentDockOrbit />
                    </div>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="agent-paused-hint"
                  role="status"
                  initial={{ opacity: 0, y: 16, scale: 0.97 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{
                    opacity: 0,
                    y: 12,
                    scale: 0.98,
                    transition: { duration: 0.32, ease: [0.22, 1, 0.36, 1] },
                  }}
                  transition={{
                    type: "spring",
                    stiffness: 300,
                    damping: 30,
                    mass: 0.88,
                  }}
                  className="pointer-events-none absolute inset-x-0 bottom-0 z-[100] flex justify-center px-3 sm:px-4"
                  style={{
                    paddingBottom:
                      "max(3.25rem, calc(env(safe-area-inset-bottom, 0px) + 2.75rem))",
                  }}
                >
                  <motion.p
                    className="mono-caps rounded-full border border-white/[0.12] bg-black/50 px-5 py-2.5 text-center text-[10px] text-white/58 backdrop-blur-md sm:text-[11px]"
                    style={{
                      boxShadow: "0 8px 32px -8px rgba(0,0,0,0.4)",
                    }}
                  >
                    Agent off · press A to enable
                  </motion.p>
                </motion.div>
              )
            ) : null}
          </AnimatePresence>

          <TwinStatusBar
            objects={detections.length}
            wsState={wsState}
            frameW={hudFrame?.w ?? 0}
            frameH={hudFrame?.h ?? 0}
            inferHz={inferHz}
            streaming={streaming}
            agentLive={streaming && agentEnabled}
            meanConfPct={
              detections.length > 0 ? `${(meanConf * 100).toFixed(0)}%` : null
            }
          />
        </div>
      </section>

      <canvas ref={captureRef} className="hidden" aria-hidden />

      <SettingsModal
        open={settingsModalOpen}
        onClose={() => setSettingsModalOpen(false)}
        status={status}
        sessionId={sessionId}
        streaming={streaming}
        onEngageCamera={() => void startCamera()}
        onStopCamera={stopCamera}
        detectorPreset={detectorPreset}
        onDetectorPresetChange={setDetectorPreset}
        videoDevices={videoDevices}
        selectedDeviceId={selectedDeviceId}
        onSelectedDeviceIdChange={setSelectedDeviceId}
        devicesLoaded={devicesLoaded}
        onRefreshVideoDevices={() => void refreshVideoDevices()}
        wsUrl={wsUrl}
        onWsUrlChange={setWsUrl}
        ttsEngine={ttsEngine}
        onTtsEngineChange={(engine) => {
          setTtsEngineState(engine);
          setTtsQueueLen(0);
        }}
        openaiVoice={openaiVoice}
        onOpenaiVoiceChange={(v) => {
          setOpenaiVoiceState(v);
        }}
        openaiRotate={openaiRotate}
        onOpenaiRotateChange={(on) => {
          setOpenaiRotateState(on);
        }}
        openaiTtsLabel={openaiTtsLabel}
        agentEnabled={agentEnabled}
        agentToggleDisabled={!streaming}
        onAgentEnabledChange={setAgentEnabled}
      />
    </div>
  );
}
