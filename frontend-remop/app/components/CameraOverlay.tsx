"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const DEFAULT_WS =
  process.env.NEXT_PUBLIC_INFERENCE_WS_URL ?? "ws://127.0.0.1:8000/ws/infer";

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
};

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

  const [streaming, setStreaming] = useState(false);
  const [status, setStatus] = useState("Idle — click Start camera");
  const [wsUrl, setWsUrl] = useState(DEFAULT_WS);

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
    ctx.lineWidth = 2;
    ctx.font = "14px system-ui, sans-serif";
    for (const d of data.detections) {
      const x = d.x1 * ow;
      const y = d.y1 * oh;
      const w = (d.x2 - d.x1) * ow;
      const h = (d.y2 - d.y1) * oh;
      ctx.strokeStyle = "#22c55e";
      ctx.strokeRect(x, y, w, h);
      const label = `${d.label} ${(d.conf * 100).toFixed(0)}% d=${d.rel_depth.toFixed(1)}`;
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.fillRect(x, y - 18, tw + 8, 18);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x + 4, y - 5);
    }
  }, []);

  const connectWs = useCallback(() => {
    clearReconnect();
    if (!streamingRef.current) return;
    try {
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      setStatus("Connecting…");
      ws.onopen = () => {
        reconnectDelayRef.current = 1000;
        setStatus("Connected — inferring");
      };
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(String(ev.data)) as InferResponse;
          busyRef.current = false;
          if (data.error) {
            setStatus(`Server: ${data.error}`);
          }
          drawDetections(data);
        } catch {
          busyRef.current = false;
        }
      };
      ws.onerror = () => {
        setStatus("WebSocket error");
      };
      ws.onclose = () => {
        wsRef.current = null;
        busyRef.current = false;
        if (!streamingRef.current) return;
        setStatus("Disconnected — reconnecting…");
        const delay = reconnectDelayRef.current;
        reconnectDelayRef.current = Math.min(delay * 2, 10_000);
        reconnectTimerRef.current = setTimeout(() => connectWsRef.current(), delay);
      };
    } catch {
      setStatus("Invalid WebSocket URL");
    }
  }, [wsUrl, drawDetections]);

  useEffect(() => {
    connectWsRef.current = connectWs;
  }, [connectWs]);

  const stopCamera = useCallback(() => {
    streamingRef.current = false;
    setStreaming(false);
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
    setStatus("Stopped");
  }, []);

  const startCamera = useCallback(async () => {
    if (streamingRef.current) return;
    const webpOk = await supportsWebpEncode();
    useWebpRef.current = webpOk;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640, max: 1280 },
          height: { ideal: 360, max: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      const v = videoRef.current;
      if (!v) return;
      v.srcObject = stream;
      await v.play();
      streamingRef.current = true;
      setStreaming(true);
      setStatus(webpOk ? "Camera on (WebP)" : "Camera on (JPEG fallback)");
      connectWs();
    } catch (e) {
      setStatus(`Camera failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, [connectWs]);

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
    const ro = new ResizeObserver(() => {
      overlay.width = video.clientWidth;
      overlay.height = video.clientHeight;
    });
    ro.observe(video);
    overlay.width = video.clientWidth;
    overlay.height = video.clientHeight;
    return () => ro.disconnect();
  }, [streaming]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  return (
    <div className="flex w-full max-w-4xl flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={startCamera}
          disabled={streaming}
          className="rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900"
        >
          Start camera
        </button>
        <button
          type="button"
          onClick={stopCamera}
          disabled={!streaming}
          className="rounded-lg border border-zinc-300 px-4 py-2 text-sm font-medium dark:border-zinc-600"
        >
          Stop
        </button>
        <span className="text-sm text-zinc-600 dark:text-zinc-400">{status}</span>
      </div>
      <label className="flex flex-col gap-1 text-sm text-zinc-600 dark:text-zinc-400">
        Inference WebSocket URL
        <input
          className="rounded border border-zinc-300 bg-white px-2 py-1 font-mono text-xs dark:border-zinc-600 dark:bg-zinc-900"
          value={wsUrl}
          onChange={(e) => setWsUrl(e.target.value)}
          disabled={streaming}
        />
      </label>
      <div className="relative inline-block max-w-full overflow-hidden rounded-lg border border-zinc-200 bg-black dark:border-zinc-700">
        <video
          ref={videoRef}
          className="block max-h-[min(80vh,720px)] w-full object-cover"
          autoPlay
          playsInline
          muted
        />
        <canvas
          ref={overlayRef}
          className="pointer-events-none absolute left-0 top-0 h-full w-full"
        />
      </div>
      <canvas ref={captureRef} className="hidden" aria-hidden />
      <p className="text-xs text-zinc-500">
        Safari: use Start camera after page load. Tab away pauses capture; WebSocket
        reconnects with backoff. Run the Python server from <code className="font-mono">backend/</code>.
      </p>
    </div>
  );
}
