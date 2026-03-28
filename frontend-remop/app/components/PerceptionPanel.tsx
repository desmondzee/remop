"use client";

import { useState } from "react";

export type PerceptionTrack = {
  label: string;
  conf: number;
  rel_depth: number;
  cx: number;
};

export type DetectorPresetId = "oiv7" | "coco";

export type PerceptionPanelView = "perception" | "depth";

export type PerceptionPanelProps = {
  closest: PerceptionTrack | null;
  tracks: PerceptionTrack[];
  depthToAccent: (relDepth: number) => string;
  /** Direct JPEG URL on inference server (/v1/depth_preview); preferred over data URL */
  depthHttpSrc: string | null;
  /** data:image/jpeg;base64,... from WebSocket when present */
  depthPreviewUrl: string | null;
  /** Server JPEG encode failure (truncated), when preview enabled but image missing */
  depthPreviewError: string | null;
  streaming: boolean;
};

export function PerceptionPanel({
  closest,
  tracks,
  depthToAccent,
  depthHttpSrc,
  depthPreviewUrl,
  depthPreviewError,
  streaming,
}: PerceptionPanelProps) {
  const [panelView, setPanelView] = useState<PerceptionPanelView>("perception");
  const [tracksOpen, setTracksOpen] = useState(false);
  /** Last direct HTTP URL that failed to load; new `depthHttpSrc` auto-retries. */
  const [failedHttpSrc, setFailedHttpSrc] = useState<string | null>(null);

  const depthImgSrc =
    depthHttpSrc && depthHttpSrc !== failedHttpSrc
      ? depthHttpSrc
      : depthPreviewUrl;

  const segBtn = (id: PerceptionPanelView, label: string) => (
    <button
      key={id}
      type="button"
      role="tab"
      aria-selected={panelView === id}
      onClick={() => setPanelView(id)}
      className={`min-w-0 flex-1 whitespace-nowrap rounded-md px-2 py-2 font-mono text-[10px] uppercase tracking-[0.1em] transition-colors sm:px-3 sm:text-[11px] sm:tracking-[0.12em] ${
        panelView === id
          ? "bg-white/[0.12] text-white/90 shadow-[0_1px_0_rgba(255,255,255,0.06)_inset]"
          : "text-white/38 hover:bg-white/[0.05] hover:text-white/55"
      }`}
    >
      {label}
    </button>
  );

  return (
    <div className="glass-panel flex min-h-0 max-h-full flex-1 flex-col overflow-hidden rounded-2xl">
      <header className="shrink-0 border-b border-white/10 px-4 py-2.5 sm:px-5 sm:py-3">
        <h2 className="mono-caps text-white/85">Perception</h2>
      </header>

      <div className="shrink-0 border-b border-white/8 px-3 py-2.5 sm:px-4 sm:py-3">
        <div
          className="flex gap-0.5 rounded-xl border border-white/[0.07] bg-black/25 p-0.5"
          role="tablist"
          aria-label="Panel view"
        >
          {segBtn("perception", "Objects")}
          {segBtn("depth", "MiDaS depth")}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
        {panelView === "depth" ? (
          <section className="px-4 py-4 sm:px-5" role="tabpanel" aria-label="MiDaS depth">
            {!streaming ? (
              <p className="text-center font-mono text-xs leading-relaxed text-white/35">
                Engage the camera to stream MiDaS depth.
              </p>
            ) : depthImgSrc ? (
              <>
                <div className="overflow-hidden rounded-xl border border-white/[0.06] bg-black/30">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={depthImgSrc}
                    alt="MiDaS relative depth (linear grayscale)"
                    className="h-auto w-full object-cover"
                    decoding="async"
                    onError={() => {
                      if (depthHttpSrc) setFailedHttpSrc(depthHttpSrc);
                    }}
                  />
                </div>
                <p className="mono-caps mt-3 text-center text-[10px] text-white/32 sm:text-[11px]">
                  Raw depth · linear grayscale · per frame
                </p>
              </>
            ) : depthPreviewError ? (
              <div className="space-y-2 text-center font-mono text-xs leading-relaxed text-white/35">
                <p>Depth preview failed on the server.</p>
                <p className="break-words rounded-lg bg-white/[0.06] px-2 py-2 text-left text-[11px] text-white/50">
                  {depthPreviewError}
                </p>
              </div>
            ) : (
              <p className="text-center font-mono text-xs leading-relaxed text-white/35">
                Depth preview not in server response. Set{" "}
                <code className="rounded bg-white/10 px-1 py-0.5 text-[10px] text-white/55">
                  INCLUDE_DEPTH_PREVIEW=1
                </code>{" "}
                on the inference server (default on).
              </p>
            )}
          </section>
        ) : (
          <div role="tabpanel" aria-label="Objects and tracks">
            <section className="border-b border-white/8 px-4 py-4 sm:px-5">
              <h3
                className="mono-caps text-white/50"
                title="Detection ranked closest in depth (MiDaS-style relative z); bar shows model confidence."
              >
                Nearest
              </h3>
              {closest ? (
                <div className="mt-3">
                  <p className="text-lg font-semibold capitalize leading-tight tracking-tight text-white">
                    {closest.label}
                  </p>
                  <div className="mt-3 h-1 overflow-hidden rounded-full bg-white/10">
                    <div
                      className="h-full rounded-full transition-all duration-150"
                      style={{
                        width: `${Math.min(100, (closest.conf * 100).toFixed(0))}%`,
                        background: `linear-gradient(90deg, ${depthToAccent(closest.rel_depth)}, var(--tw-accent))`,
                        boxShadow: `0 0 10px color-mix(in srgb, ${depthToAccent(closest.rel_depth)} 32%, transparent)`,
                      }}
                    />
                  </div>
                </div>
              ) : (
                <p className="mt-3 font-mono text-sm text-white/30">—</p>
              )}
            </section>

            <div className="border-b border-white/8 px-4 pb-4 pt-2 sm:px-5 sm:pb-5 sm:pt-3">
              <button
                type="button"
                onClick={() => setTracksOpen((o) => !o)}
                aria-expanded={tracksOpen}
                aria-controls="tracks-panel-region"
                title="Up to eight strongest detections by confidence; color reflects estimated depth."
                className="flex w-full flex-nowrap items-center justify-between gap-3 whitespace-nowrap rounded-lg border-0 bg-transparent py-2.5 text-left [-webkit-tap-highlight-color:transparent] focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-[var(--tw-accent)]/35 sm:py-3"
              >
                <span className="flex min-w-0 items-baseline gap-2">
                  <span className="mono-caps shrink-0 text-white/50">Tracks</span>
                  {tracks.length > 0 ? (
                    <span className="font-mono text-[10px] tabular-nums text-white/30 sm:text-[11px]">
                      {tracks.length}
                    </span>
                  ) : null}
                </span>
                <span
                  className={`shrink-0 font-mono text-[11px] leading-none text-[var(--tw-accent)]/70 transition-transform duration-200 ${tracksOpen ? "rotate-180" : ""}`}
                  aria-hidden
                >
                  ▾
                </span>
              </button>

              {tracksOpen ? (
                <div id="tracks-panel-region" className="mt-1 border-t border-white/[0.06] pt-3">
                  {tracks.length === 0 ? (
                    <p className="font-mono text-sm text-white/30">—</p>
                  ) : (
                    <ul className="divide-y divide-white/[0.06] font-mono text-[13px]">
                      {tracks.map((d, i) => (
                        <li
                          key={`${d.label}-${i}-${d.cx.toFixed(3)}`}
                          className="flex items-center justify-between gap-3 py-2.5 first:pt-0"
                        >
                          <span className="min-w-0 truncate capitalize text-white/85">{d.label}</span>
                          <span
                            className="shrink-0 tabular-nums"
                            style={{ color: depthToAccent(d.rel_depth) }}
                          >
                            {(d.conf * 100).toFixed(0)}%
                          </span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              ) : null}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
