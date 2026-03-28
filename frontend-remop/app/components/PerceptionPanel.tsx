"use client";

import { AnimatePresence, LayoutGroup, motion } from "motion/react";
import { useState } from "react";

export type PerceptionTrack = {
  label: string;
  conf: number;
  rel_depth: number;
  cx: number;
};

export type SessionTaskLogEntry = {
  id: string;
  /** Short line, e.g. tool names from one agent step */
  summary: string;
};

export type DetectorPresetId = "oiv7" | "coco";

export type PerceptionPanelView = "perception" | "depth";

export type PerceptionPanelProps = {
  panelView: PerceptionPanelView;
  onPanelViewChange: (view: PerceptionPanelView) => void;
  /** Resolved image URL for depth (HTTP preferred, else data URL from WebSocket) */
  depthImgSrc: string | null;
  /** Primary object: task-anchor match when available, else nearest by depth */
  focusTarget: PerceptionTrack | null;
  /** Whether `focusTarget` is tied to the current task anchor string */
  focusFromAnchor: boolean;
  /** Newest-first agent tool steps this session (minimal) */
  sessionTaskLog: SessionTaskLogEntry[];
  depthToAccent: (relDepth: number) => string;
  /** Server JPEG encode failure (truncated), when preview enabled but image missing */
  depthPreviewError: string | null;
  streaming: boolean;
};

export function PerceptionPanel({
  panelView,
  onPanelViewChange,
  depthImgSrc,
  focusTarget,
  focusFromAnchor,
  sessionTaskLog,
  depthToAccent,
  depthPreviewError,
  streaming,
}: PerceptionPanelProps) {
  const [logOpen, setLogOpen] = useState(false);

  const segBtn = (id: PerceptionPanelView, label: string) => (
    <button
      key={id}
      type="button"
      role="tab"
      aria-selected={panelView === id}
      onClick={() => onPanelViewChange(id)}
      className={`min-w-0 flex-1 whitespace-nowrap rounded-xl px-2.5 py-2.5 font-mono text-[10px] uppercase tracking-[0.1em] transition-colors sm:px-3.5 sm:text-[11px] sm:tracking-[0.12em] ${
        panelView === id
          ? "bg-white/[0.12] text-white/90 shadow-[0_1px_0_rgba(255,255,255,0.06)_inset]"
          : "text-white/55 hover:bg-white/[0.08] hover:text-white/72"
      }`}
    >
      {label}
    </button>
  );

  return (
    <div className="glass-panel flex min-h-0 max-h-full flex-1 flex-col overflow-hidden !rounded-3xl">
      <header className="shrink-0 border-b border-white/[0.08] px-5 py-4 sm:px-6 sm:py-5">
        <h2 className="mono-caps text-white/88">Perception</h2>
      </header>

      <div className="shrink-0 border-b border-white/[0.06] px-4 py-3.5 sm:px-5 sm:py-4">
        <div
          className="flex gap-1 rounded-2xl border border-white/[0.08] bg-black/35 p-1"
          role="tablist"
          aria-label="Panel view"
        >
          {segBtn("perception", "Objects")}
          {segBtn("depth", "MiDaS depth")}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
        {panelView === "depth" ? (
          <section className="px-5 py-5 sm:px-6 sm:py-6" role="tabpanel" aria-label="MiDaS depth">
            {!streaming ? (
              <p className="text-center font-mono text-xs leading-relaxed text-white/55">
                Engage the camera to stream MiDaS depth.
              </p>
            ) : depthPreviewError ? (
              <div className="space-y-2 text-center font-mono text-xs leading-relaxed text-white/55">
                <p>Depth preview failed on the server.</p>
                <p className="break-words rounded-xl bg-white/[0.08] px-3 py-2.5 text-left text-[11px] text-white/70">
                  {depthPreviewError}
                </p>
              </div>
            ) : depthImgSrc ? (
              <div className="space-y-3 text-center font-mono text-xs leading-relaxed text-white/62">
                <p>
                  The live view shows MiDaS depth (Inferno colormap, per-frame normalized),
                  streamed with each inference. Detection overlays stay on top and use the
                  same frame geometry.
                </p>
                <p className="mono-caps text-[10px] text-white/48 sm:text-[11px]">
                  False-color depth · per frame · same WebSocket as detections
                </p>
              </div>
            ) : (
              <p className="text-center font-mono text-xs leading-relaxed text-white/55">
                Depth preview not in server response. Set{" "}
                <code className="rounded-lg bg-white/12 px-1.5 py-0.5 text-[10px] text-white/72">
                  INCLUDE_DEPTH_PREVIEW=1
                </code>{" "}
                on the inference server (default on).
              </p>
            )}
          </section>
        ) : (
          <div role="tabpanel" aria-label="Objects and session log">
            <section className="border-b border-white/[0.06] px-5 py-5 sm:px-6 sm:py-6">
              <div className="flex items-baseline justify-between gap-2">
                <h3 className="mono-caps text-white/72">Focus</h3>
                <span className="font-mono text-[9px] uppercase tracking-[0.12em] text-white/38">
                  {focusFromAnchor ? "Anchor" : "Depth"}
                </span>
              </div>
              <p className="mt-1 font-mono text-[10px] leading-snug text-white/42">
                {focusFromAnchor
                  ? "Object matching the agent task anchor."
                  : "Closest object by estimated depth when no anchor match."}
              </p>
              {focusTarget ? (
                <div className="mt-4">
                  <p className="text-lg font-semibold capitalize leading-tight tracking-tight text-white">
                    {focusTarget.label}
                  </p>
                  <div className="mt-3.5 h-1 overflow-hidden rounded-full bg-white/10">
                    <div
                      className="h-full rounded-full transition-all duration-150"
                      style={{
                        width: `${Math.min(100, Math.round(focusTarget.conf * 100))}%`,
                        background: `linear-gradient(90deg, ${depthToAccent(focusTarget.rel_depth)}, var(--tw-accent))`,
                        boxShadow: `0 0 10px color-mix(in srgb, ${depthToAccent(focusTarget.rel_depth)} 32%, transparent)`,
                      }}
                    />
                  </div>
                </div>
              ) : (
                <p className="mt-4 font-mono text-sm text-white/50">—</p>
              )}
            </section>

            <LayoutGroup id="perception-session">
              <div
                id="session-log-region"
                className="border-b border-white/[0.06] px-5 pb-5 pt-3 sm:px-6 sm:pb-6 sm:pt-4"
              >
                <button
                  type="button"
                  onClick={() => setLogOpen((o) => !o)}
                  aria-expanded={logOpen}
                  aria-controls="session-log-region"
                  className="flex w-full flex-nowrap items-center justify-between gap-3 whitespace-nowrap rounded-xl border-0 bg-transparent py-2.5 text-left [-webkit-tap-highlight-color:transparent] focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-[var(--tw-accent)]/35 sm:py-3"
                >
                  <span className="flex min-w-0 items-baseline gap-2">
                    <span className="mono-caps shrink-0 text-white/65">Session</span>
                    {sessionTaskLog.length > 0 ? (
                      <span className="font-mono text-[10px] tabular-nums text-white/45 sm:text-[11px]">
                        {sessionTaskLog.length}
                      </span>
                    ) : null}
                  </span>
                  <motion.span
                    className="shrink-0 font-mono text-[11px] leading-none text-[var(--tw-accent)]/70"
                    animate={{ rotate: logOpen ? 180 : 0 }}
                    transition={{ type: "spring", stiffness: 400, damping: 28 }}
                    aria-hidden
                  >
                    ▾
                  </motion.span>
                </button>

                <AnimatePresence mode="wait" initial={false}>
                  {!logOpen ? (
                    <motion.div
                      key="session-collapsed"
                      role="region"
                      aria-label="Latest session tool"
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -4 }}
                      transition={{ type: "spring", stiffness: 340, damping: 32 }}
                      className="relative mt-3 min-h-[2.75rem] overflow-hidden"
                    >
                      <AnimatePresence mode="popLayout" initial={false}>
                        {sessionTaskLog[0] ? (
                          <motion.div
                            key={sessionTaskLog[0].id}
                            layout
                            initial={{ opacity: 0, y: 12, scale: 0.98 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, y: -14, scale: 0.98 }}
                            transition={{
                              type: "spring",
                              stiffness: 400,
                              damping: 32,
                              mass: 0.82,
                            }}
                            className="rounded-xl border border-white/[0.08] bg-black/28 px-3 py-2.5 font-mono text-[11px] leading-snug text-white/78 sm:text-[12px]"
                          >
                            {sessionTaskLog[0].summary}
                          </motion.div>
                        ) : (
                          <motion.p
                            key="session-empty"
                            layout
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="font-mono text-xs text-white/45"
                          >
                            No agent tools yet this session.
                          </motion.p>
                        )}
                      </AnimatePresence>
                    </motion.div>
                  ) : (
                    <motion.div
                      key="session-expanded"
                      role="region"
                      aria-label="Session tool history"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 8 }}
                      transition={{ type: "spring", stiffness: 320, damping: 30 }}
                      className="mt-2 border-t border-white/[0.06] pt-4"
                    >
                      {sessionTaskLog.length === 0 ? (
                        <p className="font-mono text-xs text-white/48">No agent tools yet this session.</p>
                      ) : (
                        <div
                          role="list"
                          className="flex flex-col gap-2 font-mono text-[11px] leading-snug text-white/68 sm:text-[12px]"
                        >
                          <AnimatePresence initial={false}>
                            {sessionTaskLog.map((e, i) => (
                              <motion.div
                                key={e.id}
                                role="listitem"
                                layout
                                initial={{ opacity: 0, y: -10, scale: 0.98 }}
                                animate={{ opacity: 1, y: 0, scale: 1 }}
                                exit={{ opacity: 0, y: -6 }}
                                transition={{
                                  type: "spring",
                                  stiffness: 420,
                                  damping: 32,
                                  delay: i * 0.035,
                                }}
                                className="rounded-xl border border-white/[0.06] bg-black/25 px-3 py-2 text-white/75"
                              >
                                {e.summary}
                              </motion.div>
                            ))}
                          </AnimatePresence>
                        </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </LayoutGroup>
          </div>
        )}
      </div>
    </div>
  );
}
