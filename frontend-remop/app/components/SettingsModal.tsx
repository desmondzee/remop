"use client";

import { AnimatePresence, motion } from "motion/react";
import { useEffect, useId, useState } from "react";
import { createPortal } from "react-dom";

import {
  OPENAI_TTS_VOICE_IDS,
  resetAgentTts,
  resumeAudioContextAfterUserGesture,
  setOpenaiTtsRotateVoices,
  setOpenaiTtsVoice,
  setTtsEngine,
  speakInstruction,
  type OpenaiTtsVoiceId,
  type TtsEngine,
} from "../lib/agentTts";
import type { DetectorPresetId } from "./PerceptionPanel";
import { SettingsSelect } from "./SettingsSelect";

export type SettingsModalProps = {
  open: boolean;
  onClose: () => void;
  status: string;
  sessionId: string | null;
  streaming: boolean;
  onEngageCamera: () => void;
  onStopCamera: () => void;
  detectorPreset: DetectorPresetId;
  onDetectorPresetChange: (preset: DetectorPresetId) => void;
  videoDevices: MediaDeviceInfo[];
  selectedDeviceId: string;
  onSelectedDeviceIdChange: (deviceId: string) => void;
  devicesLoaded: boolean;
  onRefreshVideoDevices: () => void;
  wsUrl: string;
  onWsUrlChange: (url: string) => void;
  ttsEngine: TtsEngine;
  onTtsEngineChange: (engine: TtsEngine) => void;
  openaiVoice: OpenaiTtsVoiceId;
  onOpenaiVoiceChange: (voice: OpenaiTtsVoiceId) => void;
  openaiRotate: boolean;
  onOpenaiRotateChange: (rotate: boolean) => void;
  openaiTtsLabel: string;
  agentEnabled: boolean;
  agentToggleDisabled: boolean;
  onAgentEnabledChange: (enabled: boolean) => void;
};

export function SettingsModal({
  open,
  onClose,
  status,
  sessionId,
  streaming,
  onEngageCamera,
  onStopCamera,
  detectorPreset,
  onDetectorPresetChange,
  videoDevices,
  selectedDeviceId,
  onSelectedDeviceIdChange,
  devicesLoaded,
  onRefreshVideoDevices,
  wsUrl,
  onWsUrlChange,
  ttsEngine,
  onTtsEngineChange,
  openaiVoice,
  onOpenaiVoiceChange,
  openaiRotate,
  onOpenaiRotateChange,
  openaiTtsLabel,
  agentEnabled,
  agentToggleDisabled,
  onAgentEnabledChange,
}: SettingsModalProps) {
  const titleId = useId();
  const [mounted, setMounted] = useState(false);

  const ttsEngineOptions = [
    { value: "browser" as const, label: "Browser (Web Speech)" },
    { value: "openai" as const, label: "OpenAI (tts-1)" },
  ];

  const openaiVoiceOptions = OPENAI_TTS_VOICE_IDS.map((id) => ({
    value: id,
    label: id,
  }));

  const detectorOptions = [
    { value: "oiv7" as const, label: "Open Images v7" },
    { value: "coco" as const, label: "MS COCO" },
  ];

  const cameraOptions =
    !devicesLoaded
      ? [{ value: "", label: "Loading…" }]
      : videoDevices.length === 0
        ? [{ value: "", label: "No device" }]
        : videoDevices.map((d, i) => ({
            value: d.deviceId,
            label: d.label || `Camera ${i + 1}`,
          }));

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!mounted || typeof document === "undefined") return null;

  return createPortal(
    <AnimatePresence>
      {open ? (
        <motion.div
          key="settings-root"
          className="fixed inset-0 z-[200] flex items-center justify-center p-4 sm:p-6"
          role="presentation"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.26, ease: [0.22, 1, 0.36, 1] }}
        >
          <button
            type="button"
            aria-label="Close settings"
            className="absolute inset-0 bg-[rgba(4,6,10,0.22)] backdrop-blur-md backdrop-saturate-150"
            onClick={onClose}
          />
          <motion.div
            role="dialog"
            aria-modal="true"
            aria-labelledby={titleId}
            className="glass-panel relative z-10 flex max-h-[min(88dvh,640px)] w-full max-w-md flex-col overflow-hidden rounded-3xl shadow-[0_24px_80px_-20px_rgba(0,0,0,0.45)]"
            style={{ willChange: "transform" }}
            initial={{ opacity: 0, scale: 0.94, y: 22 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.97, y: 14 }}
            transition={{ type: "spring", stiffness: 400, damping: 34 }}
            onClick={(e) => e.stopPropagation()}
          >
            <header className="relative shrink-0 px-5 pb-4 pt-5 sm:px-6 sm:pt-6">
              <button
                type="button"
                onClick={onClose}
                className="settings-btn-quiet absolute right-3 top-3 flex h-9 w-9 items-center justify-center rounded-full p-0 sm:right-4 sm:top-4"
                aria-label="Close"
              >
                <span className="text-lg leading-none text-white/70" aria-hidden>
                  ×
                </span>
              </button>
              <h2 id={titleId} className="mono-caps pr-10 text-left text-white/88">
                Settings
              </h2>
              <p className="mt-2 max-w-[95%] text-left font-mono text-[10px] leading-relaxed text-white/48 sm:text-[11px]">
                <kbd className="rounded border border-white/15 bg-white/[0.06] px-1 py-px font-mono text-[9px] text-white/55">
                  ⌘ .
                </kbd>{" "}
                or{" "}
                <kbd className="rounded border border-white/15 bg-white/[0.06] px-1 py-px font-mono text-[9px] text-white/55">
                  Ctrl .
                </kbd>{" "}
                · Esc · outside tap ·{" "}
                <span className="text-white/58">A</span> toggles agent when live
              </p>
            </header>

            <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-5 pb-6 sm:px-6">
              <div className="rounded-2xl border border-white/[0.06] bg-white/[0.03] px-3.5 py-3 sm:px-4 sm:py-3.5">
                <p className="mono-caps text-[0.6rem] text-white/45">Status</p>
                <p
                  className="mt-1.5 truncate font-mono text-[11px] text-[var(--tw-teal)]/90 sm:text-xs"
                  title={status}
                >
                  {status}
                </p>
              </div>

              <section className="mt-4 space-y-4 rounded-2xl border border-white/[0.06] bg-white/[0.03] p-3.5 sm:p-4">
                <p className="mono-caps text-white/55">Voice</p>
                <SettingsSelect
                  ariaLabel="Voice engine"
                  value={ttsEngine}
                  disabled={streaming}
                  options={ttsEngineOptions}
                  onChange={(v) => {
                    const next = v as TtsEngine;
                    resetAgentTts();
                    setTtsEngine(next);
                    onTtsEngineChange(next);
                  }}
                />
                {ttsEngine === "openai" ? (
                  <div className="space-y-3">
                    <div>
                      <p className="mono-caps text-[0.6rem] text-white/42">OpenAI voice</p>
                      <SettingsSelect
                        className="mt-2"
                        ariaLabel="OpenAI TTS voice"
                        value={openaiVoice}
                        disabled={streaming || openaiRotate}
                        options={openaiVoiceOptions}
                        onChange={(v) => {
                          const voice = v as OpenaiTtsVoiceId;
                          setOpenaiTtsVoice(voice);
                          onOpenaiVoiceChange(voice);
                        }}
                      />
                    </div>
                    <label className="flex cursor-pointer items-center gap-2.5 rounded-xl border border-white/[0.06] bg-black/20 px-3 py-2.5 font-mono text-xs text-white/68">
                      <input
                        type="checkbox"
                        className="h-3.5 w-3.5 rounded border-white/25 bg-transparent text-[var(--tw-accent)] focus:ring-[var(--tw-accent)]/40"
                        checked={openaiRotate}
                        disabled={streaming}
                        onChange={(e) => {
                          const on = e.target.checked;
                          setOpenaiTtsRotateVoices(on);
                          onOpenaiRotateChange(on);
                        }}
                      />
                      Rotate voices
                    </label>
                  </div>
                ) : null}
                {openaiTtsLabel ? (
                  <p className="font-mono text-[10px] text-white/48">{openaiTtsLabel}</p>
                ) : null}
                <button
                  type="button"
                  className="settings-btn-quiet w-full py-2.5 text-sm font-medium"
                  onClick={async () => {
                    await resumeAudioContextAfterUserGesture();
                    speakInstruction("Testing one, two, three.", { supersede: true });
                  }}
                >
                  Test TTS
                </button>
              </section>

              <section className="mt-4 space-y-3 rounded-2xl border border-white/[0.06] bg-white/[0.03] p-3.5 sm:p-4">
                <p className="mono-caps text-white/55">Camera &amp; agent</p>
                <div className="flex flex-col gap-2.5">
                  <button
                    type="button"
                    onClick={onEngageCamera}
                    disabled={streaming || !sessionId}
                    className="settings-btn-primary w-full py-2.5 text-sm font-medium disabled:pointer-events-none"
                  >
                    Engage camera
                  </button>
                  <button
                    type="button"
                    onClick={onStopCamera}
                    disabled={!streaming}
                    className="settings-btn-quiet w-full py-2.5 text-sm font-medium disabled:pointer-events-none"
                  >
                    Stop camera
                  </button>
                </div>
                <label className="flex cursor-pointer items-start gap-3 rounded-xl border border-white/[0.06] bg-black/20 px-3 py-3 font-mono text-xs leading-snug text-white/72">
                  <input
                    type="checkbox"
                    className="mt-0.5 h-3.5 w-3.5 shrink-0 rounded border-white/25 bg-transparent text-[var(--tw-accent)] focus:ring-[var(--tw-accent)]/40"
                    checked={agentEnabled}
                    disabled={agentToggleDisabled}
                    onChange={(e) => onAgentEnabledChange(e.target.checked)}
                  />
                  <span>
                    Voice agent
                    <span className="mt-0.5 block font-mono text-[10px] font-normal text-white/42">
                      Perception can stay on without the step loop.
                    </span>
                  </span>
                </label>
              </section>

              <section className="mt-4 space-y-4 rounded-2xl border border-white/[0.06] bg-white/[0.03] p-3.5 sm:p-4">
                <p className="mono-caps text-white/55">Detector</p>
                <SettingsSelect
                  ariaLabel="Object detector preset"
                  value={detectorPreset}
                  options={detectorOptions}
                  onChange={(v) => onDetectorPresetChange(v as DetectorPresetId)}
                />
              </section>

              <section className="mt-4 space-y-4 rounded-2xl border border-white/[0.06] bg-white/[0.03] p-3.5 sm:p-4">
                <div className="flex min-w-0 items-center justify-between gap-3">
                  <p className="mono-caps shrink-0 text-white/55">Camera</p>
                  <button
                    type="button"
                    onClick={() => void onRefreshVideoDevices()}
                    disabled={streaming}
                    className="settings-btn-quiet shrink-0 rounded-lg px-3 py-1.5 text-xs font-medium disabled:pointer-events-none"
                  >
                    Refresh
                  </button>
                </div>
                <SettingsSelect
                  ariaLabel="Camera device"
                  value={selectedDeviceId}
                  disabled={streaming}
                  options={cameraOptions}
                  onChange={onSelectedDeviceIdChange}
                />
              </section>

              <section className="mt-4 space-y-4 rounded-2xl border border-white/[0.06] bg-white/[0.03] p-3.5 sm:p-4 pb-5">
                <p className="mono-caps text-white/55">WebSocket URL</p>
                <input
                  className="settings-field font-mono text-[11px] sm:text-xs"
                  value={wsUrl}
                  onChange={(e) => onWsUrlChange(e.target.value)}
                  disabled={streaming}
                  spellCheck={false}
                  autoComplete="off"
                />
              </section>
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>,
    document.body
  );
}
