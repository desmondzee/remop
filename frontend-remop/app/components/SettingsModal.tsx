"use client";

import { useEffect, useId } from "react";
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
}: SettingsModalProps) {
  const titleId = useId();

  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  if (!open || typeof document === "undefined") return null;

  return createPortal(
    <div
      className="fixed inset-0 z-[200] flex items-center justify-center p-4 sm:p-6"
      role="presentation"
    >
      <button
        type="button"
        aria-label="Close settings"
        className="absolute inset-0 bg-black/58 backdrop-blur-2xl transition-opacity"
        onClick={onClose}
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="glass-panel-strong relative z-10 flex max-h-[min(88dvh,640px)] w-full max-w-md flex-col overflow-hidden rounded-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="shrink-0 border-b border-white/10 px-5 py-4 text-center sm:px-6">
          <h2 id={titleId} className="mono-caps text-white/90">
            Settings
          </h2>
          <p className="mt-1.5 font-mono text-[10px] text-white/52 sm:text-[11px]">
            Press ⌘ . or Ctrl . to open or close (no toolbar button) · Esc or backdrop to close
          </p>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-5 py-4 sm:px-6 sm:py-5">
          <div className="min-w-0 overflow-x-auto">
            <p
              className="w-max whitespace-nowrap font-mono text-[11px] leading-tight text-[var(--tw-teal)]"
              title={status}
            >
              {status}
            </p>
          </div>

          <div className="mt-6 min-w-0 border-t border-white/10 pt-6">
            <p className="mono-caps whitespace-nowrap text-white/62">Voice</p>
            <select
              className="glass-input mt-2 w-full px-3 py-2.5 text-sm"
              value={ttsEngine}
              disabled={streaming}
              onChange={(e) => {
                const next = e.target.value as TtsEngine;
                resetAgentTts();
                setTtsEngine(next);
                onTtsEngineChange(next);
              }}
            >
              <option value="browser">Browser (Web Speech)</option>
              <option value="openai">OpenAI (tts-1)</option>
            </select>
            {ttsEngine === "openai" ? (
              <div className="mt-3 space-y-3">
                <div>
                  <p className="mono-caps text-[10px] text-white/52">OpenAI voice</p>
                  <select
                    className="glass-input mt-1.5 w-full px-3 py-2.5 text-sm"
                    value={openaiVoice}
                    disabled={streaming || openaiRotate}
                    onChange={(e) => {
                      const v = e.target.value as OpenaiTtsVoiceId;
                      setOpenaiTtsVoice(v);
                      onOpenaiVoiceChange(v);
                    }}
                  >
                    {OPENAI_TTS_VOICE_IDS.map((id) => (
                      <option key={id} value={id}>
                        {id}
                      </option>
                    ))}
                  </select>
                </div>
                <label className="flex cursor-pointer items-center gap-2 font-mono text-xs text-white/68">
                  <input
                    type="checkbox"
                    className="rounded border-white/20"
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
              <p className="mt-2 font-mono text-[10px] text-white/55">{openaiTtsLabel}</p>
            ) : null}
            <button
              type="button"
              className="glass-btn-ghost mt-3 w-full py-2.5 text-sm font-medium"
              onClick={async () => {
                await resumeAudioContextAfterUserGesture();
                speakInstruction("Testing one, two, three.", { supersede: true });
              }}
            >
              Test TTS
            </button>
          </div>

          <div className="mt-5 flex min-w-0 flex-col gap-2">
            <button
              type="button"
              onClick={onEngageCamera}
              disabled={streaming || !sessionId}
              className="glass-btn-primary shrink-0 whitespace-nowrap py-2.5 text-sm font-medium disabled:opacity-45"
            >
              Engage camera
            </button>
            <button
              type="button"
              onClick={onStopCamera}
              disabled={!streaming}
              className="glass-btn-ghost shrink-0 whitespace-nowrap py-2.5 text-sm font-medium disabled:opacity-35"
            >
              Stop
            </button>
          </div>

          <div className="mt-6 min-w-0">
            <p className="mono-caps whitespace-nowrap text-white/62">Detector</p>
            <select
              className="glass-input mt-2 w-full min-w-0 whitespace-nowrap px-3 py-2.5 text-sm"
              value={detectorPreset}
              onChange={(e) => onDetectorPresetChange(e.target.value as DetectorPresetId)}
            >
              <option value="oiv7">Open Images v7</option>
              <option value="coco">MS COCO</option>
            </select>
          </div>

          <div className="mt-6 min-w-0">
            <div className="flex min-w-0 flex-nowrap items-center justify-between gap-3">
              <p className="mono-caps min-w-0 shrink-0 whitespace-nowrap text-white/62">Camera</p>
              <button
                type="button"
                onClick={() => void onRefreshVideoDevices()}
                disabled={streaming}
                className="glass-btn-ghost shrink-0 whitespace-nowrap rounded-lg px-2 py-1 text-xs font-medium disabled:opacity-35"
              >
                Refresh
              </button>
            </div>
            <select
              className="glass-input mt-2 w-full min-w-0 whitespace-nowrap px-3 py-2.5 text-sm"
              value={selectedDeviceId}
              onChange={(e) => onSelectedDeviceIdChange(e.target.value)}
              disabled={streaming}
            >
              {!devicesLoaded && <option value="">Loading…</option>}
              {devicesLoaded && videoDevices.length === 0 && <option value="">No device</option>}
              {devicesLoaded &&
                videoDevices.map((d, i) => (
                  <option key={d.deviceId || `cam-${i}`} value={d.deviceId}>
                    {d.label || `Camera ${i + 1}`}
                  </option>
                ))}
            </select>
          </div>

          <div className="mt-6 min-w-0 pb-1">
            <p className="mono-caps whitespace-nowrap text-white/62">WebSocket URL</p>
            <input
              className="glass-input mt-2 w-full min-w-0 whitespace-nowrap px-3 py-2.5 font-mono text-[11px] sm:text-xs"
              value={wsUrl}
              onChange={(e) => onWsUrlChange(e.target.value)}
              disabled={streaming}
              spellCheck={false}
              autoComplete="off"
            />
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}
