"use client";

import { AnimatePresence, motion } from "motion/react";
import { useEffect, useId, useRef, useState } from "react";

export type SettingsSelectOption = {
  value: string;
  label: string;
};

type SettingsSelectProps = {
  value: string;
  options: SettingsSelectOption[];
  onChange: (value: string) => void;
  disabled?: boolean;
  /** Accessible label (visually hidden) */
  ariaLabel: string;
  className?: string;
};

/**
 * Custom listbox — no native `<select>` menu chrome.
 */
export function SettingsSelect({
  value,
  options,
  onChange,
  disabled = false,
  ariaLabel,
  className = "",
}: SettingsSelectProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const listId = useId();
  const selected = options.find((o) => o.value === value);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (rootRef.current?.contains(e.target as Node)) return;
      setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  return (
    <div ref={rootRef} className={`relative ${className}`}>
      <button
        type="button"
        disabled={disabled}
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-controls={listId}
        aria-label={ariaLabel}
        onClick={() => !disabled && setOpen((o) => !o)}
        onKeyDown={(e) => {
          if (e.key === "Escape" && open) {
            e.preventDefault();
            e.stopPropagation();
            setOpen(false);
          }
        }}
        className="settings-field flex w-full items-center justify-between gap-2 text-left text-sm disabled:cursor-not-allowed"
      >
        <span className="min-w-0 truncate">
          {selected?.label ?? (value.trim() ? value : "—")}
        </span>
        <motion.span
          aria-hidden
          className="shrink-0 text-white/45"
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
        >
          <svg width="18" height="18" viewBox="0 0 20 20" fill="none" aria-hidden>
            <path
              d="M5.5 7.5L10 12l4.5-4.5"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </motion.span>
      </button>

      <AnimatePresence>
        {open && !disabled ? (
          <motion.ul
            id={listId}
            role="listbox"
            aria-label={ariaLabel}
            initial={{ opacity: 0, y: -8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.99 }}
            transition={{ type: "spring", stiffness: 520, damping: 38, mass: 0.65 }}
            className="absolute left-0 right-0 top-[calc(100%+0.35rem)] z-30 max-h-52 overflow-y-auto overscroll-contain rounded-xl border border-white/[0.12] bg-[rgba(18,22,30,0.97)] py-1 shadow-[0_16px_48px_-12px_rgba(0,0,0,0.55)] backdrop-blur-xl"
            style={{
              backdropFilter: "blur(24px) saturate(1.15)",
              WebkitBackdropFilter: "blur(24px) saturate(1.15)",
            }}
          >
            {options.map((opt) => {
              const active = opt.value === value;
              return (
                <li key={opt.value} role="presentation">
                  <button
                    type="button"
                    role="option"
                    aria-selected={active}
                    className={`flex w-full px-3 py-2.5 text-left text-sm transition-colors ${
                      active
                        ? "bg-[rgba(59,158,255,0.18)] text-white"
                        : "text-white/78 hover:bg-white/[0.08] hover:text-white"
                    }`}
                    onClick={() => {
                      onChange(opt.value);
                      setOpen(false);
                    }}
                  >
                    {opt.label}
                  </button>
                </li>
              );
            })}
          </motion.ul>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
