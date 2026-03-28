"use client";

export type TwinBrandHudProps = {
  /** Product / brand wordmark */
  brand?: string;
  /** Weekday + calendar date (composed as one “date” segment) */
  weekday: string;
  dateLine: string;
  time: string;
  location: string;
  className?: string;
};

/**
 * Top-left context pill: brand · location · date · time (glass HUD theme).
 */
export function TwinBrandHud({
  brand = "remop",
  weekday,
  dateLine,
  time,
  location,
  className = "",
}: TwinBrandHudProps) {
  const dateSegment = `${weekday} · ${dateLine}`;
  const fullTitle = `${brand} · ${location} · ${dateSegment} · ${time}`;

  return (
    <nav
      aria-label="Brand and local context"
      className={`pointer-events-none absolute left-0 top-0 z-[25] max-w-[min(calc(100vw-1.5rem),28rem)] pl-[max(0.65rem,env(safe-area-inset-left))] pt-[max(0.45rem,env(safe-area-inset-top))] ${className}`}
    >
      <div
        className="inline-flex max-w-full items-center gap-2.5 rounded-full border border-[var(--tw-glass-border)] bg-[var(--tw-panel)] px-3.5 py-2 backdrop-blur-xl sm:gap-3 sm:px-4 sm:py-2.5"
        style={{
          WebkitBackdropFilter: "blur(20px) saturate(1.35)",
          boxShadow: `var(--tw-glass-elev-shadow)`,
        }}
      >
        <span className="shrink-0 font-mono text-[11px] font-semibold tracking-[0.14em] text-white/95 sm:text-xs">
          {brand}
        </span>
        <span
          className="hidden h-3.5 w-px shrink-0 bg-white/[0.14] sm:block"
          aria-hidden
        />
        <p
          className="min-w-0 truncate font-mono text-[9px] leading-snug sm:text-[10px]"
          title={fullTitle}
        >
          <span className="max-w-[9rem] truncate text-white/52 sm:max-w-[12rem] sm:text-white/55">
            {location}
          </span>
          <span className="text-white/25"> · </span>
          <span className="tabular-nums text-white/58">{dateSegment}</span>
          <span className="text-white/25"> · </span>
          <span className="tabular-nums text-[var(--tw-accent)]/88">{time}</span>
        </p>
      </div>
    </nav>
  );
}
