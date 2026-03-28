"use client";

type TwinStatusBarProps = {
  objects: number;
  wsState: string;
  frameW: number;
  frameH: number;
  inferHz: number;
  streaming: boolean;
  /** Voice agent step loop is polling when true. */
  agentLive: boolean;
  meanConfPct: string | null;
};

/** Label + value; fixed `ch` width on value keeps digits from shifting. */
function Pair({
  label,
  value,
  title,
  valueClassName = "text-white/70",
  minCh,
}: {
  label: string;
  value: string | number;
  title?: string;
  valueClassName?: string;
  minCh: number;
}) {
  return (
    <span className="inline-flex items-baseline gap-1" title={title}>
      <span className="font-mono text-[8px] uppercase tracking-[0.14em] text-white/50">
        {label}
      </span>
      <span
        className={`inline-block text-right font-mono text-[9px] tabular-nums leading-none tracking-tight ${valueClassName}`}
        style={{ minWidth: `${minCh}ch` }}
      >
        {value}
      </span>
    </span>
  );
}

export function TwinStatusBar({
  objects,
  wsState,
  frameW,
  frameH,
  inferHz,
  streaming,
  agentLive,
  meanConfPct,
}: TwinStatusBarProps) {
  const frame =
    frameW > 0 && frameH > 0 ? `${frameW}×${frameH}` : "—";
  const hz = streaming ? `~${inferHz}` : "—";
  const conf = meanConfPct ?? "—";
  const agt = streaming ? (agentLive ? "ON" : "OFF") : "—";

  return (
    <div
      className="pointer-events-none absolute inset-x-0 bottom-0 z-40"
      aria-label="Perception status"
    >
      <div
        className="pointer-events-auto w-full border-t border-[var(--tw-glass-border)] bg-[var(--tw-panel)]"
        style={{
          backdropFilter: "blur(var(--tw-glass-blur)) saturate(1.2)",
          WebkitBackdropFilter: "blur(var(--tw-glass-blur)) saturate(1.2)",
          boxShadow: "0 -12px 40px -12px rgba(0,0,0,0.35)",
          paddingTop: "0.5rem",
          paddingBottom: "max(0.5rem, env(safe-area-inset-bottom))",
          paddingLeft: "max(0.75rem, env(safe-area-inset-left))",
          paddingRight: "max(0.75rem, env(safe-area-inset-right))",
        }}
      >
        <div className="mx-auto flex w-full max-w-none flex-wrap items-baseline justify-center gap-x-5 gap-y-1.5 sm:gap-x-8 sm:gap-y-1 md:justify-between md:px-2">
          <div className="flex flex-wrap items-baseline justify-center gap-x-5 gap-y-1 sm:gap-x-6">
            <Pair
              label="Obj"
              title="Detections in the current frame"
              value={objects}
              valueClassName="text-[var(--tw-teal)]/70"
              minCh={4}
            />
            <Pair
              label="WS"
              title="WebSocket to inference server"
              value={wsState}
              valueClassName="text-[var(--tw-accent)]/65"
              minCh={4}
            />
            <Pair label="Frm" title="Last processed frame size" value={frame} minCh={11} />
            <Pair label="Hz" title="Inference updates per second" value={hz} minCh={5} />
          </div>
          <div className="flex flex-wrap items-baseline justify-center gap-x-5 gap-y-1 sm:gap-x-6">
            <Pair
              label="Agt"
              title="Voice agent step loop (camera can stay on when OFF)"
              value={agt}
              valueClassName={
                agentLive ? "text-[var(--tw-teal)]/72" : "text-white/45"
              }
              minCh={3}
            />
            <Pair label="μ" title="Mean detection confidence" value={conf} minCh={4} />
          </div>
        </div>
      </div>
    </div>
  );
}
