"use client";

type TwinStatusBarProps = {
  objects: number;
  wsState: string;
  frameW: number;
  frameH: number;
  inferHz: number;
  streaming: boolean;
  meanConfPct: string | null;
};

/** Label + value; fixed `ch` width on value keeps digits from shifting. */
function Pair({
  label,
  value,
  title,
  valueClassName = "text-white/45",
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
      <span className="font-mono text-[8px] uppercase tracking-[0.14em] text-white/28">
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
  meanConfPct,
}: TwinStatusBarProps) {
  const frame =
    frameW > 0 && frameH > 0 ? `${frameW}×${frameH}` : "—";
  const hz = streaming ? `~${inferHz}` : "—";
  const conf = meanConfPct ?? "—";

  return (
    <div
      className="pointer-events-none absolute inset-x-0 bottom-0 z-40 flex justify-start pb-[max(0.15rem,env(safe-area-inset-bottom))] pl-[max(0.4rem,env(safe-area-inset-left))] pr-3 pt-0"
      aria-label="Perception status"
    >
      <div
        className="pointer-events-auto rounded-md border border-white/[0.04] bg-black/16 px-1.5 py-px backdrop-blur-[8px] sm:px-2 sm:py-0.5"
        style={{
          WebkitBackdropFilter: "blur(8px)",
          boxShadow: `0 24px 48px -16px rgba(0,0,0,0.05)`,
        }}
      >
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 sm:gap-x-3.5">
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
          <Pair label="μ" title="Mean detection confidence" value={conf} minCh={4} />
        </div>
      </div>
    </div>
  );
}
