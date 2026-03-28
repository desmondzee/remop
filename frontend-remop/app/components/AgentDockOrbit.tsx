"use client";

import { useId } from "react";

/**
 * Orbiting “comet” on the agent card border (~1s sweep, slow tail fade, pause, loop).
 * SVG avoids fragile CSS mask-composite across WebKit/Firefox.
 */
export function AgentDockOrbit() {
  const gid = useId().replace(/:/g, "");

  return (
    <svg
      className="agent-dock-orbit-svg pointer-events-none absolute inset-0 z-[60] h-full w-full overflow-visible rounded-[1.25rem]"
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      aria-hidden
    >
      <defs>
        <linearGradient
          id={`agent-orbit-grad-${gid}`}
          gradientUnits="userSpaceOnUse"
          x1="0"
          y1="0"
          x2="100"
          y2="0"
        >
          <stop offset="0%" stopColor="rgba(180, 200, 255, 0)" />
          <stop offset="18%" stopColor="rgba(165, 198, 255, 0.08)" />
          <stop offset="38%" stopColor="rgba(145, 192, 255, 0.28)" />
          <stop offset="58%" stopColor="rgba(135, 198, 255, 0.55)" />
          <stop offset="76%" stopColor="rgba(128, 200, 255, 0.88)" />
          <stop offset="86%" stopColor="rgba(255, 200, 240, 0.72)" />
          <stop offset="96%" stopColor="rgba(200, 185, 255, 0.22)" />
          <stop offset="100%" stopColor="rgba(190, 200, 255, 0)" />
        </linearGradient>
        <filter id={`agent-orbit-glow-${gid}`} x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="0.85" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <rect
        className="agent-dock-orbit-stroke"
        x="1.1"
        y="1.1"
        width="97.8"
        height="97.8"
        rx="5.2"
        ry="5.2"
        fill="none"
        stroke={`url(#agent-orbit-grad-${gid})`}
        strokeWidth="1.02"
        strokeLinecap="round"
        pathLength={100}
        strokeDasharray="16 84"
        filter={`url(#agent-orbit-glow-${gid})`}
        vectorEffect="nonScalingStroke"
      />
    </svg>
  );
}
