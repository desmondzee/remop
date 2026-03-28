"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";

const DEFAULT_CPS = 52;

type AgentRevealTextProps = {
  text: string;
  className?: string;
  /** Approximate characters revealed per second (ignored when reduced motion). */
  charPerSec?: number;
};

/**
 * Reveals text progressively so full agent responses feel streamed even when
 * the API returns the string in one chunk.
 */
export function AgentRevealText({
  text,
  className,
  charPerSec = DEFAULT_CPS,
}: AgentRevealTextProps) {
  const [n, setN] = useState(0);
  const targetRef = useRef("");
  const nRef = useRef(0);
  const reduceRef = useRef(false);

  useEffect(() => {
    reduceRef.current = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }, []);

  useEffect(() => {
    nRef.current = n;
  }, [n]);

  useEffect(() => {
    const t = text ?? "";
    const prev = targetRef.current;
    if (t === prev) return;
    if (!t) {
      targetRef.current = "";
      nRef.current = 0;
      setN(0);
      return;
    }
    if (reduceRef.current) {
      targetRef.current = t;
      nRef.current = t.length;
      setN(t.length);
      return;
    }
    if (!prev || !t.startsWith(prev)) {
      targetRef.current = t;
      nRef.current = 0;
      setN(0);
    } else {
      targetRef.current = t;
    }
  }, [text]);

  useEffect(() => {
    if (reduceRef.current) return;
    let id: number;
    let last = performance.now();

    const loop = (now: number) => {
      const len = targetRef.current.length;
      let cur = nRef.current;
      if (cur < len) {
        const dt = Math.min(0.12, (now - last) / 1000);
        last = now;
        cur = Math.min(len, cur + Math.max(1, Math.round(charPerSec * dt)));
        nRef.current = cur;
        setN(cur);
      }
      id = requestAnimationFrame(loop);
    };

    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [charPerSec]);

  const cap = text.length;
  const visibleLen = Math.min(n, cap);
  const visible = text.slice(0, visibleLen);
  const catchingUp = visibleLen < cap;

  return (
    <span className={className}>
      {visible || "—"}
      {catchingUp && visible ? (
        <motion.span
          aria-hidden
          className="ml-px inline-block h-[0.85em] w-px translate-y-px bg-white/55 align-middle"
          animate={{ opacity: [0.2, 1, 0.2] }}
          transition={{ duration: 0.9, repeat: Infinity, ease: "easeInOut" }}
        />
      ) : null}
    </span>
  );
}
