/** Detection box (0–1 coords) + fields needed for HUD / perception focus. */
export type AnchorMatchableDetection = {
  label: string;
  conf: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  rel_depth: number;
  cx: number;
};

/**
 * Pick the best detection for an agent task anchor string (fuzzy label match).
 */
export function matchTaskAnchorToDetection(
  detections: AnchorMatchableDetection[],
  anchor: string
): AnchorMatchableDetection | null {
  const a = anchor.trim().toLowerCase();
  if (!a) return null;
  let best: AnchorMatchableDetection | null = null;
  let bestScore = -1;
  for (const d of detections) {
    const l = d.label.trim().toLowerCase();
    if (!l) continue;
    let score = -1;
    if (l === a) score = 1 + d.conf;
    else if (l.includes(a)) score = 0.88 * d.conf;
    else if (a.includes(l)) score = 0.78 * d.conf;
    else {
      const words = a.split(/\s+/).filter((w) => w.length > 2);
      const hits = words.filter((w) => l.includes(w)).length;
      if (hits > 0) score = (0.45 + 0.12 * hits) * d.conf;
    }
    if (score > bestScore) {
      bestScore = score;
      best = d;
    }
  }
  return bestScore > 0 ? best : null;
}
