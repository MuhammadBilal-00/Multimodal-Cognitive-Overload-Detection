import type { Landmark } from './features';

export interface FaceStats { index: number; area: number; cx: number; cy: number }

// A challenger face must be this near (normalized units) to the previous
// primary's centroid to count as "the same person still in frame".
export const STICKINESS_RADIUS = 0.15;
// A non-incumbent face must be this many times larger to take over the
// primary slot — defeats MediaPipe's frame-to-frame index reordering and
// area jitter between similar-size faces.
export const HYSTERESIS = 1.3;

export function faceStats(faces: Landmark[][]): FaceStats[] {
  return faces.map((lms, index) => {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const l of lms) {
      if (l.x < minX) minX = l.x;
      if (l.x > maxX) maxX = l.x;
      if (l.y < minY) minY = l.y;
      if (l.y > maxY) maxY = l.y;
    }
    return {
      index,
      area: (maxX - minX) * (maxY - minY),
      cx: (minX + maxX) / 2,
      cy: (minY + maxY) / 2,
    };
  });
}

// Picks which detected face the OVERLAY highlights as primary (largest bbox
// area, with centroid stickiness + size hysteresis so the highlight doesn't
// flip on detection jitter) and drives the "People" count. DISPLAY ONLY:
// the model's input comes from the separate numFaces:1 feature landmarker
// (usePipeline.ts), whose single face is selected by MediaPipe internally —
// with multiple people in frame it is NOT guaranteed to be the face this
// function highlights. Documented as a known limitation (thesis §5.2.2);
// reconciling the two would require feeding display landmarks to the model,
// which J1 parity evidence forbids (numFaces:4 fails the gate on blinks).
export function selectPrimaryFace(
  faces: Landmark[][],
  prev: { cx: number; cy: number } | null,
): FaceStats | null {
  const stats = faceStats(faces).filter((s) => s.area > 0);
  if (!stats.length) return null;
  const largest = stats.reduce((a, b) => (b.area > a.area ? b : a));
  if (!prev) return largest;
  let incumbent: FaceStats | null = null;
  let best = STICKINESS_RADIUS;
  for (const s of stats) {
    const d = Math.hypot(s.cx - prev.cx, s.cy - prev.cy);
    if (d < best) { best = d; incumbent = s; }
  }
  if (!incumbent) return largest;
  return largest.area > HYSTERESIS * incumbent.area ? largest : incumbent;
}
