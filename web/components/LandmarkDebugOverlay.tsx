'use client';
import { useEffect, useRef } from 'react';
import {
  type Landmark, LEFT_EYE_EAR, RIGHT_EYE_EAR, MOUTH, LEFT_BROW, RIGHT_BROW,
  NOSE_TIP, CHIN, LEFT_IRIS, RIGHT_IRIS, INTEROCULAR,
} from '../lib/features';

// Visual verification tool for CONTRACT.md section 4 ("verify visually on
// Day 3 — draw the indices on a still frame and check each sits where it
// should"). Draws every contract-referenced landmark group in its own
// color with a text label, plus all 478 points dim in the background for
// context. Purely a debug aid — never used by the feature-extraction path.
const GROUPS: { indices: number[]; labels: string[]; color: string }[] = [
  { indices: LEFT_EYE_EAR, labels: ['L1', 'L2', 'L3', 'L4', 'L5', 'L6'], color: '#22d3ee' },
  { indices: RIGHT_EYE_EAR, labels: ['R1', 'R2', 'R3', 'R4', 'R5', 'R6'], color: '#facc15' },
  { indices: MOUTH, labels: ['M-L', 'M-R', 'M-U', 'M-D'], color: '#f472b6' },
  { indices: LEFT_BROW, labels: ['LB1', 'LB2', 'LB3', 'LB4', 'LB5'], color: '#fb923c' },
  { indices: RIGHT_BROW, labels: ['RB1', 'RB2', 'RB3', 'RB4', 'RB5'], color: '#a3e635' },
  { indices: [NOSE_TIP], labels: ['NOSE'], color: '#ffffff' },
  { indices: [CHIN], labels: ['CHIN'], color: '#ffffff' },
  { indices: LEFT_IRIS, labels: ['LI', '', '', '', ''], color: '#34d399' },
  { indices: RIGHT_IRIS, labels: ['RI', '', '', '', ''], color: '#34d399' },
];

export default function LandmarkDebugOverlay({
  landmarks, mirrored = true,
}: { landmarks: Landmark[] | null; mirrored?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext('2d')!;
    ctx.clearRect(0, 0, c.width, c.height);
    if (!landmarks) return;

    const px = (l: Landmark) => ({ x: (mirrored ? 1 - l.x : l.x) * c.width, y: l.y * c.height });

    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    for (const l of landmarks) {
      const { x, y } = px(l);
      ctx.fillRect(x - 0.5, y - 0.5, 1, 1);
    }

    ctx.font = 'bold 11px monospace';
    ctx.lineWidth = 3;
    for (const { indices, labels, color } of GROUPS) {
      ctx.fillStyle = color;
      ctx.strokeStyle = 'rgba(0,0,0,0.85)';
      indices.forEach((idx, i) => {
        const l = landmarks[idx];
        if (!l) return;
        const { x, y } = px(l);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
        const label = labels[i];
        if (label) {
          ctx.strokeText(label, x + 5, y - 5);
          ctx.fillText(label, x + 5, y - 5);
        }
      });
    }

    // Interocular corners (33, 263) double as LEFT_EYE_EAR p1 / RIGHT_EYE_EAR
    // p4 above — ring them so the reused points are visually distinct.
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    for (const idx of INTEROCULAR) {
      const l = landmarks[idx]; if (!l) continue;
      const { x, y } = px(l);
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, [landmarks, mirrored]);

  return <canvas ref={ref} width={640} height={480}
    className="pointer-events-none absolute inset-0 h-full w-full" />;
}
