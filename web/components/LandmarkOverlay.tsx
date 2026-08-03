'use client';
import { useEffect, useRef } from 'react';
import type { Landmark } from '../lib/features';

export default function LandmarkOverlay({
  landmarks, mirrored = true,
}: { landmarks: Landmark[] | null; mirrored?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext('2d')!;
    ctx.clearRect(0, 0, c.width, c.height);
    if (!landmarks) return;
    ctx.fillStyle = '#22d3ee';
    for (const l of landmarks) {
      const x = (mirrored ? 1 - l.x : l.x) * c.width;
      ctx.fillRect(x - 0.5, l.y * c.height - 0.5, 1.5, 1.5);
    }
  }, [landmarks, mirrored]);
  return <canvas ref={ref} width={640} height={480}
    className="pointer-events-none absolute inset-0 h-full w-full" />;
}
