'use client';
import { useEffect, useRef } from 'react';

export default function Sparkline({ values, max = 3 }: { values: number[]; max?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext('2d')!;
    ctx.clearRect(0, 0, c.width, c.height);
    if (values.length < 2) return;
    ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 2; ctx.beginPath();
    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * c.width;
      const y = c.height - 4 - (v / max) * (c.height - 8);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }, [values, max]);
  return <canvas ref={ref} width={560} height={80} className="w-full" />;
}
