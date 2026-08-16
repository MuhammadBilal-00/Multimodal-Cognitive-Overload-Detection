'use client';
import { useEffect, useRef, useState } from 'react';

export interface TrendSeries {
  name: string;
  color: string;
  values: number[]; // 0..1 probabilities, one per prediction tick
}

// Multi-line trend of the four state probabilities over the retained
// prediction history (real model output — one point per completed 3 s
// inference window, ~20 points ≈ last 60 s). Replaced the old single
// quantized-level step line, which only plotted argmax(engagement) and
// read as a placeholder rather than the model's actual confidence.
export default function TrendChart({ series, height = 96 }: { series: TrendSeries[]; height?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  // Measured, not hardcoded: the canvas's CSS width is 100% of its container
  // (see the returned <canvas> below), but the backing store used to be a
  // fixed 560px regardless — stretching/blurring the line art on any
  // container narrower than that. 560 here is only the pre-measurement
  // fallback for the first paint.
  const [width, setWidth] = useState(560);
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;

  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ro = new ResizeObserver(([entry]) => {
      const w = entry.contentRect.width;
      if (w > 0) setWidth(w);
    });
    ro.observe(c);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext('2d')!;
    c.width = width * dpr; c.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const padTop = 6, padBottom = 6, padX = 2;
    const plotH = height - padTop - padBottom;

    // Gridlines at 0 / 50 / 100 %.
    ctx.strokeStyle = '#e3e6ec';
    ctx.lineWidth = 1;
    [0, 0.5, 1].forEach((f) => {
      const y = padTop + plotH - f * plotH;
      ctx.beginPath();
      ctx.moveTo(padX, y);
      ctx.lineTo(width - padX, y);
      ctx.stroke();
    });

    const longest = Math.max(...series.map((s) => s.values.length), 2);

    series.forEach((s) => {
      if (s.values.length < 2) return;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      s.values.forEach((v, i) => {
        // Right-align shorter histories so the newest point always sits at
        // the right edge, same x-scale as a full 20-point window.
        const idx = i + (longest - s.values.length);
        const x = padX + (idx / (longest - 1)) * (width - padX * 2);
        const y = padTop + plotH - v * plotH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Dot on the latest sample.
      const lastV = s.values[s.values.length - 1];
      const lastX = width - padX;
      const lastY = padTop + plotH - lastV * plotH;
      ctx.fillStyle = s.color;
      ctx.beginPath();
      ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [series, dpr, height, width]);

  return <canvas ref={ref} style={{ width: '100%', height }} className="block" />;
}
