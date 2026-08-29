'use client';
import { useEffect, useRef } from 'react';
import type { Landmark } from '../lib/features';
import { coverFit } from '../lib/overlayFit';

export default function LandmarkOverlay({
  faces, primaryIndex, videoSize, mirrored = true,
}: {
  faces: Landmark[][];
  primaryIndex: number;
  videoSize: { w: number; h: number };
  mirrored?: boolean;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext('2d')!;
    ctx.clearRect(0, 0, c.width, c.height);
    // Match object-cover so dots track the face even on 16:9 streams.
    const fit = coverFit(videoSize.w, videoSize.h, c.width, c.height, mirrored);
    faces.forEach((landmarks, i) => {
      // Cyan marks the DISPLAY-primary face (largest bbox, sticky); other
      // faces draw dimmed. NOTE: this is a display concept only — the model
      // is fed by the separate numFaces:1 feature landmarker, whose single
      // detected face is chosen by MediaPipe internally and is not guaranteed
      // to be this highlighted face when several people are in frame (see
      // lib/faceLandmarker.ts for why the two detectors must stay separate).
      ctx.fillStyle = i === primaryIndex ? '#22d3ee' : '#71717a';
      for (const l of landmarks) {
        ctx.fillRect(fit.x(l.x) - 0.5, fit.y(l.y) - 0.5, 1.5, 1.5);
      }
    });
  }, [faces, primaryIndex, videoSize, mirrored]);
  return <canvas ref={ref} width={640} height={480} aria-hidden="true"
    className="pointer-events-none absolute inset-0 h-full w-full" />;
}
