'use client';
// J1 feature-parity gate, browser side. Runs the SAME 100 pre-extracted
// PNG frames (ml/scripts/make_parity_fixture.py) through the production
// model-feeding landmarker (createFeatureLandmarker(), numFaces: 1 — the
// SAME path usePipeline.ts feeds computeFeatures() from; NOT the numFaces:
// 4 display landmarker, which measurably fails this gate on blink frames,
// see lib/faceLandmarker.ts) and lib/features.ts, then compares against
// ml/tests/fixtures/parity_expected.json (the Python reference).
// Driven by web/tests/e2e/features.parity.test.ts; result lands on
// window.__PARITY_RESULT__.
//
// Static PNGs instead of a seeked <video> element: headless Chromium does
// not reliably present frames after currentTime seeks (verified: the old
// harness's browser feature series was near-constant while Python's
// varied). Each frame here is a distinct fetch of a distinct image, so
// that whole class of bug is structurally impossible.
import { useEffect, useState } from 'react';
import { createFeatureLandmarker } from '../../lib/faceLandmarker';
import { computeFeatures, FEATURE_NAMES, type Landmark } from '../../lib/features';

interface ParityExpected {
  version: string;
  num_frames: number;
  pitch_centre_used: number;
  feature_names: string[];
  features: number[][];
  video: { width: number; height: number; fps: number; sample_step: number };
}

export interface ParityResult {
  ok: boolean;
  error: string | null;
  numFrames: number;
  framesCompared: number;
  faceMismatchFrames: number;
  perFeatureMaxAbsDiff: Record<string, number>;
  perFeatureMeanAbsDiff: Record<string, number>;
  worstFeatureDiff: number;
  tolerance: number;
  actual: number[][]; // full per-frame series, for offline diagnosis
}

// CONTRACT.md Amendment 2 — see that document for why 0.02, not the
// original 1e-4 from BUILD_PLAN_1.md §J1.
const TOLERANCE = 0.02;

declare global {
  interface Window {
    __PARITY_RESULT__?: ParityResult;
  }
}

export default function ParityTestPage() {
  const [status, setStatus] = useState('running…');

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const result: ParityResult = {
        ok: false, error: null, numFrames: 0, framesCompared: 0, faceMismatchFrames: 0,
        perFeatureMaxAbsDiff: {}, perFeatureMeanAbsDiff: {}, worstFeatureDiff: 0,
        tolerance: TOLERANCE, actual: [],
      };
      try {
        const expected: ParityExpected = await (
          await fetch('/api/parity-fixtures/parity_expected.json')).json();
        const landmarker = await createFeatureLandmarker();

        const nf = FEATURE_NAMES.length;
        const FACE = nf - 1;
        const n = expected.num_frames;
        const maxAbs = new Array(nf).fill(0);
        const sumAbs = new Array(nf).fill(0);
        let compared = 0;
        let faceMismatch = 0;

        const canvas = document.createElement('canvas');
        canvas.width = expected.video.width;
        canvas.height = expected.video.height;
        const ctx = canvas.getContext('2d')!;

        for (let k = 0; k < n; k++) {
          const frameName = `frame_${String(k).padStart(3, '0')}.png`;
          const blob = await (await fetch(`/api/parity-fixtures/parity_frames/${frameName}`)).blob();
          const bitmap = await createImageBitmap(blob);
          ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
          bitmap.close();

          const res = landmarker.detectForVideo(canvas, k * 100);
          const lm = res.faceLandmarks && res.faceLandmarks.length > 0
            ? (res.faceLandmarks[0] as Landmark[]) : null;
          const actual = computeFeatures(lm, canvas.width, canvas.height, expected.pitch_centre_used);
          result.actual.push(Array.from(actual));

          const exp = expected.features[k];
          if (exp[FACE] !== actual[FACE]) { faceMismatch += 1; continue; }
          if (exp[FACE] === 0) continue;
          compared += 1;
          for (let f = 0; f < nf; f++) {
            const d = Math.abs(exp[f] - actual[f]);
            if (d > maxAbs[f]) maxAbs[f] = d;
            sumAbs[f] += d;
          }
        }
        landmarker.close();

        result.numFrames = n;
        result.framesCompared = compared;
        result.faceMismatchFrames = faceMismatch;
        result.perFeatureMaxAbsDiff = Object.fromEntries(
          FEATURE_NAMES.map((name, f) => [name, maxAbs[f]]));
        result.perFeatureMeanAbsDiff = Object.fromEntries(
          FEATURE_NAMES.map((name, f) => [name, sumAbs[f] / Math.max(compared, 1)]));
        result.worstFeatureDiff = Math.max(...maxAbs);
        // Same gates the old harness validated: at least 95% of frames
        // comparable, at most 2% face-presence disagreement, worst
        // per-feature diff under tolerance.
        result.ok = compared >= n * 0.95
          && faceMismatch <= Math.ceil(n * 0.02)
          && result.worstFeatureDiff < TOLERANCE;
      } catch (e) {
        result.error = e instanceof Error ? (e.stack ?? e.message) : String(e);
      }
      if (!cancelled) {
        window.__PARITY_RESULT__ = result;
        setStatus(result.ok ? 'PASS' : `FAIL (see window.__PARITY_RESULT__)`);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  return <pre>{status}</pre>;
}
