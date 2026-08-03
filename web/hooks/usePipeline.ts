'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { FaceLandmarker } from '@mediapipe/tasks-vision';
import { createLandmarker } from '../lib/faceLandmarker';
import { computeFeatures, type Landmark } from '../lib/features';
import { RingBuffer } from '../lib/ringBuffer';
import { loadScaler, type Scaler } from '../lib/scaler';
import { initInference, runInference } from '../lib/inference';

export interface Prediction { engagement: number[]; states: number[]; ms: number }

export function usePipeline() {
  // Camera-ready and models-ready are two independent async chains with no
  // ordering guarantee (a fake/warm camera can resolve well before several
  // MB of WASM finish loading, or vice versa) — deriving status from both
  // flags avoids the race a single imperative setStatus() sequence had, where
  // whichever chain finished LAST always overwrote the other's more-advanced
  // status text.
  const [modelsReady, setModelsReady] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [landmarks, setLandmarks] = useState<Landmark[] | null>(null);
  const [features, setFeatures] = useState<Float32Array | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [perf, setPerf] = useState({
    renderFps: 0, sampleHz: 0, inferMs: [] as number[],
    modelBytes: 0, backend: '-', threads: 0, landmarkCount: 0,
  });

  const status = !modelsReady ? 'loading models…'
    : !cameraReady ? 'waiting for camera'
    : !prediction ? 'filling 3s window…'
    : 'live';

  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const scalerRef = useRef<Scaler | null>(null);
  const bufferRef = useRef(new RingBuffer());
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const rafRef = useRef(0);
  const lastSampleRef = useRef(0);
  const sampleCountRef = useRef(0);
  const inFlightRef = useRef(false);
  const fpsCounter = useRef({ frames: 0, samples: 0, last: performance.now() });
  const inferTimes = useRef<number[]>([]);

  useEffect(() => {
    let dead = false;
    let createdLandmarker: FaceLandmarker | null = null;
    (async () => {
      try {
        const [lmk, scaler, info] = await Promise.all([
          createLandmarker(), loadScaler(), initInference(),
        ]);
        // React StrictMode double-invokes this effect in dev; a landmarker
        // created by an already-cleaned-up run must be closed, not leaked
        // (it holds WASM + a GL context — "Memory climbs steadily" territory).
        if (dead) { lmk.close(); return; }
        createdLandmarker = lmk;
        landmarkerRef.current = lmk;
        scalerRef.current = scaler;
        setPerf((p) => ({ ...p, modelBytes: info.modelBytes, backend: info.backend, threads: info.threads }));
        setModelsReady(true);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      dead = true;
      cancelAnimationFrame(rafRef.current);
      createdLandmarker?.close();
    };
  }, []);

  const loop = useCallback((now: number) => {
    rafRef.current = requestAnimationFrame(loop);
    const video = videoRef.current, lmk = landmarkerRef.current, scaler = scalerRef.current;
    if (!video || !lmk || !scaler || video.readyState < 2) return;

    const c = fpsCounter.current;
    c.frames++;
    if (now - c.last >= 1000) {
      // Snapshot before resetting: setPerf's updater runs lazily at React's
      // next flush, by which point a later tick may have already zeroed
      // these ref fields — reading c.* inside the updater would then report 0.
      const renderFps = c.frames, sampleHz = c.samples, inferMs = [...inferTimes.current];
      c.frames = 0; c.samples = 0; c.last = now;
      setPerf((p) => ({ ...p, renderFps, sampleHz, inferMs }));
    }

    if (now - lastSampleRef.current < 100) return; // contract: 10 Hz sampling
    lastSampleRef.current = now;
    c.samples++;

    const result = lmk.detectForVideo(video, now);
    const lm = (result.faceLandmarks[0] as Landmark[] | undefined) ?? null;
    if (lm && lm.length !== 478) console.error(`landmark count ${lm.length}, expected 478 — iris missing?`);
    setLandmarks(lm);
    setPerf((p) => (p.landmarkCount === (lm?.length ?? 0) ? p : { ...p, landmarkCount: lm?.length ?? 0 }));

    const f = computeFeatures(lm, video.videoWidth, video.videoHeight, scaler.pitch_centre);
    setFeatures(f);
    bufferRef.current.push(f);
    sampleCountRef.current++;

    if (bufferRef.current.isFull() && sampleCountRef.current % 5 === 0 && !inFlightRef.current) {
      inFlightRef.current = true;
      runInference(bufferRef.current.window(), scaler)
        .then((pred) => {
          inferTimes.current = [...inferTimes.current.slice(-29), pred.ms];
          setPrediction(pred);
        })
        .catch((e) => setError(e instanceof Error ? e.message : String(e)))
        .finally(() => { inFlightRef.current = false; });
    }
  }, []);

  const onVideoReady = useCallback((v: HTMLVideoElement) => {
    videoRef.current = v;
    setCameraReady(true);
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(loop);
  }, [loop]);

  return { status, error, landmarks, features, prediction, perf, onVideoReady };
}
