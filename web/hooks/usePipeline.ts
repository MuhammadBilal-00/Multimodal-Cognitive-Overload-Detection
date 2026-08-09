'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { FaceLandmarker } from '@mediapipe/tasks-vision';
import { createDisplayLandmarker, createFeatureLandmarker } from '../lib/faceLandmarker';
import { computeFeatures, type Landmark } from '../lib/features';
import { selectPrimaryFace } from '../lib/primaryFace';
import { RingBuffer } from '../lib/ringBuffer';
import { loadScaler, type Scaler } from '../lib/scaler';
import { initInference, runInference } from '../lib/inference';

export interface Prediction { engagement: number[]; states: number[]; ms: number }

// Samples between inferences: 10 Hz × 3 s — one prediction per non-overlapping
// 3.0 s window. CONTRACT.md §6 Amendment 1 (was 5 = 2 Hz).
const INFERENCE_STRIDE = 30;

// Minimal debug hook for ml/scripts/e2e_app_test.py (J2): an external
// Playwright driver polls real app state instead of scraping the DOM.
// Deliberately small — just enough to assert "a prediction arrived and is
// well-formed"; not a general-purpose state dump.
export interface EngineDebugState {
  status: string;
  prediction: Prediction | null;
  facePresent: boolean;
}

declare global {
  interface Window {
    __ENGINE_STATE?: EngineDebugState;
  }
}

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
  const [faceState, setFaceState] = useState<{ faces: Landmark[][]; primaryIndex: number }>(
    { faces: [], primaryIndex: -1 });
  const [features, setFeatures] = useState<Float32Array | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [perf, setPerf] = useState({
    renderFps: 0, sampleHz: 0, inferMs: [] as number[],
    modelBytes: 0, backend: '-', threads: 0, landmarkCount: 0, faceCount: 0,
  });

  const status = !modelsReady ? 'loading models…'
    : !cameraReady ? 'waiting for camera'
    : !prediction ? 'filling 3s window…'
    : 'live';

  // Two landmarkers, deliberately: displayLandmarker (numFaces: 4) drives
  // the on-screen overlay + "People" count; featureLandmarker (numFaces: 1)
  // is the ONLY one ever fed into computeFeatures(), because the model was
  // trained on Python features extracted with num_faces=1 and numFaces > 1
  // measurably shifts landmarks enough to fail J1 on blink frames (see
  // lib/faceLandmarker.ts). Costs a second WASM detection pass per sampled
  // frame.
  const displayLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const featureLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const scalerRef = useRef<Scaler | null>(null);
  const bufferRef = useRef(new RingBuffer());
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const rafRef = useRef(0);
  const lastSampleRef = useRef(0);
  const sampleCountRef = useRef(0);
  const inFlightRef = useRef(false);
  const prevPrimaryRef = useRef<{ cx: number; cy: number } | null>(null);
  const fpsCounter = useRef({ frames: 0, samples: 0, last: performance.now() });
  const inferTimes = useRef<number[]>([]);

  useEffect(() => {
    let dead = false;
    let createdDisplayLandmarker: FaceLandmarker | null = null;
    let createdFeatureLandmarker: FaceLandmarker | null = null;
    (async () => {
      try {
        const [displayLmk, featureLmk, scaler, info] = await Promise.all([
          createDisplayLandmarker(), createFeatureLandmarker(), loadScaler(), initInference(),
        ]);
        // React StrictMode double-invokes this effect in dev; a landmarker
        // created by an already-cleaned-up run must be closed, not leaked
        // (it holds WASM + a GL context — "Memory climbs steadily" territory).
        if (dead) { displayLmk.close(); featureLmk.close(); return; }
        createdDisplayLandmarker = displayLmk;
        createdFeatureLandmarker = featureLmk;
        displayLandmarkerRef.current = displayLmk;
        featureLandmarkerRef.current = featureLmk;
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
      createdDisplayLandmarker?.close();
      createdFeatureLandmarker?.close();
    };
  }, []);

  const loop = useCallback((now: number) => {
    rafRef.current = requestAnimationFrame(loop);
    const video = videoRef.current, displayLmk = displayLandmarkerRef.current,
      featureLmk = featureLandmarkerRef.current, scaler = scalerRef.current;
    if (!video || !displayLmk || !featureLmk || !scaler || video.readyState < 2) return;

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

    // Overlay/People-count path: all detected faces, display purposes only.
    const displayResult = displayLmk.detectForVideo(video, now);
    const faces = displayResult.faceLandmarks as Landmark[][];
    const primary = selectPrimaryFace(faces, prevPrimaryRef.current);
    prevPrimaryRef.current = primary ? { cx: primary.cx, cy: primary.cy } : null;
    setFaceState({ faces, primaryIndex: primary?.index ?? -1 });

    // Model-feeding path: numFaces:1 only — see lib/faceLandmarker.ts for
    // why this must stay a separate detection from the overlay above.
    const featureResult = featureLmk.detectForVideo(video, now);
    const lm = featureResult.faceLandmarks && featureResult.faceLandmarks.length > 0
      ? (featureResult.faceLandmarks[0] as Landmark[]) : null;
    if (lm && lm.length !== 478) console.error(`landmark count ${lm.length}, expected 478 — iris missing?`);
    setPerf((p) => (p.landmarkCount === (lm?.length ?? 0) && p.faceCount === faces.length
      ? p : { ...p, landmarkCount: lm?.length ?? 0, faceCount: faces.length }));

    const f = computeFeatures(lm, video.videoWidth, video.videoHeight, scaler.pitch_centre);
    setFeatures(f);
    bufferRef.current.push(f);
    sampleCountRef.current++;

    if (bufferRef.current.isFull() && sampleCountRef.current % INFERENCE_STRIDE === 0 && !inFlightRef.current) {
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

  useEffect(() => {
    window.__ENGINE_STATE = {
      status, prediction,
      facePresent: features ? features[12] === 1 : false,
    };
  }, [status, prediction, features]);

  const landmarks = faceState.faces[faceState.primaryIndex] ?? null;
  return {
    status, error, landmarks, faces: faceState.faces,
    primaryIndex: faceState.primaryIndex, features, prediction, perf, onVideoReady,
  };
}
