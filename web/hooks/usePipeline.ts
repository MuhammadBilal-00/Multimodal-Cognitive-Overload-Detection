'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { FaceLandmarker } from '@mediapipe/tasks-vision';
import { createDisplayLandmarker, createFeatureLandmarker } from '../lib/faceLandmarker';
import { computeFeatures, type Landmark } from '../lib/features';
import { selectPrimaryFace } from '../lib/primaryFace';
import { RingBuffer } from '../lib/ringBuffer';
import { loadScaler, type Scaler } from '../lib/scaler';
import { initInference, runInference } from '../lib/inference';
import type { OodReport } from '../lib/mathUtils';
import { STATE_CHANNELS } from '../lib/states';

export interface Prediction {
  engagement: number[];
  states: number[];
  // How far this window sits from the scaler's training distribution, and
  // whether a face was even present. Drives the "reading may be unreliable"
  // and "no face" notes instead of showing a confident-looking percentage
  // that isn't — see lib/mathUtils.ts distributionCheck().
  ood: OodReport;
  ms: number;
}

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
  // Named view of prediction.states. The bare positional array is exactly
  // what let a channel-order bug sit unnoticed in docs/results/app_e2e.json
  // for weeks — recording the names alongside makes the next one obvious.
  statesNamed: Record<string, number> | null;
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
  // Real stream dimensions. getUserMedia treats WebcamFeed's 640x480 request
  // as *ideal*, so the delivered stream may be 16:9; the landmark overlay
  // needs the true size to line its dots up with the object-cover video.
  const [videoSize, setVideoSize] = useState({ w: 0, h: 0 });
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
  // Benchmark pause is explicit, set by BenchmarkPanel via setPaused below.
  // Tab-hidden pause is read live off document.hidden/hasFocus() in the loop
  // guard instead — NOT cached via a visibilitychange listener. A listener-
  // driven flag only updates on the transition event; if the tab is already
  // hidden when this effect mounts (or an embedding context never fires a
  // clean show/hide pair — confirmed happening under at least one automated
  // browser-driver setup, where document.hidden reads true even while
  // document.hasFocus() is true), the flag can latch "paused" with nothing
  // left to ever clear it. Reading both live each frame is just as cheap and
  // can't get stuck. Requiring hidden AND NOT focused (not hidden alone)
  // additionally avoids pausing in exactly that hidden-but-focused case.
  const pausedByBenchmarkRef = useRef(false);

  useEffect(() => {
    let dead = false;
    let createdDisplayLandmarker: FaceLandmarker | null = null;
    let createdFeatureLandmarker: FaceLandmarker | null = null;
    (async () => {
      // allSettled, not all: each landmarker holds WASM + a GL context
      // ("Memory climbs steadily" territory), so one that resolved must be
      // captured and closed even if a sibling promise (e.g. loadScaler()
      // failing feature_names validation) rejects. Promise.all previously
      // skipped the destructuring entirely on any single rejection, orphaning
      // whichever landmarker(s) had already succeeded.
      const [displayResult, featureResult, scalerResult, infoResult] = await Promise.allSettled([
        createDisplayLandmarker(), createFeatureLandmarker(), loadScaler(), initInference(),
      ]);
      if (displayResult.status === 'fulfilled') createdDisplayLandmarker = displayResult.value;
      if (featureResult.status === 'fulfilled') createdFeatureLandmarker = featureResult.value;

      // React StrictMode double-invokes this effect in dev; a landmarker
      // created by an already-cleaned-up run must be closed, not leaked.
      if (dead) { createdDisplayLandmarker?.close(); createdFeatureLandmarker?.close(); return; }

      if (displayResult.status !== 'fulfilled' || featureResult.status !== 'fulfilled'
          || scalerResult.status !== 'fulfilled' || infoResult.status !== 'fulfilled') {
        createdDisplayLandmarker?.close();
        createdFeatureLandmarker?.close();
        createdDisplayLandmarker = null;
        createdFeatureLandmarker = null;
        const failure = [displayResult, featureResult, scalerResult, infoResult]
          .find((r): r is PromiseRejectedResult => r.status === 'rejected');
        const reason = failure?.reason;
        setError(reason instanceof Error ? reason.message : String(reason));
        return;
      }

      displayLandmarkerRef.current = createdDisplayLandmarker;
      featureLandmarkerRef.current = createdFeatureLandmarker;
      scalerRef.current = scalerResult.value;
      setPerf((p) => ({ ...p, modelBytes: infoResult.value.modelBytes, backend: infoResult.value.backend, threads: infoResult.value.threads }));
      setModelsReady(true);
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
    if (!video || !displayLmk || !featureLmk || !scaler || video.readyState < 2
        || pausedByBenchmarkRef.current || (document.hidden && !document.hasFocus())) return;

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

    setVideoSize((s) => (s.w === video.videoWidth && s.h === video.videoHeight
      ? s : { w: video.videoWidth, h: video.videoHeight }));

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

  // Exposed so BenchmarkPanel can pause live sampling/inference while it
  // runs its own 300-cycle loop against the same ONNX session singleton
  // (lib/inference.ts) — otherwise both consumers contend for the same WASM
  // thread pool and neither result is clean.
  const setPaused = useCallback((p: boolean) => { pausedByBenchmarkRef.current = p; }, []);

  useEffect(() => {
    window.__ENGINE_STATE = {
      status, prediction,
      facePresent: features ? features[12] === 1 : false,
      statesNamed: prediction
        ? Object.fromEntries(
            STATE_CHANNELS.map((c, i) => [c.key, prediction.states[i]]))
        : null,
    };
  }, [status, prediction, features]);

  const landmarks = faceState.faces[faceState.primaryIndex] ?? null;
  return {
    status, error, landmarks, faces: faceState.faces,
    primaryIndex: faceState.primaryIndex, features, prediction, perf,
    videoSize, onVideoReady, setPaused,
  };
}
