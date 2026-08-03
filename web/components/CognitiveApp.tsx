'use client';
import { useEffect, useRef, useState } from 'react';
import WebcamFeed from './WebcamFeed';
import LandmarkOverlay from './LandmarkOverlay';
import LandmarkDebugOverlay from './LandmarkDebugOverlay';
import PredictionPanel from './PredictionPanel';
import PerfHUD from './PerfHUD';
import FeaturePanel from './FeaturePanel';
import BenchmarkPanel from './BenchmarkPanel';
import { usePipeline } from '../hooks/usePipeline';

export default function CognitiveApp() {
  const { status, error, landmarks, faces, primaryIndex, features, prediction, perf, onVideoReady } = usePipeline();
  const [history, setHistory] = useState<number[]>([]);
  const [debugLandmarks, setDebugLandmarks] = useState(false);
  const lastPred = useRef<typeof prediction>(null);

  useEffect(() => {
    if (!prediction || prediction === lastPred.current) return;
    lastPred.current = prediction;
    const level = prediction.engagement.indexOf(Math.max(...prediction.engagement));
    setHistory((h) => [...h.slice(-19), level]); // 20 points @ 1 per 3 s = 60 s
  }, [prediction]);

  return (
    <main className="min-h-screen bg-zinc-950 p-6 text-zinc-100">
      <header className="mb-6 flex items-baseline justify-between">
        <h1 className="text-2xl font-bold">Cognitive State — In-Browser Inference</h1>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setDebugLandmarks((v) => !v)}
            className={`rounded-full px-3 py-1 text-sm ${debugLandmarks ? 'bg-cyan-500 text-zinc-950' : 'bg-zinc-800 text-zinc-300'}`}
          >
            {debugLandmarks ? 'Debug landmarks: ON' : 'Debug landmarks: OFF'}
          </button>
          <span className={`rounded-full px-3 py-1 text-sm ${error ? 'bg-red-900 text-red-200' : 'bg-zinc-800 text-cyan-400'}`}>
            {error ?? status}
          </span>
        </div>
      </header>
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-6">
          <div className="relative">
            <WebcamFeed onVideoReady={onVideoReady} />
            {debugLandmarks
              ? <LandmarkDebugOverlay landmarks={landmarks} />
              : <LandmarkOverlay faces={faces} primaryIndex={primaryIndex} />}
          </div>
          <FeaturePanel features={features} />
        </div>
        <div className="space-y-6">
          <PredictionPanel prediction={prediction} history={history} />
          <PerfHUD perf={perf} />
          <BenchmarkPanel />
        </div>
      </div>
    </main>
  );
}
