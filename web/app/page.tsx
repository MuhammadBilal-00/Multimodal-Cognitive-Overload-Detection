'use client';
import { useEffect, useRef, useState } from 'react';
import WebcamFeed from '../components/WebcamFeed';
import LandmarkOverlay from '../components/LandmarkOverlay';
import PredictionPanel from '../components/PredictionPanel';
import PerfHUD from '../components/PerfHUD';
import FeaturePanel from '../components/FeaturePanel';
import BenchmarkPanel from '../components/BenchmarkPanel';
import { usePipeline } from '../hooks/usePipeline';

export default function Home() {
  const { status, error, landmarks, features, prediction, perf, onVideoReady } = usePipeline();
  const [history, setHistory] = useState<number[]>([]);
  const lastPred = useRef<typeof prediction>(null);

  useEffect(() => {
    if (!prediction || prediction === lastPred.current) return;
    lastPred.current = prediction;
    const level = prediction.engagement.indexOf(Math.max(...prediction.engagement));
    setHistory((h) => [...h.slice(-119), level]); // 120 points @ 2 Hz = 60 s
  }, [prediction]);

  return (
    <main className="min-h-screen bg-zinc-950 p-6 text-zinc-100">
      <header className="mb-6 flex items-baseline justify-between">
        <h1 className="text-2xl font-bold">Cognitive State — In-Browser Inference</h1>
        <span className={`rounded-full px-3 py-1 text-sm ${error ? 'bg-red-900 text-red-200' : 'bg-zinc-800 text-cyan-400'}`}>
          {error ?? status}
        </span>
      </header>
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-6">
          <div className="relative">
            <WebcamFeed onVideoReady={onVideoReady} />
            <LandmarkOverlay landmarks={landmarks} />
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
