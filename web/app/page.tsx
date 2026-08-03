'use client';
import WebcamFeed from '../components/WebcamFeed';
import LandmarkOverlay from '../components/LandmarkOverlay';
import { usePipeline } from '../hooks/usePipeline';

export default function Home() {
  const { status, error, landmarks, prediction, perf, onVideoReady } = usePipeline();
  return (
    <main className="min-h-screen bg-zinc-950 p-6 text-zinc-100">
      <div className="relative max-w-xl">
        <WebcamFeed onVideoReady={onVideoReady} />
        <LandmarkOverlay landmarks={landmarks} />
      </div>
      <pre className="mt-4 text-xs">{error ?? status}{'\n'}{JSON.stringify({ perf, prediction }, null, 2)}</pre>
    </main>
  );
}
