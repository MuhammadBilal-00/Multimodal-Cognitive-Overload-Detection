'use client';
import { useEffect, useRef, useState } from 'react';
import WebcamFeed from './WebcamFeed';
import LandmarkOverlay from './LandmarkOverlay';
import LandmarkDebugOverlay from './LandmarkDebugOverlay';
import PredictionPanel from './PredictionPanel';
import PerfHUD from './PerfHUD';
import FeaturePanel from './FeaturePanel';
import BenchmarkPanel from './BenchmarkPanel';
import { EyeIcon, ActivityIcon, ListIcon, GaugeIcon, PeopleIcon } from './icons';
import { usePipeline } from '../hooks/usePipeline';

function formatElapsed(seconds: number) {
  const m = Math.floor(seconds / 60).toString().padStart(2, '0');
  const s = (seconds % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

function ToggleButton({
  active, onClick, icon, label,
}: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) {
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-colors ${
        active
          ? 'bg-[var(--accent-soft)] text-[var(--accent)]'
          : 'text-[var(--foreground-muted)] hover:bg-[var(--surface-sunken)]'
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

export default function CognitiveApp() {
  const { status, error, landmarks, faces, primaryIndex, features, prediction, perf, onVideoReady } = usePipeline();
  const [history, setHistory] = useState<number[][]>([]);
  const [debugLandmarks, setDebugLandmarks] = useState(false);
  const [statsOpen, setStatsOpen] = useState(false);
  const [featuresOpen, setFeaturesOpen] = useState(false);
  const [benchmarkOpen, setBenchmarkOpen] = useState(false);
  const lastPred = useRef<typeof prediction>(null);

  useEffect(() => {
    if (!prediction || prediction === lastPred.current) return;
    lastPred.current = prediction;
    setHistory((h) => [...h.slice(-19), prediction.states]); // 20 points @ 1 per 3 s = 60 s
  }, [prediction]);

  const isLive = status === 'live' && !error;
  const [liveSince, setLiveSince] = useState<number | null>(null);
  useEffect(() => {
    if (isLive && liveSince === null) setLiveSince(Date.now());
    if (!isLive && liveSince !== null) setLiveSince(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLive]);
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (liveSince === null) { setElapsed(0); return; }
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - liveSince) / 1000)), 1000);
    return () => clearInterval(id);
  }, [liveSince]);

  return (
    <main className="min-h-screen bg-[var(--background)] p-4 text-[var(--foreground)] md:p-6">
      <div className="mx-auto max-w-6xl">
        <header className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-[var(--border)] bg-[var(--surface)] px-5 py-4 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-full bg-[var(--accent-soft)] text-[var(--accent)]">
              <ActivityIcon className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-lg font-semibold leading-tight">Cognitive State</h1>
              <p className="text-xs text-[var(--foreground-muted)]">Live in-browser inference — nothing leaves your device</p>
            </div>
          </div>
          {error ? (
            <span className="flex items-center gap-2 rounded-full bg-red-50 px-3 py-1.5 text-sm font-medium text-[var(--warn)]">
              <span className="h-2 w-2 rounded-full bg-[var(--warn)]" />
              {error}
            </span>
          ) : (
            <span className="flex items-center gap-2 rounded-full bg-[var(--surface-sunken)] px-3 py-1.5 text-sm font-medium">
              <span
                className={`h-2 w-2 rounded-full ${isLive ? 'animate-live-pulse bg-[var(--live)]' : 'bg-zinc-400'}`}
              />
              <span className={isLive ? 'text-[var(--live)]' : 'text-[var(--foreground-muted)]'}>
                {isLive ? 'Live' : status}
              </span>
              {isLive && <span className="tabular-nums text-[var(--foreground-muted)]">{formatElapsed(elapsed)}</span>}
            </span>
          )}
        </header>

        <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
          <div className="space-y-4">
            <div className="relative">
              <WebcamFeed onVideoReady={onVideoReady} />
              {debugLandmarks
                ? <LandmarkDebugOverlay landmarks={landmarks} />
                : <LandmarkOverlay faces={faces} primaryIndex={primaryIndex} />}
              {faces.length > 0 && (
                <span className="absolute right-3 top-3 flex items-center gap-1.5 rounded-full bg-black/55 px-3 py-1 text-sm font-medium text-white backdrop-blur">
                  <PeopleIcon className="h-4 w-4" />
                  {faces.length}
                </span>
              )}
            </div>

            <div className="flex flex-wrap items-center justify-center gap-1 rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-2 shadow-sm">
              <ToggleButton
                active={debugLandmarks}
                onClick={() => setDebugLandmarks((v) => !v)}
                icon={<EyeIcon className="h-4 w-4" />}
                label="Landmarks"
              />
              <ToggleButton
                active={statsOpen}
                onClick={() => setStatsOpen((v) => !v)}
                icon={<ActivityIcon className="h-4 w-4" />}
                label="Stats"
              />
              <ToggleButton
                active={featuresOpen}
                onClick={() => setFeaturesOpen((v) => !v)}
                icon={<ListIcon className="h-4 w-4" />}
                label="Features"
              />
              <ToggleButton
                active={benchmarkOpen}
                onClick={() => setBenchmarkOpen((v) => !v)}
                icon={<GaugeIcon className="h-4 w-4" />}
                label="Benchmark"
              />
            </div>

            {statsOpen && <PerfHUD perf={perf} />}
            {featuresOpen && <FeaturePanel features={features} />}
            {benchmarkOpen && <BenchmarkPanel />}
          </div>

          <div className="space-y-4">
            <PredictionPanel prediction={prediction} history={history} />
          </div>
        </div>
      </div>
    </main>
  );
}
