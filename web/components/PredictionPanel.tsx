'use client';
import Sparkline from './Sparkline';
import type { Prediction } from '../hooks/usePipeline';

const LEVELS = ['Very Low', 'Low', 'High', 'Very High'];
const STATES = ['Bored', 'Confused', 'Engaged', 'Frustrated'];

export default function PredictionPanel({
  prediction, history,
}: { prediction: Prediction | null; history: number[] }) {
  const level = prediction ? prediction.engagement.indexOf(Math.max(...prediction.engagement)) : null;
  return (
    <section className="rounded-2xl bg-zinc-900 p-6">
      <h2 className="text-sm font-semibold uppercase tracking-widest text-zinc-500">Engagement</h2>
      <div className="mt-2 flex items-baseline gap-4">
        <span className="text-6xl font-bold text-cyan-400">{level === null ? '—' : level}</span>
        <span className="text-2xl text-zinc-300">{level === null ? 'warming up' : LEVELS[level]}</span>
      </div>
      <div className="mt-6 space-y-3">
        {STATES.map((name, i) => {
          const p = prediction?.states[i] ?? 0;
          return (
            <div key={name} className="flex items-center gap-3">
              <span className="w-28 text-lg text-zinc-300">{name}</span>
              <div className="h-4 flex-1 rounded bg-zinc-800">
                <div className="h-4 rounded bg-cyan-400 transition-[width] duration-300"
                     style={{ width: `${(p * 100).toFixed(1)}%` }} />
              </div>
              <span className="w-16 text-right text-lg tabular-nums">{(p * 100).toFixed(0)}%</span>
            </div>
          );
        })}
      </div>
      <div className="mt-6">
        <h3 className="text-xs uppercase tracking-widest text-zinc-500">last 60 s</h3>
        <Sparkline values={history} />
      </div>
    </section>
  );
}
