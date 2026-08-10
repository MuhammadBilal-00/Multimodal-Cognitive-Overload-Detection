'use client';
import TrendChart from './Sparkline';
import type { Prediction } from '../hooks/usePipeline';

const LEVELS = ['Very Low', 'Low', 'High', 'Very High'];
const STATES = [
  { name: 'Bored', color: '#64748b' },
  { name: 'Confused', color: '#d97706' },
  { name: 'Engaged', color: '#16a34a' },
  { name: 'Frustrated', color: '#dc2626' },
];

export default function PredictionPanel({
  prediction, history,
}: { prediction: Prediction | null; history: number[][] }) {
  const level = prediction ? prediction.engagement.indexOf(Math.max(...prediction.engagement)) : null;
  const series = STATES.map((s, i) => ({
    name: s.name,
    color: s.color,
    values: history.map((h) => h[i]),
  }));
  const hasTrend = history.length >= 2;

  return (
    <section className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-6 shadow-sm">
      <h2 className="text-xs font-semibold uppercase tracking-widest text-[var(--foreground-muted)]">Engagement</h2>
      <div className="mt-2 flex items-baseline gap-4">
        <span className="text-6xl font-bold text-[var(--accent)]">{level === null ? '—' : level}</span>
        <span className="text-2xl text-[var(--foreground)]">{level === null ? 'warming up' : LEVELS[level]}</span>
      </div>
      <div className="mt-6 space-y-3">
        {STATES.map(({ name, color }, i) => {
          const p = prediction?.states[i] ?? 0;
          return (
            <div key={name} className="flex items-center gap-3">
              <span className="w-24 text-sm font-medium text-[var(--foreground)]">{name}</span>
              <div className="h-3 flex-1 rounded-full bg-[var(--surface-sunken)]">
                <div className="h-3 rounded-full transition-[width] duration-300"
                     style={{ width: `${(p * 100).toFixed(1)}%`, backgroundColor: color }} />
              </div>
              <span className="w-12 text-right text-sm font-medium tabular-nums text-[var(--foreground-muted)]">
                {(p * 100).toFixed(0)}%
              </span>
            </div>
          );
        })}
      </div>
      <div className="mt-6">
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
          <h3 className="text-xs font-semibold uppercase tracking-widest text-[var(--foreground-muted)]">Last 60 s</h3>
          <div className="flex flex-wrap gap-3">
            {STATES.map(({ name, color }) => (
              <span key={name} className="flex items-center gap-1.5 text-xs text-[var(--foreground-muted)]">
                <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color }} />
                {name}
              </span>
            ))}
          </div>
        </div>
        <div className="mt-2 rounded-xl bg-[var(--surface-sunken)] p-2">
          {hasTrend
            ? <TrendChart series={series} />
            : (
              <div className="grid h-24 place-items-center text-sm text-[var(--foreground-muted)]">
                Collecting data — trend appears after ~6 s of predictions
              </div>
            )}
        </div>
      </div>
    </section>
  );
}
