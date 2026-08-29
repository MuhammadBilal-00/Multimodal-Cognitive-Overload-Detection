'use client';
import TrendChart from './Sparkline';
import { STATE_CHANNELS } from '../lib/states';
import { FEATURE_NAMES } from '../lib/features';
import type { Prediction } from '../hooks/usePipeline';

const LEVELS = ['Very Low', 'Low', 'High', 'Very High'];

// Plain-language nudges for the features most likely to put a real user
// outside DAiSEE's laptop-webcam framing.
const FEATURE_ADVICE: Record<string, string> = {
  face_area: 'try sitting further back',
  gaze_y: 'try raising your camera to eye level',
  gaze_x: 'try centring yourself in frame',
  yaw: 'try facing the camera straight on',
  pitch: 'try levelling your camera',
  roll: 'try keeping your head upright',
};

export default function PredictionPanel({
  prediction, history,
}: { prediction: Prediction | null; history: number[][] }) {
  // No-face windows standardise to extreme out-of-distribution inputs the
  // model has essentially never seen (DAiSEE detection rate 99.96%), so its
  // output there is arbitrary — showing "Frustrated 99%" under a no-face
  // caveat is honest-but-distracting garbage. Suppress the numbers entirely
  // and show placeholders; the notice below explains why.
  const noFace = prediction?.ood.noFace ?? false;
  const level = prediction && !noFace
    ? prediction.engagement.indexOf(Math.max(...prediction.engagement)) : null;
  const series = STATE_CHANNELS.map((s, i) => ({
    name: s.label,
    color: s.color,
    values: history.map((h) => h[i]),
  }));
  const hasTrend = history.length >= 2;
  const ood = prediction?.ood ?? null;
  const advice = ood?.offenders
    .map((o) => FEATURE_ADVICE[FEATURE_NAMES[o.feature]])
    .filter((a): a is string => Boolean(a))[0];

  return (
    <section className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-6 shadow-sm">
      <h2 className="text-xs font-semibold uppercase tracking-widest text-[var(--foreground-muted)]">Engagement</h2>
      <div className="mt-2 flex items-baseline gap-4">
        <span className="text-6xl font-bold text-[var(--accent)]">{level === null ? '—' : level}</span>
        <span className="text-2xl text-[var(--foreground)]">
          {noFace ? 'no face' : level === null ? 'warming up' : LEVELS[level]}
        </span>
      </div>

      {/* Muted, not red: a close-sitting user is a normal user meeting a real
          model limitation, not an error. --warn is reserved for camera/model
          failures in the header. */}
      {ood?.noFace && (
        <p role="status" aria-live="polite" className="mt-4 rounded-xl bg-[var(--surface-sunken)] px-3 py-2 text-xs leading-relaxed text-[var(--foreground-muted)]">
          <span className="font-semibold text-[var(--foreground)]">No face detected.</span>{' '}
          Most of the last 3 seconds had nobody in frame, so the model has no
          meaningful input — readings resume as soon as a face is back.
        </p>
      )}
      {ood && !ood.noFace && ood.outOfDistribution && (
        <p role="status" aria-live="polite" className="mt-4 rounded-xl bg-[var(--surface-sunken)] px-3 py-2 text-xs leading-relaxed text-[var(--foreground-muted)]">
          <span className="font-semibold text-[var(--foreground)]">Reading may be unreliable.</span>{' '}
          Your camera framing sits outside the range this model was trained on
          ({ood.sustainedSigma.toFixed(1)}σ){advice ? ` — ${advice}` : ''}.
        </p>
      )}

      <div className="mt-6">
        <div className="flex items-baseline justify-between gap-3">
          <h3 className="text-xs font-semibold uppercase tracking-widest text-[var(--foreground-muted)]">
            State likelihood
          </h3>
        </div>
        <p className="mt-1 text-xs text-[var(--foreground-muted)]">
          Each state is scored independently — these don’t add up to 100%.
        </p>
        <div className="mt-3 space-y-3">
          {STATE_CHANNELS.map(({ key, label, color }, i) => {
            const p = noFace ? 0 : prediction?.states[i] ?? 0;
            return (
              <div key={key} className="flex items-center gap-3">
                <span className="w-24 text-sm font-medium text-[var(--foreground)]">{label}</span>
                <div className="h-3 flex-1 rounded-full bg-[var(--surface-sunken)]">
                  <div className="h-3 rounded-full transition-[width] duration-300"
                       style={{ width: `${(p * 100).toFixed(1)}%`, backgroundColor: color }} />
                </div>
                <span className="w-12 text-right text-sm font-medium tabular-nums text-[var(--foreground-muted)]">
                  {noFace ? '—' : `${(p * 100).toFixed(0)}%`}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="mt-6">
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
          <h3 className="text-xs font-semibold uppercase tracking-widest text-[var(--foreground-muted)]">Last 60 s</h3>
          <div className="flex flex-wrap gap-3">
            {STATE_CHANNELS.map(({ key, label, color }) => (
              <span key={key} className="flex items-center gap-1.5 text-xs text-[var(--foreground-muted)]">
                <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color }} />
                {label}
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
