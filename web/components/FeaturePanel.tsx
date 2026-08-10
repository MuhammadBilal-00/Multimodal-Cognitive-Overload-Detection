'use client';
import { FEATURE_NAMES } from '../lib/features';

export default function FeaturePanel({ features }: { features: Float32Array | null }) {
  return (
    <section className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-6 shadow-sm">
      <h2 className="text-xs font-semibold uppercase tracking-widest text-[var(--foreground-muted)]">
        Live features (raw, 10 Hz)
      </h2>
      <div className="mt-3 grid grid-cols-2 gap-x-8 gap-y-1 font-mono text-sm">
        {FEATURE_NAMES.map((name, i) => (
          <div key={name} className="flex justify-between border-b border-[var(--border)] py-1">
            <span className="text-[var(--foreground-muted)]">{name}</span>
            <span className="tabular-nums text-[var(--foreground)]">{features ? features[i].toFixed(4) : '—'}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
