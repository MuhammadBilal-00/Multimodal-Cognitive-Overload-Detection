'use client';
import { FEATURE_NAMES } from '../lib/features';

export default function FeaturePanel({ features }: { features: Float32Array | null }) {
  return (
    <section className="rounded-2xl bg-zinc-900 p-6">
      <h2 className="text-sm font-semibold uppercase tracking-widest text-zinc-500">Live features (raw, 10 Hz)</h2>
      <div className="mt-3 grid grid-cols-2 gap-x-8 gap-y-1 font-mono text-sm">
        {FEATURE_NAMES.map((name, i) => (
          <div key={name} className="flex justify-between border-b border-zinc-800 py-1">
            <span className="text-zinc-400">{name}</span>
            <span className="tabular-nums text-zinc-100">{features ? features[i].toFixed(4) : '—'}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
