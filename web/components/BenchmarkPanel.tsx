'use client';
import { useState } from 'react';
import { runBenchmark, type BenchmarkResult } from '../lib/benchmark';

export default function BenchmarkPanel() {
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [running, setRunning] = useState(false);

  async function go() {
    setRunning(true);
    try {
      const r = await runBenchmark(300);
      setResult(r);
      const blob = new Blob([JSON.stringify(r, null, 2)], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `benchmark-${r.timestamp.replace(/[:.]/g, '-')}.json`;
      a.click();
      URL.revokeObjectURL(a.href);
    } finally {
      setRunning(false);
    }
  }

  return (
    <section className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-[var(--foreground-muted)]">Benchmark</h2>
        <button onClick={go} disabled={running}
          className="rounded-lg bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#1765cc] disabled:opacity-50">
          {running ? 'Running 300 cycles…' : 'Run 300 inferences'}
        </button>
      </div>
      {result && (
        <pre className="mt-4 whitespace-pre-wrap font-mono text-sm text-[var(--foreground-muted)]">
{`mean ${result.meanMs.toFixed(2)} ms   p50 ${result.p50.toFixed(2)}   p95 ${result.p95.toFixed(2)}   p99 ${result.p99.toFixed(2)}
${result.meanFps.toFixed(0)} inferences/s   heap Δ ${result.heapDeltaMB?.toFixed(2) ?? 'n/a'} MB`}
        </pre>
      )}
    </section>
  );
}
