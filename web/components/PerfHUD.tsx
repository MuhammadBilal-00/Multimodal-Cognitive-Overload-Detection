'use client';

function Stat({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-widest text-[var(--foreground-muted)]">{label}</div>
      <div className="text-2xl font-bold tabular-nums text-[var(--foreground)]">
        {value}<span className="ml-1 text-sm font-normal text-[var(--foreground-muted)]">{unit}</span>
      </div>
    </div>
  );
}

export default function PerfHUD({ perf }: {
  perf: { renderFps: number; sampleHz: number; inferMs: number[]; modelBytes: number; backend: string; threads: number; landmarkCount: number; faceCount: number };
}) {
  const meanMs = perf.inferMs.length
    ? perf.inferMs.reduce((a, b) => a + b, 0) / perf.inferMs.length : 0;
  return (
    <section className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-6 shadow-sm">
      <h2 className="mb-4 text-xs font-semibold uppercase tracking-widest text-[var(--foreground-muted)]">
        Performance
      </h2>
      <div className="grid grid-cols-3 gap-x-4 gap-y-5 md:grid-cols-4">
        <Stat label="Render" value={String(perf.renderFps)} unit="fps" />
        <Stat label="Sampling" value={String(perf.sampleHz)} unit="Hz" />
        <Stat label="Inference" value={meanMs.toFixed(1)} unit="ms" />
        <Stat label="Model" value={(perf.modelBytes / 1024).toFixed(0)} unit="KB" />
        <Stat label="Backend" value={`${perf.backend}×${perf.threads}`} />
        <Stat label="Landmarks" value={String(perf.landmarkCount)} />
        <Stat label="People" value={String(perf.faceCount)} />
      </div>
    </section>
  );
}
