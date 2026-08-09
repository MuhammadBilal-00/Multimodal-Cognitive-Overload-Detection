import { initInference, runInference } from './inference';
import { loadScaler } from './scaler';

export interface BenchmarkResult {
  cycles: number; meanMs: number; p50: number; p95: number; p99: number;
  meanFps: number; heapDeltaMB: number | null;
  backend: string; threads: number; userAgent: string; timestamp: string;
  // Rough hardware hints, readable only from inside the page — a
  // Playwright driver collecting this JSON externally can't inject these
  // after the fact. CPU model / RAM amount still need a human-supplied
  // --machine-label; these are proxies, not exact hardware identification.
  hardwareConcurrency: number;
  deviceMemory: number | null;
}

const pct = (sorted: number[], p: number) =>
  sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];

export async function runBenchmark(cycles = 300): Promise<BenchmarkResult> {
  const info = await initInference();
  const scaler = await loadScaler();
  const win = new Float32Array(390).map(() => Math.random() * 2 - 1);

  const mem = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
  const heapBefore = mem?.usedJSHeapSize ?? null;

  for (let i = 0; i < 10; i++) await runInference(win, scaler); // warm-up

  const times: number[] = [];
  for (let i = 0; i < cycles; i++) times.push((await runInference(win, scaler)).ms);

  const heapAfter = mem?.usedJSHeapSize ?? null;
  const sorted = [...times].sort((a, b) => a - b);
  const meanMs = times.reduce((a, b) => a + b, 0) / times.length;
  return {
    cycles, meanMs, p50: pct(sorted, 50), p95: pct(sorted, 95), p99: pct(sorted, 99),
    meanFps: 1000 / meanMs,
    heapDeltaMB: heapBefore != null && heapAfter != null ? (heapAfter - heapBefore) / 1048576 : null,
    backend: info.backend, threads: info.threads,
    userAgent: navigator.userAgent, timestamp: new Date().toISOString(),
    hardwareConcurrency: navigator.hardwareConcurrency,
    deviceMemory: (navigator as unknown as { deviceMemory?: number }).deviceMemory ?? null,
  };
}
