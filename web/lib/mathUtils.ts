export function softmax(x: Float32Array | number[]): number[] {
  const m = Math.max(...Array.from(x));
  const e = Array.from(x, (v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function sigmoid(x: Float32Array | number[]): number[] {
  return Array.from(x, (v) => 1 / (1 + Math.exp(-v)));
}

export function standardise(win: Float32Array, mean: number[], std: number[]): Float32Array {
  const out = new Float32Array(win.length);
  for (let i = 0; i < win.length; i++) {
    const f = i % 13;
    out[i] = (win[i] - mean[f]) / (std[f] || 1e-8);
  }
  return out;
}
