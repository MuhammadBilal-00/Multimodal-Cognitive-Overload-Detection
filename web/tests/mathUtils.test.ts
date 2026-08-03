import { describe, it, expect } from 'vitest';
import { softmax, sigmoid, standardise } from '../lib/mathUtils';

describe('mathUtils', () => {
  it('softmax sums to 1 and orders correctly', () => {
    const p = softmax([1, 2, 3, 4]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(p[3]).toBeGreaterThan(p[0]);
  });
  it('sigmoid(0) = 0.5', () => {
    expect(sigmoid([0])[0]).toBeCloseTo(0.5, 6);
  });
  it('standardise applies (x-mean)/std element-wise per feature', () => {
    const win = new Float32Array(390).fill(2);
    const mean = new Array(13).fill(1), std = new Array(13).fill(2);
    const s = standardise(win, mean, std);
    expect(s[0]).toBeCloseTo(0.5, 6);
    expect(s[389]).toBeCloseTo(0.5, 6);
  });
});
