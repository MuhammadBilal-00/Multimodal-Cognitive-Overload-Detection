import { describe, it, expect } from 'vitest';
import { softmax, sigmoid, standardise, distributionCheck } from '../lib/mathUtils';

// Builds a 30x13 standardised window. `z` fills every real feature;
// face_present (index 12) is set to 1 on the first `presentFrames` frames and
// 0 after, which is what the scaler's identity mean/std produces.
function window(z: number, presentFrames = 30, feature?: number): Float32Array {
  const w = new Float32Array(390);
  for (let t = 0; t < 30; t++) {
    const present = t < presentFrames;
    for (let f = 0; f < 12; f++) {
      w[t * 13 + f] = present && (feature === undefined || feature === f) ? z : 0;
    }
    w[t * 13 + 12] = present ? 1 : 0;
  }
  return w;
}

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

  describe('distributionCheck', () => {
    it('a window at the training mean is in-distribution', () => {
      const r = distributionCheck(window(0));
      expect(r.noFace).toBe(false);
      expect(r.outOfDistribution).toBe(false);
      expect(r.sustainedSigma).toBeCloseTo(0, 6);
    });

    it('ignores face_present, which is 1 whenever a face is there', () => {
      // Every real feature at the mean, face_present at 1 for all 30 frames.
      expect(distributionCheck(window(0)).sustainedSigma).toBe(0);
    });

    it('flags a sustained offset and names the offending feature', () => {
      const FACE_AREA = 11;
      const r = distributionCheck(window(6, 30, FACE_AREA));
      expect(r.outOfDistribution).toBe(true);
      expect(r.offenders[0].feature).toBe(FACE_AREA);
      expect(r.offenders[0].sigma).toBeCloseTo(6, 6);
    });

    it('does not flag a brief transient (median, not max)', () => {
      // One extreme frame out of 30 — a yawn, not a framing problem.
      const w = window(0);
      for (let f = 0; f < 12; f++) w[3 * 13 + f] = 12;
      const r = distributionCheck(w);
      expect(r.outOfDistribution).toBe(false);
    });

    // Regression guard. Missing-face frames are all-zero by CONTRACT §2.1,
    // and a raw 0 does NOT standardise to ~0 — it lands at -8.93σ on
    // brow_left for the shipped scaler. Scanning those frames would pin the
    // OOD warning on permanently the moment anyone stepped out of frame.
    it('reports no-face rather than OOD when the face is gone', () => {
      const w = window(0, 0);
      for (let t = 0; t < 30; t++) {
        w[t * 13 + 4] = -8.93; // brow_left, as a real zeroed frame scales to
        w[t * 13 + 5] = -9.06; // brow_right
      }
      const r = distributionCheck(w);
      expect(r.noFace).toBe(true);
      expect(r.outOfDistribution).toBe(false);
      expect(r.facePresentFrames).toBe(0);
    });

    it('still reports no-face when under half the window has a face', () => {
      expect(distributionCheck(window(0, 14)).noFace).toBe(true);
      expect(distributionCheck(window(0, 15)).noFace).toBe(false);
    });
  });
});
