import { describe, it, expect } from 'vitest';
import { computeFeatures, FEATURE_NAMES } from '../lib/features';

const W = 100, H = 100, PC = 0.5;

function blankLandmarks(): { x: number; y: number; z: number }[] {
  return Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
}

describe('computeFeatures', () => {
  it('has the frozen 13-name order', () => {
    expect(FEATURE_NAMES).toEqual([
      'ear_left', 'ear_right', 'ear_mean', 'mar', 'brow_left', 'brow_right',
      'yaw', 'pitch', 'roll', 'gaze_x', 'gaze_y', 'face_area', 'face_present']);
  });

  it('missing face -> thirteen zeros, face_present 0', () => {
    const f = computeFeatures(null, W, H, PC);
    expect(Array.from(f)).toEqual(new Array(13).fill(0));
  });

  it('468 landmarks (no iris) -> treated as missing face', () => {
    const f = computeFeatures(blankLandmarks().slice(0, 468), W, H, PC);
    expect(f[12]).toBe(0);
  });

  it('computes EAR from the contract formula', () => {
    const lm = blankLandmarks();
    // left eye p1..p6 = [33,160,158,133,153,144]; craft |p2-p6|=|p3-p5|=0.02H, |p1-p4|=0.1W
    lm[33]  = { x: 0.30, y: 0.50, z: 0 }; lm[133] = { x: 0.40, y: 0.50, z: 0 };
    lm[160] = { x: 0.33, y: 0.49, z: 0 }; lm[144] = { x: 0.33, y: 0.51, z: 0 };
    lm[158] = { x: 0.37, y: 0.49, z: 0 }; lm[153] = { x: 0.37, y: 0.51, z: 0 };
    const f = computeFeatures(lm, W, H, PC);
    // EAR = (2 + 2) / (2 * 10) = 0.2
    expect(f[0]).toBeCloseTo(0.2, 5);
    expect(f[12]).toBe(1);
  });

  it('level eyes -> roll 0; symmetric nose -> yaw 0', () => {
    const lm = blankLandmarks();
    lm[33] = { x: 0.3, y: 0.5, z: 0 }; lm[263] = { x: 0.7, y: 0.5, z: 0 };
    lm[1] = { x: 0.5, y: 0.6, z: 0 };
    const f = computeFeatures(lm, W, H, PC);
    expect(f[8]).toBeCloseTo(0, 5);
    expect(f[6]).toBeCloseTo(0, 5);
  });

  it('pitch follows contract formula', () => {
    const lm = blankLandmarks();
    lm[33] = { x: 0.3, y: 0.4, z: 0 }; lm[263] = { x: 0.7, y: 0.4, z: 0 };
    lm[1] = { x: 0.5, y: 0.55, z: 0 }; lm[152] = { x: 0.5, y: 0.7, z: 0 };
    // pitch = (55-40)/|70-40| - 0.5 = 0.5 - 0.5 = 0
    const f = computeFeatures(lm, W, H, PC);
    expect(f[7]).toBeCloseTo(0, 4);
  });
});
