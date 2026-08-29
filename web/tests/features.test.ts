import { describe, it, expect } from 'vitest';
import {
  computeFeatures, FEATURE_NAMES, LEFT_BROW, LEFT_EYE_EAR, RIGHT_EYE_EAR,
  MOUTH, LEFT_IRIS,
} from '../lib/features';

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

  // Port of ml/tests/test_features.py::test_brow_left — same fixture, same
  // expected value, so the two implementations are pinned to the same number.
  it('brow_left matches the Python reference fixture (3/10 = 0.3)', () => {
    const lm = blankLandmarks();
    const [p1, p2, p3, p4, p5, p6] = LEFT_EYE_EAR;
    // set_left_eye_open (pixels /100 -> normalised): corners (0,0) and (4,0)
    lm[p1] = { x: 0.00, y: 0.00, z: 0 }; lm[p4] = { x: 0.04, y: 0.00, z: 0 };
    lm[p2] = { x: 0.01, y: -0.01, z: 0 }; lm[p6] = { x: 0.01, y: 0.01, z: 0 };
    lm[p3] = { x: 0.03, y: -0.01, z: 0 }; lm[p5] = { x: 0.03, y: 0.01, z: 0 };
    lm[263] = { x: 0.10, y: 0.00, z: 0 }; // interocular = 10 px
    for (const idx of LEFT_BROW) lm[idx] = { x: 0.02, y: -0.03, z: 0 };
    const f = computeFeatures(lm, W, H, PC);
    expect(f[4]).toBeCloseTo(0.3, 5);
  });

  // Regression test for the Amendment 4 brow bug. The Python fixture above is
  // vertically symmetric, so the corner MIDPOINT (contract) and the centroid
  // of all six EAR landmarks (the old, wrong implementation) coincide on it —
  // it cannot distinguish the two. This fixture makes the lids asymmetric:
  // corner midpoint stays (2,0); the 6-point centroid moves to (2,-1/3).
  // Contract answer: |(2,-3)-(2,0)| / 10 = 0.3. Old code gave 8/3/10 ≈ 0.2667.
  it('brow eye-centre is the CORNER MIDPOINT, not the 6-landmark centroid', () => {
    const lm = blankLandmarks();
    const [p1, p2, p3, p4, p5, p6] = LEFT_EYE_EAR;
    lm[p1] = { x: 0.00, y: 0.00, z: 0 }; lm[p4] = { x: 0.04, y: 0.00, z: 0 };
    lm[p2] = { x: 0.01, y: -0.02, z: 0 }; lm[p3] = { x: 0.03, y: -0.02, z: 0 };
    lm[p5] = { x: 0.03, y: 0.01, z: 0 }; lm[p6] = { x: 0.01, y: 0.01, z: 0 };
    lm[263] = { x: 0.10, y: 0.00, z: 0 };
    for (const idx of LEFT_BROW) lm[idx] = { x: 0.02, y: -0.03, z: 0 };
    const f = computeFeatures(lm, W, H, PC);
    expect(f[4]).toBeCloseTo(0.3, 5);
    expect(f[4]).not.toBeCloseTo(8 / 3 / 10, 3); // the old centroid answer
  });

  // Port of ml/tests/test_features.py::test_mar — same fixture, same value.
  it('mar = vertical / horizontal lip distance (2/4 = 0.5)', () => {
    const lm = blankLandmarks();
    const [left, right, upper, lower] = MOUTH;
    lm[left] = { x: 0.00, y: 0.00, z: 0 }; lm[right] = { x: 0.04, y: 0.00, z: 0 };
    lm[upper] = { x: 0.02, y: -0.01, z: 0 }; lm[lower] = { x: 0.02, y: 0.01, z: 0 };
    const f = computeFeatures(lm, W, H, PC);
    expect(f[3]).toBeCloseTo(0.5, 5);
  });

  it('ear_mean is the mean of the two EARs', () => {
    const lm = blankLandmarks();
    // Both eyes get the Python set_left_eye_open geometry (EAR = 0.5),
    // offset apart so the two eyes don't overlap.
    const place = (idx: number[], ox: number) => {
      const [p1, p2, p3, p4, p5, p6] = idx;
      lm[p1] = { x: ox + 0.00, y: 0.50, z: 0 }; lm[p4] = { x: ox + 0.04, y: 0.50, z: 0 };
      lm[p2] = { x: ox + 0.01, y: 0.49, z: 0 }; lm[p6] = { x: ox + 0.01, y: 0.51, z: 0 };
      lm[p3] = { x: ox + 0.03, y: 0.49, z: 0 }; lm[p5] = { x: ox + 0.03, y: 0.51, z: 0 };
    };
    place(LEFT_EYE_EAR, 0.2);
    place(RIGHT_EYE_EAR, 0.6);
    const f = computeFeatures(lm, W, H, PC);
    expect(f[0]).toBeCloseTo(0.5, 5);
    expect(f[1]).toBeCloseTo(0.5, 5);
    expect(f[2]).toBeCloseTo((f[0] + f[1]) / 2, 6);
  });

  it('gaze offsets are normalised by eye width/height and averaged over eyes', () => {
    const lm = blankLandmarks();
    // Left eye: corners 33=(0,0) and 133=(4,0) -> mid (2,0), width 4.
    // Lids at (1,±1)/(3,±1) -> eye height = (2+2)/2 = 2.
    const [p1, p2, p3, p4, p5, p6] = LEFT_EYE_EAR;
    lm[p1] = { x: 0.00, y: 0.00, z: 0 }; lm[p4] = { x: 0.04, y: 0.00, z: 0 };
    lm[p2] = { x: 0.01, y: -0.01, z: 0 }; lm[p6] = { x: 0.01, y: 0.01, z: 0 };
    lm[p3] = { x: 0.03, y: -0.01, z: 0 }; lm[p5] = { x: 0.03, y: 0.01, z: 0 };
    // Left iris entirely at (3, 0.5): gx = (3-2)/4 = 0.25, gy = 0.5/2 = 0.25.
    for (const idx of LEFT_IRIS) lm[idx] = { x: 0.03, y: 0.005, z: 0 };
    // Right eye left at the blank (0.5,0.5) point: zero width/height ->
    // 0/EPS = 0 contribution, so the two-eye average halves the left values.
    for (const idx of RIGHT_EYE_EAR) lm[idx] = { x: 0.5, y: 0.5, z: 0 };
    const f = computeFeatures(lm, W, H, PC);
    expect(f[9]).toBeCloseTo(0.125, 5);  // gaze_x
    expect(f[10]).toBeCloseTo(0.125, 5); // gaze_y
  });

  it('face_area = landmark bbox area / frame area', () => {
    const lm = blankLandmarks().map(() => ({ x: 0.4, y: 0.4, z: 0 }));
    lm[10] = { x: 0.6, y: 0.7, z: 0 }; // bbox 0.2W x 0.3H = 600 px^2 / 10000
    const f = computeFeatures(lm, W, H, PC);
    expect(f[11]).toBeCloseTo(0.06, 5);
  });
});
