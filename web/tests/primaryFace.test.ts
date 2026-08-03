import { describe, it, expect } from 'vitest';
import { faceStats, selectPrimaryFace, HYSTERESIS } from '../lib/primaryFace';
import type { Landmark } from '../lib/features';

// Square face of side `size` with top-left corner at (x, y).
function face(x: number, y: number, size: number): Landmark[] {
  return [
    { x, y, z: 0 },
    { x: x + size, y, z: 0 },
    { x, y: y + size, z: 0 },
    { x: x + size, y: y + size, z: 0 },
  ];
}

describe('faceStats', () => {
  it('computes bbox area and centroid from landmark extents', () => {
    const [s] = faceStats([face(0.1, 0.2, 0.4)]);
    expect(s.index).toBe(0);
    expect(s.area).toBeCloseTo(0.16, 6);
    expect(s.cx).toBeCloseTo(0.3, 6);
    expect(s.cy).toBeCloseTo(0.4, 6);
  });
});

describe('selectPrimaryFace', () => {
  it('returns null for no faces', () => {
    expect(selectPrimaryFace([], null)).toBeNull();
  });
  it('returns the only face', () => {
    expect(selectPrimaryFace([face(0.1, 0.1, 0.2)], null)?.index).toBe(0);
  });
  it('picks the larger face when there is no previous primary', () => {
    const p = selectPrimaryFace([face(0.1, 0.1, 0.2), face(0.5, 0.1, 0.4)], null);
    expect(p?.index).toBe(1);
  });
  it('keeps the incumbent when the challenger is under the hysteresis ratio', () => {
    const small = face(0.1, 0.1, 0.3); // area 0.09 at centroid (0.25, 0.25)
    const big = face(0.5, 0.1, 0.32);  // area ~0.102 < HYSTERESIS × 0.09
    const p = selectPrimaryFace([small, big], { cx: 0.25, cy: 0.25 });
    expect(p?.index).toBe(0);
  });
  it('lets a clearly larger face take over past the hysteresis ratio', () => {
    const small = face(0.1, 0.1, 0.2); // area 0.04
    const big = face(0.5, 0.1, 0.2 * Math.sqrt(HYSTERESIS) * 1.1);
    const p = selectPrimaryFace([small, big], { cx: 0.2, cy: 0.2 });
    expect(p?.index).toBe(1);
  });
  it('falls back to largest when prev is outside the stickiness radius', () => {
    const p = selectPrimaryFace(
      [face(0.05, 0.05, 0.2), face(0.6, 0.6, 0.3)],
      { cx: 0.99, cy: 0.01 },
    );
    expect(p?.index).toBe(1);
  });
  it('follows the same face across an array-order swap', () => {
    const a = face(0.1, 0.1, 0.3);
    const b = face(0.6, 0.6, 0.3);
    const first = selectPrimaryFace([a, b], { cx: 0.25, cy: 0.25 });
    expect(first?.index).toBe(0);
    const swapped = selectPrimaryFace([b, a], { cx: first!.cx, cy: first!.cy });
    expect(swapped?.index).toBe(1); // same face a, now at index 1
  });
});
