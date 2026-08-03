import { describe, it, expect } from 'vitest';
import { validateScaler } from '../lib/scaler';
import { FEATURE_NAMES } from '../lib/features';

const good = {
  mean: new Array(13).fill(0), std: new Array(13).fill(1),
  feature_names: [...FEATURE_NAMES], pitch_centre: 0.5, version: '1.0',
};

describe('validateScaler', () => {
  it('accepts the contract schema', () => {
    expect(validateScaler(good).pitch_centre).toBe(0.5);
  });
  it('THROWS on feature_names mismatch', () => {
    const bad = { ...good, feature_names: [...FEATURE_NAMES].reverse() };
    expect(() => validateScaler(bad)).toThrow(/feature_names/);
  });
  it('throws on wrong lengths', () => {
    expect(() => validateScaler({ ...good, mean: [0] })).toThrow();
  });
});
