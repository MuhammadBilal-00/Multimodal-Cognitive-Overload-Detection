import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { STATE_CHANNELS, PYTHON_LABEL_COLS } from '../lib/states';

// The states head's channel order is decided in Python and is invisible from
// the browser side — nothing in the model file or scaler.json names the
// channels. Before this test the UI listed them alphabetically, which swapped
// indices 1 and 2 and made the "Confused" bar display P(engagement).
describe('state channel ordering', () => {
  it('matches the Python LABEL_COLS order', () => {
    expect(STATE_CHANNELS.map((s) => s.key)).toEqual(
      PYTHON_LABEL_COLS.map((c) => c.toLowerCase()));
  });

  it('is not alphabetical (the bug that was fixed)', () => {
    const keys = STATE_CHANNELS.map((s) => s.key);
    expect(keys).not.toEqual([...keys].sort());
  });

  // Reads the actual Python source so the two can never silently diverge:
  // if someone reorders LABEL_COLS in ml/src/labels.py, this fails.
  it('agrees with ml/src/labels.py at source level', () => {
    const labelsPy = readFileSync(
      join(__dirname, '..', '..', 'ml', 'src', 'labels.py'), 'utf8');
    const match = labelsPy.match(/^LABEL_COLS\s*=\s*\[([^\]]*)\]/m);
    expect(match, 'LABEL_COLS not found in ml/src/labels.py').toBeTruthy();
    const pythonOrder = Array.from(match![1].matchAll(/"([^"]+)"/g), (m) => m[1]);
    expect(pythonOrder).toEqual([...PYTHON_LABEL_COLS]);
  });
});
