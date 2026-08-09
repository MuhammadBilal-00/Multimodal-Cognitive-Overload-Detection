// J1 feature-parity gate. Reruns web/lib/features.ts against the 100-frame
// DAiSEE fixture (ml/scripts/make_parity_fixture.py) via the /parity-test
// route and asserts it matches ml/src/features.py (Python reference)
// within CONTRACT.md Amendment 2's tolerance (0.02, max-abs-diff).
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { expect, test } from '@playwright/test';
import type { ParityResult } from '../../app/parity-test/page';

const REPORT_PATH = path.resolve(__dirname, '../../../docs/results/parity_report.json');

test('features.ts matches features.py within tolerance', async ({ page }) => {
  await page.goto('/parity-test');

  await page.waitForFunction(
    () => (window as unknown as { __PARITY_RESULT__?: unknown }).__PARITY_RESULT__ !== undefined,
    { timeout: 240_000 },
  );
  const result = await page.evaluate(
    () => (window as unknown as { __PARITY_RESULT__: ParityResult }).__PARITY_RESULT__,
  );

  mkdirSync(path.dirname(REPORT_PATH), { recursive: true });
  writeFileSync(REPORT_PATH, JSON.stringify(result, null, 1));

  expect(result.error, `parity harness threw: ${result.error}`).toBeNull();
  expect(
    result.framesCompared,
    `only ${result.framesCompared}/${result.numFrames} frames were comparable`,
  ).toBeGreaterThanOrEqual(Math.floor(result.numFrames * 0.95));
  expect(
    result.faceMismatchFrames,
    'too many face-presence disagreements vs the Python reference',
  ).toBeLessThanOrEqual(Math.ceil(result.numFrames * 0.02));
  expect(
    result.worstFeatureDiff,
    `worst per-feature diff ${result.worstFeatureDiff} >= tolerance ${result.tolerance}: ` +
      JSON.stringify(result.perFeatureMaxAbsDiff, null, 1),
  ).toBeLessThan(result.tolerance);
  expect(result.ok).toBe(true);
});
