// Source-level contract guards, in the same spirit as states.test.ts (which
// parses ml/src/labels.py as text): lib/inference.ts loads onnxruntime-web
// via a runtime browser import ('/ort/ort.wasm.min.mjs') that cannot be
// mocked in node, and lib/faceLandmarker.ts needs MediaPipe's WASM fileset —
// so the load-bearing constants in both are pinned here by reading the
// source. Crude, but it makes the following silent failures loud:
//   - swapping softmax/sigmoid between the engagement and states heads
//     (every numeric test would still pass — the audit's exact observation)
//   - changing the tensor shape or input/output tensor names off contract §5
//   - lifting createFeatureLandmarker off numFaces:1 or off the CPU delegate
//     (both empirically load-bearing: CONTRACT.md Amendment 2, gaze_y 0.86
//     at numFaces:4; parity_report_gpu.json fails the gate on GPU)
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const read = (p: string) => readFileSync(join(__dirname, '..', p), 'utf8');

describe('inference.ts contract constants (CONTRACT.md §5)', () => {
  const src = read('lib/inference.ts');

  it('applies softmax to the engagement head and sigmoid to the states head', () => {
    expect(src).toMatch(/engagement:\s*softmax\(out\.engagement/);
    expect(src).toMatch(/states:\s*sigmoid\(out\.states/);
    // and never the other way around
    expect(src).not.toMatch(/engagement:\s*sigmoid\(/);
    expect(src).not.toMatch(/states:\s*softmax\(/);
  });

  it('builds a [1, 30, 13] float32 tensor named "features"', () => {
    expect(src).toMatch(/'float32',\s*\w+,\s*\[1,\s*30,\s*13\]/);
    expect(src).toMatch(/\{\s*features:\s*tensor\s*\}/);
  });
});

describe('faceLandmarker.ts config invariants (CONTRACT.md Amendment 2)', () => {
  const src = read('lib/faceLandmarker.ts');

  it('feature landmarker is numFaces 1, display landmarker numFaces 4', () => {
    expect(src).toMatch(
      /createFeatureLandmarker\(\)[\s\S]{0,80}createFaceLandmarker\(1\)/);
    expect(src).toMatch(
      /createDisplayLandmarker\(\)[\s\S]{0,80}createFaceLandmarker\(4\)/);
  });

  it('pins the CPU delegate and the self-hosted model asset', () => {
    expect(src).toMatch(/delegate:\s*'CPU'/);
    expect(src).not.toMatch(/delegate:\s*'GPU'/);
    expect(src).toMatch(/'\/models\/face_landmarker\.task'/);
  });
});
