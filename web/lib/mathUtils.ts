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

export interface OodReport {
  /** Frames in the window where a face was actually detected (0..30). */
  facePresentFrames: number;
  /** Too few usable frames to say anything about the distribution. */
  noFace: boolean;
  /** Largest per-feature MEDIAN |z| across face-present frames. */
  sustainedSigma: number;
  /** Worst-offending feature indices, largest first (max 2). */
  offenders: { feature: number; sigma: number }[];
  outOfDistribution: boolean;
}

const FRAME = 13;
const FACE_PRESENT = 12;

function median(sorted: number[]): number {
  if (!sorted.length) return 0;
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// How far the current window sits from the distribution the scaler was fitted
// on, reported rather than corrected.
//
// Deliberately NOT clamping. Clamping would silently alter inference against a
// frozen contract (CONTRACT §5 says the input is "already standardised"), was
// never done at training time, sits downstream of the J1 parity gate so no
// existing test would cover it, and wouldn't even help — a +13σ input pinned
// back to +5σ still saturates the states head. It would just make a wrong
// number look trustworthy.
//
// Two traps this has to avoid:
//
// 1. Missing-face frames are all-zero by contract (§2.1). A raw 0 is NOT
//    near the mean once standardised — for the shipped scaler it lands at
//    -8.93σ on brow_left and -9.06σ on brow_right. Scanning them would flag
//    OOD permanently the moment anyone steps out of frame. So face-absent
//    frames are excluded and reported separately as `noFace`.
// 2. A per-window max fires on one yawn. The symptom being targeted is a
//    *persistent* framing offset (sitting closer than a DAiSEE participant,
//    or a camera angle unlike DAiSEE's laptop framing), so this uses the
//    per-feature median over face-present frames. The features are also
//    distinctly non-Gaussian — `mar` has a hard floor at 0, only -0.57σ
//    below its own mean — which a max-based rule handles badly.
//
// Feature 12 (face_present) is skipped: the scaler gives it identity mean 0 /
// std 1 (ml/src/dataset.py fit_scaler), so it reads as exactly 1σ whenever a
// face is there. That is not a distribution shift, and including it would put
// a permanent floor under the metric.
//
// Note for anyone tempted to lower the threshold: brow_left/brow_right carry
// a known systematic JS-vs-Python offset of ~0.011 mean-abs
// (docs/results/parity_report.json), which against std 0.0279 is ~0.39σ — an
// order of magnitude worse than any other feature. Far below 3σ, so it won't
// cause false trips here, but it would be the first thing to misfire.
export function distributionCheck(
  standardised: Float32Array,
  sigmaThreshold = 3,
  minFaceFrames = 15,
): OodReport {
  const frames = standardised.length / FRAME;
  const perFeature: number[][] = Array.from({ length: FRAME }, () => []);
  let facePresentFrames = 0;

  for (let t = 0; t < frames; t++) {
    const base = t * FRAME;
    if (standardised[base + FACE_PRESENT] !== 1) continue;
    facePresentFrames++;
    for (let f = 0; f < FRAME; f++) {
      if (f === FACE_PRESENT) continue;
      perFeature[f].push(Math.abs(standardised[base + f]));
    }
  }

  if (facePresentFrames < minFaceFrames) {
    return {
      facePresentFrames, noFace: true, sustainedSigma: 0,
      offenders: [], outOfDistribution: false,
    };
  }

  const sustained = perFeature.map((vals, f) =>
    f === FACE_PRESENT ? 0 : median(vals.sort((a, b) => a - b)));
  const offenders = sustained
    .map((sigma, feature) => ({ feature, sigma }))
    .sort((a, b) => b.sigma - a.sigma)
    .slice(0, 2);
  const sustainedSigma = offenders.length ? offenders[0].sigma : 0;

  return {
    facePresentFrames,
    noFace: false,
    sustainedSigma,
    offenders,
    outOfDistribution: sustainedSigma >= sigmaThreshold,
  };
}
