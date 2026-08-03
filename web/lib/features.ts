// Port target: ml/src/features.py (CONTRACT.md sections 2-4). Conventions:
// PIXEL coords, IMAGE-LEFT naming, un-mirrored frames, y grows downward.
export const FEATURE_NAMES = [
  'ear_left', 'ear_right', 'ear_mean', 'mar', 'brow_left', 'brow_right',
  'yaw', 'pitch', 'roll', 'gaze_x', 'gaze_y', 'face_area', 'face_present',
] as const;

const LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144];   // p1..p6
const RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]; // p1..p6
const MOUTH = [61, 291, 13, 14];                      // left, right, upper, lower
const LEFT_BROW = [70, 63, 105, 66, 107];
const RIGHT_BROW = [300, 293, 334, 296, 336];
const INTEROCULAR = [33, 263];
const NOSE_TIP = 1;
const CHIN = 152;
const LEFT_IRIS = [468, 469, 470, 471, 472];
const RIGHT_IRIS = [473, 474, 475, 476, 477];
const EPS = 1e-8;

export interface Landmark { x: number; y: number; z: number }
interface Pt { x: number; y: number }

const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);
const mean = (pts: Pt[]): Pt => ({
  x: pts.reduce((s, p) => s + p.x, 0) / pts.length,
  y: pts.reduce((s, p) => s + p.y, 0) / pts.length,
});

function ear(px: Pt[], idx: number[]): number {
  const [p1, p2, p3, p4, p5, p6] = idx.map((i) => px[i]);
  return (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4) + EPS);
}

function gaze(px: Pt[], iris: number[], outer: number, inner: number, earIdx: number[]) {
  const centre = mean(iris.map((i) => px[i]));
  const c1 = px[outer], c2 = px[inner];
  const mid = mean([c1, c2]);
  const width = dist(c1, c2);
  // J1-CHECK: eye height defined as mean of the two vertical EAR distances.
  const [, p2, p3, , p5, p6] = earIdx.map((i) => px[i]);
  const height = (dist(p2, p6) + dist(p3, p5)) / 2;
  return { gx: (centre.x - mid.x) / (width + EPS), gy: (centre.y - mid.y) / (height + EPS) };
}

export function computeFeatures(
  landmarks: Landmark[] | null | undefined,
  frameWidth: number,
  frameHeight: number,
  pitchCentre: number,
): Float32Array {
  const out = new Float32Array(13); // all zeros, face_present = 0
  if (!landmarks || landmarks.length < 478) return out;

  const px: Pt[] = landmarks.map((l) => ({ x: l.x * frameWidth, y: l.y * frameHeight }));

  const leftEyeOuter = px[INTEROCULAR[0]];
  const rightEyeOuter = px[INTEROCULAR[1]];
  const interocular = dist(leftEyeOuter, rightEyeOuter) + EPS;

  const earLeft = ear(px, LEFT_EYE_EAR);
  const earRight = ear(px, RIGHT_EYE_EAR);

  const [mL, mR, mU, mD] = MOUTH.map((i) => px[i]);
  const mar = dist(mU, mD) / (dist(mL, mR) + EPS);

  // J1-CHECK: brow = distance(mean of 5 brow pts, mean of 6 eye pts) / interocular.
  const browLeft = dist(mean(LEFT_BROW.map((i) => px[i])), mean(LEFT_EYE_EAR.map((i) => px[i]))) / interocular;
  const browRight = dist(mean(RIGHT_BROW.map((i) => px[i])), mean(RIGHT_EYE_EAR.map((i) => px[i]))) / interocular;

  const nose = px[NOSE_TIP];
  const chin = px[CHIN];
  const roll = Math.atan2(rightEyeOuter.y - leftEyeOuter.y, rightEyeOuter.x - leftEyeOuter.x);
  const dL = dist(nose, leftEyeOuter);
  const dR = dist(nose, rightEyeOuter);
  const yaw = (dL - dR) / (dL + dR + EPS);
  const eyeMid = mean([leftEyeOuter, rightEyeOuter]);
  const pitch = (nose.y - eyeMid.y) / (Math.abs(chin.y - eyeMid.y) + EPS) - pitchCentre;

  const gL = gaze(px, LEFT_IRIS, 33, 133, LEFT_EYE_EAR);
  const gR = gaze(px, RIGHT_IRIS, 263, 362, RIGHT_EYE_EAR);
  const gazeX = (gL.gx + gR.gx) / 2;
  const gazeY = (gL.gy + gR.gy) / 2;

  // J1-CHECK: face bbox from landmark extremes (pixels) / frame area.
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of px) {
    if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
  }
  const faceArea = ((maxX - minX) * (maxY - minY)) / (frameWidth * frameHeight);

  out.set([earLeft, earRight, (earLeft + earRight) / 2, mar, browLeft, browRight,
    yaw, pitch, roll, gazeX, gazeY, faceArea, 1.0]);
  return out;
}
