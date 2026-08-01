/**
 * Track B port of ml/src/features.py — THE REFERENCE IS THE PYTHON FILE.
 * Ported line by line; if this file and features.py disagree, features.py
 * and CONTRACT.md win.
 *
 * CONVENTIONS (FROZEN, CONTRACT.md §4):
 * 1. COORDINATES ARE PIXELS. Convert MediaPipe's normalised output first:
 *      x_px = x_norm * frameWidth
 *      y_px = y_norm * frameHeight
 *      z_px = z_norm * frameWidth
 *    (helper: landmarksToPixels)
 * 2. frameShape is [HEIGHT, WIDTH].
 * 3. "LEFT" = IMAGE-LEFT in an UN-MIRRORED frame. Never feed the
 *    landmarker a mirrored (selfie-view) canvas — mirror for display only.
 * 4. y grows downward. Positive gazeX = iris toward image-right.
 * 5. No face -> thirteen zeros with facePresent = 0.0. Never interpolate,
 *    never drop the frame.
 */

export const LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144];   // p1..p6
export const RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]; // p1..p6
export const MOUTH = [61, 291, 13, 14];                      // L, R, up, low
export const LEFT_BROW = [70, 63, 105, 66, 107];
export const RIGHT_BROW = [300, 293, 334, 296, 336];
export const INTEROCULAR = [33, 263];                        // outer corners
export const NOSE_TIP = 1;
export const CHIN = 152;
export const LEFT_IRIS = [468, 469, 470, 471, 472];
export const RIGHT_IRIS = [473, 474, 475, 476, 477];

export const FEATURE_NAMES = [
  "ear_left", "ear_right", "ear_mean", "mar",
  "brow_left", "brow_right", "yaw", "pitch", "roll",
  "gaze_x", "gaze_y", "face_area", "face_present",
];

const EPS = 1e-8;

/** MediaPipe result landmarks (normalised {x,y,z}) -> [[x,y,z] px] * 478. */
export function landmarksToPixels(faceLandmarks, width, height) {
  const out = [];
  for (const p of faceLandmarks) {
    out.push([p.x * width, p.y * height, p.z * width]);
  }
  return out;
}

function distance(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return Math.sqrt(dx * dx + dy * dy);
}

function midpoint(a, b) {
  return [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
}

function meanPoint(landmarks, indices) {
  let sumX = 0.0;
  let sumY = 0.0;
  for (const idx of indices) {
    sumX += landmarks[idx][0];
    sumY += landmarks[idx][1];
  }
  return [sumX / indices.length, sumY / indices.length];
}

export function eyeAspectRatio(landmarks, eyeIndices) {
  const p1 = landmarks[eyeIndices[0]];
  const p2 = landmarks[eyeIndices[1]];
  const p3 = landmarks[eyeIndices[2]];
  const p4 = landmarks[eyeIndices[3]];
  const p5 = landmarks[eyeIndices[4]];
  const p6 = landmarks[eyeIndices[5]];
  const vertical = distance(p2, p6) + distance(p3, p5);
  const horizontal = distance(p1, p4);
  return vertical / (2.0 * horizontal + EPS);
}

export function mouthAspectRatio(landmarks) {
  const cornerLeft = landmarks[MOUTH[0]];
  const cornerRight = landmarks[MOUTH[1]];
  const lipUpper = landmarks[MOUTH[2]];
  const lipLower = landmarks[MOUTH[3]];
  const vertical = distance(lipUpper, lipLower);
  const horizontal = distance(cornerLeft, cornerRight);
  return vertical / (horizontal + EPS);
}

export function browRatio(landmarks, browIndices, eyeIndices, interocular) {
  const browCentre = meanPoint(landmarks, browIndices);
  const eyeCentre = midpoint(landmarks[eyeIndices[0]], landmarks[eyeIndices[3]]);
  const dx = browCentre[0] - eyeCentre[0];
  const dy = browCentre[1] - eyeCentre[1];
  const dist = Math.sqrt(dx * dx + dy * dy);
  return dist / (interocular + EPS);
}

export function headRoll(landmarks) {
  const leftEyeOuter = landmarks[INTEROCULAR[0]];
  const rightEyeOuter = landmarks[INTEROCULAR[1]];
  const dy = rightEyeOuter[1] - leftEyeOuter[1];
  const dx = rightEyeOuter[0] - leftEyeOuter[0];
  return Math.atan2(dy, dx);
}

export function headYaw(landmarks) {
  const nose = landmarks[NOSE_TIP];
  const leftEyeOuter = landmarks[INTEROCULAR[0]];
  const rightEyeOuter = landmarks[INTEROCULAR[1]];
  const dLeft = distance(nose, leftEyeOuter);
  const dRight = distance(nose, rightEyeOuter);
  return (dLeft - dRight) / (dLeft + dRight + EPS);
}

export function headPitch(landmarks, pitchCentre) {
  const nose = landmarks[NOSE_TIP];
  const chin = landmarks[CHIN];
  const leftEyeOuter = landmarks[INTEROCULAR[0]];
  const rightEyeOuter = landmarks[INTEROCULAR[1]];
  const eyeMid = midpoint(leftEyeOuter, rightEyeOuter);
  const raw = (nose[1] - eyeMid[1])
    / (Math.abs(chin[1] - eyeMid[1]) + EPS);
  return raw - pitchCentre;
}

export function eyeGaze(landmarks, eyeIndices, irisIndices) {
  const p1 = landmarks[eyeIndices[0]];
  const p2 = landmarks[eyeIndices[1]];
  const p3 = landmarks[eyeIndices[2]];
  const p4 = landmarks[eyeIndices[3]];
  const p5 = landmarks[eyeIndices[4]];
  const p6 = landmarks[eyeIndices[5]];
  const eyeCentre = midpoint(p1, p4);
  const eyeWidth = distance(p1, p4);
  const eyeHeight = (distance(p2, p6) + distance(p3, p5)) / 2.0;
  const irisCentre = meanPoint(landmarks, irisIndices);
  const gazeX = (irisCentre[0] - eyeCentre[0]) / (eyeWidth + EPS);
  const gazeY = (irisCentre[1] - eyeCentre[1]) / (eyeHeight + EPS);
  return [gazeX, gazeY];
}

export function faceBboxAreaRatio(landmarks, frameShape) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of landmarks) {
    if (p[0] < minX) minX = p[0];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] < minY) minY = p[1];
    if (p[1] > maxY) maxY = p[1];
  }
  const bboxArea = (maxX - minX) * (maxY - minY);
  const frameArea = frameShape[0] * frameShape[1];
  return bboxArea / (frameArea + EPS);
}

/**
 * The contract function. landmarks: 478 x [x,y,z] in PIXELS, or null.
 * frameShape: [HEIGHT, WIDTH]. Returns Float32Array(13), order per
 * FEATURE_NAMES / CONTRACT.md §2.
 */
export function computeFeatures(landmarks, frameShape, pitchCentre = 0.0) {
  if (landmarks === null || landmarks === undefined) {
    return new Float32Array(13);
  }

  const interocular = distance(landmarks[INTEROCULAR[0]],
                               landmarks[INTEROCULAR[1]]);

  const earLeft = eyeAspectRatio(landmarks, LEFT_EYE_EAR);
  const earRight = eyeAspectRatio(landmarks, RIGHT_EYE_EAR);
  const earMean = (earLeft + earRight) / 2.0;

  const mar = mouthAspectRatio(landmarks);

  const browLeft = browRatio(landmarks, LEFT_BROW, LEFT_EYE_EAR, interocular);
  const browRight = browRatio(landmarks, RIGHT_BROW, RIGHT_EYE_EAR, interocular);

  const yaw = headYaw(landmarks);
  const pitch = headPitch(landmarks, pitchCentre);
  const roll = headRoll(landmarks);

  const gazeLeft = eyeGaze(landmarks, LEFT_EYE_EAR, LEFT_IRIS);
  const gazeRight = eyeGaze(landmarks, RIGHT_EYE_EAR, RIGHT_IRIS);
  const gazeX = (gazeLeft[0] + gazeRight[0]) / 2.0;
  const gazeY = (gazeLeft[1] + gazeRight[1]) / 2.0;

  const faceArea = faceBboxAreaRatio(landmarks, frameShape);

  return new Float32Array([
    earLeft, earRight, earMean, mar,
    browLeft, browRight, yaw, pitch, roll,
    gazeX, gazeY, faceArea, 1.0,
  ]);
}
