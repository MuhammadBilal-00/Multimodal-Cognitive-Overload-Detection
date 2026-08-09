import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

async function createFaceLandmarker(numFaces: number): Promise<FaceLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
  // CPU delegate only: the model was trained on features extracted with the
  // CPU/XNNPACK landmarker, and the GPU delegate shifts landmarks enough to
  // fail the J1 parity gate (docs/results/parity_report_gpu.json — gaze_y
  // 0.05 > 0.02 tol; CPU worst-case 0.0079).
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate: 'CPU' },
    runningMode: 'VIDEO' as const,
    numFaces,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false, // contract 3.2: never use these
  });
}

// Feeds computeFeatures(). CONTRACT.md's feature vector must be computed
// the same way the Python training pipeline extracted it, and
// ml/src/extract.py always runs num_faces=1. numFaces > 1 measurably
// shifts landmarks even for the one real face in frame — enough to fail
// the J1 parity gate on blink frames (gaze_y worst-case 0.86 at numFaces:4
// vs 0.016 at numFaces:1; CONTRACT.md Amendment 2). Never feed this
// landmarker's output into anything display-only; use
// createDisplayLandmarker() for that.
export function createFeatureLandmarker(): Promise<FaceLandmarker> {
  return createFaceLandmarker(1);
}

// Drives the multi-face overlay + "People" count only (LandmarkOverlay,
// PerfHUD). Never feed its output into computeFeatures() — see
// createFeatureLandmarker() above for why. Running two landmarkers costs a
// second WASM detection pass per sampled frame; accepted so the multi-face
// UI (commit c16cd9a) and tight feature parity can both be kept.
export function createDisplayLandmarker(): Promise<FaceLandmarker> {
  return createFaceLandmarker(4);
}
