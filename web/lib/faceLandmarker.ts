import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

export async function createLandmarker(): Promise<FaceLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
  // CPU delegate only: the model was trained on features extracted with the
  // CPU/XNNPACK landmarker, and the GPU delegate shifts landmarks enough to
  // fail the J1 parity gate (docs/results/parity_report_gpu.json — gaze_y
  // 0.05 > 0.02 tol; CPU worst-case 0.0079).
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate: 'CPU' },
    runningMode: 'VIDEO' as const,
    numFaces: 4,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false, // contract 3.2: never use these
  });
}
