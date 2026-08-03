import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

export async function createLandmarker(): Promise<FaceLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
  const opts = (delegate: 'GPU' | 'CPU') => ({
    baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate },
    runningMode: 'VIDEO' as const,
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false, // contract 3.2: never use these
  });
  try {
    return await FaceLandmarker.createFromOptions(fileset, opts('GPU'));
  } catch (e) {
    console.warn('GPU delegate failed, falling back to CPU', e);
    return await FaceLandmarker.createFromOptions(fileset, opts('CPU'));
  }
}
