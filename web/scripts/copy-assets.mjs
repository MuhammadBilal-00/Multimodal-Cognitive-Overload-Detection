import { cpSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const root = new URL('..', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1');

mkdirSync(join(root, 'public/ort'), { recursive: true });
const ortDist = join(root, 'node_modules/onnxruntime-web/dist');
for (const f of readdirSync(ortDist)) {
  if (f.endsWith('.wasm') || f.endsWith('.mjs')) {
    cpSync(join(ortDist, f), join(root, 'public/ort', f));
  }
}

cpSync(
  join(root, 'node_modules/@mediapipe/tasks-vision/wasm'),
  join(root, 'public/mediapipe/wasm'),
  { recursive: true },
);

// face_landmarker.task is gitignored (README download step) and NOT
// covered by this script — it doesn't come from node_modules. Its absence
// doesn't fail the build (the app still compiles), but createFeatureLandmarker()/createDisplayLandmarker()
// will 404 at runtime, and that 404 surfaces as an opaque timeout in the
// J1 Playwright gate rather than an obvious error. Warn loudly here so a
// clean checkout doesn't go looking for a mystery CI failure.
const modelPath = join(root, 'public/models/face_landmarker.task');
if (!existsSync(modelPath)) {
  console.warn(
    '\nWARNING: web/public/models/face_landmarker.task is missing.\n' +
    '  The app and the J1 parity test (`npm run test:parity`) both need it.\n' +
    '  Download it — see README.md, "MediaPipe model asset":\n' +
    '  Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile web\\public\\models\\face_landmarker.task\n',
  );
}

console.log('assets copied');
