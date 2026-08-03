import { cpSync, mkdirSync, readdirSync } from 'node:fs';
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

console.log('assets copied');
