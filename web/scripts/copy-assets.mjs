/**
 * Copies runtime assets that must be served statically into public/:
 *   - MediaPipe tasks-vision WASM + face_landmarker.task -> public/mediapipe/
 *   - onnxruntime-web WASM/MJS                            -> public/ort/
 * Rerun after npm install or after updating ml/assets/face_landmarker.task.
 *   node scripts/copy-assets.mjs
 */
import { cpSync, mkdirSync, copyFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const repoRoot = dirname(webRoot);

mkdirSync(join(webRoot, "public", "mediapipe"), { recursive: true });
mkdirSync(join(webRoot, "public", "ort"), { recursive: true });

cpSync(join(webRoot, "node_modules", "@mediapipe", "tasks-vision", "wasm"),
       join(webRoot, "public", "mediapipe", "wasm"), { recursive: true });

copyFileSync(join(repoRoot, "ml", "assets", "face_landmarker.task"),
             join(webRoot, "public", "mediapipe", "face_landmarker.task"));

const ortDist = join(webRoot, "node_modules", "onnxruntime-web", "dist");
for (const f of readdirSync(ortDist)) {
  if (f.endsWith(".wasm") || f.endsWith(".mjs")) {
    copyFileSync(join(ortDist, f), join(webRoot, "public", "ort", f));
  }
}
console.log("assets copied to public/mediapipe and public/ort");
