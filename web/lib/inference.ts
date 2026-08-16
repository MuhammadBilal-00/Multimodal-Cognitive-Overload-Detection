// onnxruntime-web's own package (any entry: umbrella, /wasm, /webgpu, ...)
// fails production Terser minification when webpack statically bundles it —
// its .mjs distributables use import.meta/dynamic-import patterns webpack's
// asset pipeline mis-handles. onnxruntime-web's own source dodges this for
// ITS internal dynamic imports via a webpackIgnore comment; we apply the
// same trick to our own top-level import, loading the copy already
// self-hosted at /ort/ (Task 3's copy-assets.mjs) as a genuine browser
// runtime import that webpack never sees. Types only come from the package
// (import type is fully erased, so it can never trigger the bundling bug).
import type * as OrtType from 'onnxruntime-web';
import { softmax, sigmoid, standardise, distributionCheck } from './mathUtils';
import type { Scaler } from './scaler';

const ORT_MODULE_PATH = '/ort/ort.wasm.min.mjs';

let ort: typeof OrtType | null = null;
let session: OrtType.InferenceSession | null = null;
let sessionInfo = { backend: 'wasm', threads: 0, modelBytes: 0 };

export async function initInference(modelUrl = '/model/model_int8.onnx') {
  if (session && ort) return sessionInfo;
  if (!ort) {
    ort = (await import(/* webpackIgnore: true */ ORT_MODULE_PATH)) as typeof OrtType;
  }
  ort.env.wasm.wasmPaths = '/ort/';
  ort.env.wasm.numThreads = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency ?? 4 : 4;
  ort.env.wasm.simd = true;
  const res = await fetch(modelUrl);
  if (!res.ok) throw new Error(`model fetch failed: ${res.status}`);
  const bytes = await res.arrayBuffer();
  session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  sessionInfo = { backend: 'wasm', threads: ort.env.wasm.numThreads as number, modelBytes: bytes.byteLength };
  return sessionInfo;
}

export async function runInference(win: Float32Array, scaler: Scaler) {
  if (!session || !ort) throw new Error('initInference() not called');
  const t0 = performance.now();
  const std = standardise(win, scaler.mean, scaler.std);
  const tensor = new ort.Tensor('float32', std, [1, 30, 13]);
  const out = await session.run({ features: tensor });
  const ms = performance.now() - t0;
  return {
    engagement: softmax(out.engagement.data as Float32Array),
    // sigmoid, not softmax: the four states are independent binary heads
    // (BCEWithLogitsLoss in ml/src/train.py), so they legitimately don't
    // sum to 1. Channel order is documented in lib/states.ts.
    states: sigmoid(out.states.data as Float32Array),
    ood: distributionCheck(std),
    ms,
  };
}
