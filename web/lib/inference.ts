import * as ort from 'onnxruntime-web';
import { softmax, sigmoid, standardise } from './mathUtils';
import type { Scaler } from './scaler';

let session: ort.InferenceSession | null = null;
let sessionInfo = { backend: 'wasm', threads: 0, modelBytes: 0 };

export async function initInference(modelUrl = '/model/model_int8.onnx') {
  if (session) return sessionInfo;
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
  if (!session) throw new Error('initInference() not called');
  const t0 = performance.now();
  const std = standardise(win, scaler.mean, scaler.std);
  const tensor = new ort.Tensor('float32', std, [1, 30, 13]);
  const out = await session.run({ features: tensor });
  const ms = performance.now() - t0;
  return {
    engagement: softmax(out.engagement.data as Float32Array),
    states: sigmoid(out.states.data as Float32Array),
    ms,
  };
}
