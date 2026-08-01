/**
 * Inference pipeline glue: scaler application, sliding window, and output
 * head post-processing. Mirrors ml/src/dataset.py order EXACTLY:
 *   1. computeFeatures(..., pitchCentre)   (pitch centred where face present)
 *   2. z = (x - mean) / std                (scaler.json, train-only fit)
 * Softmax/sigmoid live HERE, not in the ONNX graph (CONTRACT.md §5).
 */

export const WINDOW = 30;          // frames (3.0 s at 10 FPS)
export const INFERENCE_STRIDE = 5; // run the model every 5th frame
export const TARGET_FPS = 10;

export const ENGAGEMENT_LABELS = [
  "very low", "low", "engaged", "very engaged"];
export const STATE_LABELS = [
  "boredom", "engagement", "confusion", "frustration"];

export function standardize(features, scaler) {
  const out = new Float32Array(features.length);
  for (let i = 0; i < features.length; i++) {
    out[i] = (features[i] - scaler.mean[i]) / scaler.std[i];
  }
  return out;
}

export function softmax(logits) {
  const m = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - m));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export function sigmoid(logits) {
  return logits.map((v) => 1 / (1 + Math.exp(-v)));
}

/** Fixed-size FIFO of standardized frames; flattens to [1, 30, 13]. */
export class WindowBuffer {
  constructor(windowSize = WINDOW, numFeatures = 13) {
    this.windowSize = windowSize;
    this.numFeatures = numFeatures;
    this.frames = [];
    this.framesSinceInference = 0;
  }

  push(standardizedFrame) {
    this.frames.push(standardizedFrame);
    if (this.frames.length > this.windowSize) this.frames.shift();
    this.framesSinceInference += 1;
  }

  get full() {
    return this.frames.length === this.windowSize;
  }

  /** True when the buffer is full and stride frames passed since last run. */
  shouldInfer() {
    return this.full && this.framesSinceInference >= INFERENCE_STRIDE;
  }

  markInferred() {
    this.framesSinceInference = 0;
  }

  toTensorData() {
    const flat = new Float32Array(this.windowSize * this.numFeatures);
    for (let t = 0; t < this.frames.length; t++) {
      flat.set(this.frames[t], t * this.numFeatures);
    }
    return flat;
  }
}

/** Decode both heads from an onnxruntime-web results map. */
export function decodeOutputs(results) {
  const engagement = softmax(Array.from(results.engagement.data));
  const states = sigmoid(Array.from(results.states.data));
  let top = 0;
  for (let i = 1; i < engagement.length; i++) {
    if (engagement[i] > engagement[top]) top = i;
  }
  return {
    engagement,
    engagementLabel: ENGAGEMENT_LABELS[top],
    engagementIndex: top,
    states,
  };
}
