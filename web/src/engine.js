/**
 * EngagementEngine: owns the full client-side pipeline.
 *   webcam (un-mirrored) -> canvas -> FaceLandmarker (CPU delegate, VIDEO
 *   mode) -> features.js -> pitch-centre + scaler -> WindowBuffer ->
 *   int8 ONNX every 5th frame -> softmax/sigmoid.
 *
 * The landmarker MUST run the CPU (XNNPACK) delegate: it is the backend
 * the training features were extracted with; the GPU delegate shifts
 * landmarks past the parity tolerance (docs/results/parity_report_gpu.json).
 */

import { computeFeatures, landmarksToPixels } from "./features.js";
import {
  TARGET_FPS, WindowBuffer, decodeOutputs, standardize,
} from "./pipeline.js";

export class EngagementEngine {
  constructor({ onUpdate, onError }) {
    this.onUpdate = onUpdate || (() => {});
    this.onError = onError || console.error;
    this.buffer = new WindowBuffer();
    this.running = false;
    this.frameCount = 0;
    this.timestampMs = 0;
    this.stats = { landmarkMs: 0, inferMs: 0, effectiveFps: 0 };
    this.lastPrediction = null;
  }

  async init() {
    const ort = await import("onnxruntime-web");
    ort.env.wasm.wasmPaths = "/ort/";
    this.ort = ort;
    this.session = await ort.InferenceSession.create(
      "/model/model_int8.onnx", { executionProviders: ["wasm"] });

    this.scaler = await (await fetch("/model/scaler.json")).json();

    const { FilesetResolver, FaceLandmarker } =
      await import("@mediapipe/tasks-vision");
    const vision = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
    this.landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "/mediapipe/face_landmarker.task",
        delegate: "CPU", // REQUIRED for train/deploy parity - see header
      },
      runningMode: "VIDEO",
      numFaces: 1,
    });
  }

  async start(videoElement) {
    this.video = videoElement;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 }, audio: false,
    });
    this.video.srcObject = stream;
    await this.video.play();

    // Processing canvas: the UN-MIRRORED frame (contract requirement).
    // Any mirroring for display happens in CSS only.
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.video.videoWidth || 640;
    this.canvas.height = this.video.videoHeight || 480;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });

    this.running = true;
    this.frameCount = 0;
    this.timestampMs = 0;
    this._lastTickWall = performance.now();
    this._tickTimer = setInterval(() => {
      this._tick().catch(this.onError);
    }, 1000 / TARGET_FPS);
  }

  stop() {
    this.running = false;
    if (this._tickTimer) clearInterval(this._tickTimer);
    if (this.video && this.video.srcObject) {
      for (const track of this.video.srcObject.getTracks()) track.stop();
      this.video.srcObject = null;
    }
  }

  async _tick() {
    if (this.running !== true || this._busy) return;
    this._busy = true;
    try {
      const now = performance.now();
      this.stats.effectiveFps =
        0.9 * this.stats.effectiveFps + 0.1 * (1000 / (now - this._lastTickWall));
      this._lastTickWall = now;

      const h = this.canvas.height;
      const w = this.canvas.width;
      this.ctx.drawImage(this.video, 0, 0, w, h);

      // Timestamps advance by exactly 100 ms per processed frame, matching
      // the training-time sampling clock (extract.py).
      this.timestampMs += 1000 / TARGET_FPS;
      const t0 = performance.now();
      const result = this.landmarker.detectForVideo(
        this.canvas, Math.round(this.timestampMs));
      this.stats.landmarkMs = performance.now() - t0;

      let features;
      if (result.faceLandmarks && result.faceLandmarks.length > 0) {
        const px = landmarksToPixels(result.faceLandmarks[0], w, h);
        features = computeFeatures(px, [h, w], this.scaler.pitch_centre);
      } else {
        features = computeFeatures(null, [h, w], this.scaler.pitch_centre);
      }
      const facePresent = features[12] === 1.0;
      this.buffer.push(standardize(features, this.scaler));
      this.frameCount += 1;

      if (this.buffer.shouldInfer()) {
        const tensor = new this.ort.Tensor(
          "float32", this.buffer.toTensorData(), [1, 30, 13]);
        const t1 = performance.now();
        const outputs = await this.session.run({ features: tensor });
        this.stats.inferMs = performance.now() - t1;
        this.buffer.markInferred();
        this.lastPrediction = decodeOutputs(outputs);
      }

      this.onUpdate({
        frameCount: this.frameCount,
        facePresent,
        bufferFill: this.buffer.frames.length,
        prediction: this.lastPrediction,
        stats: { ...this.stats },
      });
    } finally {
      this._busy = false;
    }
  }
}
