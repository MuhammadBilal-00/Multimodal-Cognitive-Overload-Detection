"use client";

/**
 * J3 benchmark page: model load time, inference latency distribution, and
 * landmarker latency on a static frame. Results render as a table and are
 * exposed on window.__BENCH for automated collection.
 */

import { useCallback, useState } from "react";

async function runBenchmark(setProgress) {
  const report = { userAgent: navigator.userAgent };

  const ort = await import("onnxruntime-web");
  ort.env.wasm.wasmPaths = "/ort/";

  const x = new ort.Tensor("float32", new Float32Array(30 * 13), [1, 30, 13]);
  report.models = {};
  for (const name of ["model_int8.onnx", "model_fp32.onnx"]) {
    setProgress(`loading ${name}…`);
    let t0 = performance.now();
    let session;
    try {
      session = await ort.InferenceSession.create(
        `/model/${name}`, { executionProviders: ["wasm"] });
    } catch {
      report.models[name] = { available: false };
      continue;
    }
    const loadMs = performance.now() - t0;

    setProgress(`timing ${name}…`);
    for (let i = 0; i < 20; i++) await session.run({ features: x });
    const times = [];
    for (let i = 0; i < 200; i++) {
      const t = performance.now();
      await session.run({ features: x });
      times.push(performance.now() - t);
    }
    times.sort((a, b) => a - b);
    report.models[name] = {
      available: true,
      loadMs,
      inferMs: {
        p50: times[100], p90: times[180], p99: times[198],
        mean: times.reduce((a, b) => a + b, 0) / times.length,
      },
    };
  }

  let t0;
  setProgress("loading landmarker…");
  const { FilesetResolver, FaceLandmarker } =
    await import("@mediapipe/tasks-vision");
  const vision = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
  t0 = performance.now();
  const landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "/mediapipe/face_landmarker.task",
                   delegate: "CPU" },
    runningMode: "VIDEO",
    numFaces: 1,
  });
  report.landmarkerLoadMs = performance.now() - t0;

  setProgress("timing landmarker (static frame, no face)…");
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 480;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, 640, 480);
  const lmTimes = [];
  for (let i = 0; i < 50; i++) {
    const t = performance.now();
    landmarker.detectForVideo(canvas, (i + 1) * 100);
    lmTimes.push(performance.now() - t);
  }
  lmTimes.sort((a, b) => a - b);
  report.landmarkMs = { p50: lmTimes[25], p90: lmTimes[45] };
  report.note =
    "landmark timing is on a faceless gray frame (detector path only); " +
    "with a face present expect higher values - see live page stats";
  return report;
}

export default function Bench() {
  const [progress, setProgress] = useState(null);
  const [report, setReport] = useState(null);

  const run = useCallback(async () => {
    setReport(null);
    try {
      const r = await runBenchmark(setProgress);
      setReport(r);
      window.__BENCH = r;
    } catch (e) {
      setProgress(String(e));
    } finally {
      setProgress(null);
    }
  }, []);

  return (
    <main>
      <h1>Benchmark</h1>
      <p className="sub">
        Model + landmarker latency on this device. <a href="/">Live page</a>
      </p>
      <div className="panel">
        <button className="primary" onClick={run} disabled={!!progress}>
          {progress || "Run benchmark"}
        </button>
        {report && (
          <pre data-testid="bench-report"
               style={{ marginTop: 16, fontSize: "0.85rem",
                        whiteSpace: "pre-wrap" }}>
            {JSON.stringify(report, null, 1)}
          </pre>
        )}
      </div>
    </main>
  );
}
