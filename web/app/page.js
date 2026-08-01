"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { EngagementEngine } from "../src/engine.js";
import { ENGAGEMENT_LABELS, STATE_LABELS } from "../src/pipeline.js";

function Bars({ labels, values, topIndex }) {
  return labels.map((label, i) => (
    <div className="bar-row" key={label}>
      <span className="bar-label">{label}</span>
      <div className="bar-track">
        <div
          className={"bar-fill" + (i === topIndex ? " top" : "")}
          style={{ width: `${(values ? values[i] * 100 : 0).toFixed(1)}%` }}
        />
      </div>
      <span className="bar-value">
        {values ? `${(values[i] * 100).toFixed(0)}%` : "–"}
      </span>
    </div>
  ));
}

export default function Home() {
  const videoRef = useRef(null);
  const engineRef = useRef(null);
  const [phase, setPhase] = useState("idle"); // idle|loading|running|error
  const [error, setError] = useState(null);
  const [state, setState] = useState(null);

  const start = useCallback(async () => {
    setPhase("loading");
    setError(null);
    try {
      const engine = new EngagementEngine({
        onUpdate: (update) => {
          setState(update);
          if (typeof window !== "undefined") window.__ENGINE_STATE = update;
        },
        onError: (err) => {
          setError(String(err));
          setPhase("error");
        },
      });
      await engine.init();
      await engine.start(videoRef.current);
      engineRef.current = engine;
      setPhase("running");
    } catch (err) {
      setError(String(err));
      setPhase("error");
    }
  }, []);

  const stop = useCallback(() => {
    engineRef.current?.stop();
    engineRef.current = null;
    setPhase("idle");
  }, []);

  useEffect(() => () => engineRef.current?.stop(), []);

  const pred = state?.prediction;
  return (
    <main>
      <h1>Cognitive State Detection</h1>
      <p className="sub">
        Engagement recognition running entirely in your browser — video never
        leaves this device. <a href="/bench">Benchmark page</a>
      </p>

      <div className="grid">
        <div className="panel">
          <video ref={videoRef} className="preview" muted playsInline />
          <div style={{ marginTop: 12 }}>
            {phase !== "running" ? (
              <button className="primary" onClick={start}
                      disabled={phase === "loading"}>
                {phase === "loading" ? "Loading models…" : "Start camera"}
              </button>
            ) : (
              <button className="primary" onClick={stop}>Stop</button>
            )}
          </div>
          {error && <p className="notice">{error}</p>}
          {state && !state.facePresent && phase === "running" && (
            <p className="notice">No face detected</p>
          )}
        </div>

        <div className="panel">
          <div className="headline" data-testid="engagement-label">
            {pred ? pred.engagementLabel : "warming up…"}
          </div>
          <Bars labels={ENGAGEMENT_LABELS}
                values={pred?.engagement}
                topIndex={pred?.engagementIndex ?? -1} />
          <div style={{ marginTop: 18, fontWeight: 600, fontSize: "0.9rem" }}>
            States (independent)
          </div>
          <Bars labels={STATE_LABELS} values={pred?.states} topIndex={-1} />
          <p className="statline" data-testid="stats">
            {state
              ? `frames ${state.frameCount} · window ${state.bufferFill}/30 · ` +
                `landmarks ${state.stats.landmarkMs.toFixed(1)} ms · ` +
                `inference ${state.stats.inferMs.toFixed(2)} ms · ` +
                `${state.stats.effectiveFps.toFixed(1)} fps`
              : "not running"}
          </p>
        </div>
      </div>
    </main>
  );
}
