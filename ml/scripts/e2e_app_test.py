"""J2: end-to-end test of the current app with a fake webcam.

Feeds the parity clip to Chromium via --use-file-for-fake-video-capture,
starts the production Next.js server, and waits for a real prediction to
land via window.__ENGINE_STATE (exposed by hooks/usePipeline.ts). Asserts
the full pipeline end to end: webcam -> landmarker -> features -> scaler
-> window -> int8 ONNX -> UI.

There is no "Start camera" button to click — the current app starts the
camera automatically once permission is granted (usePipeline.ts /
CognitiveApp.tsx), unlike the deleted JS scaffold this script originally
targeted.

Writes docs/results/app_screenshot.png and app_e2e.json.

Usage: python ml/scripts/e2e_app_test.py
"""

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WEB_ROOT = REPO_ROOT / "web"
RESULTS_DIR = REPO_ROOT / "docs" / "results"
SCRATCH = REPO_ROOT / "artifacts"
PORT = 3123

LEVELS = ["Very Low", "Low", "High", "Very High"]  # PredictionPanel.tsx


def make_y4m() -> Path:
    """Chromium's fake capture device needs y4m (raw) input."""
    y4m = SCRATCH / "parity_cam.y4m"
    if not y4m.exists():
        SCRATCH.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error",
             "-i", str(REPO_ROOT / "ml" / "tests" / "fixtures"
                       / "parity_clip.webm"),
             "-pix_fmt", "yuv420p", str(y4m)],
            check=True)
    return y4m


def wait_for_port(port: int, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise SystemExit(f"server on :{port} did not come up")


def main() -> None:
    from playwright.sync_api import sync_playwright

    y4m = make_y4m()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    next_bin = WEB_ROOT / "node_modules" / "next" / "dist" / "bin" / "next"
    if not next_bin.exists():
        raise SystemExit(f"{next_bin} not found — run `npm install` in web/ first")
    if not (WEB_ROOT / ".next").exists():
        raise SystemExit("web/.next not found — run `npm run build` in web/ first")

    server = subprocess.Popen(
        ["node", str(next_bin), "start", "-p", str(PORT)],
        cwd=WEB_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_port(PORT)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-video-capture={y4m}",
            ])
            # bypass_csp: the app's production CSP (script-src without
            # 'unsafe-eval') correctly blocks Playwright's eval-injected
            # wait_for_function/evaluate — a tooling constraint, not an app
            # bug. Functionality-under-CSP is validated separately by
            # ml/scripts/privacy_trace.py, which drives the page without
            # main-world eval and fails on any unexpected CSP violation.
            context = browser.new_context(permissions=["camera"],
                                          bypass_csp=True)
            page = context.new_page()
            console = []
            page.on("console", lambda m: console.append(f"[{m.type}] {m.text}"))

            page.goto(f"http://127.0.0.1:{PORT}/")
            # Model load (WASM download) + camera start + filling the 3.0 s
            # window (CONTRACT.md §6 Amendment 1: one prediction per 3 s, not
            # the old 2 Hz stride) before the first prediction exists.
            # 90 s is a CI-cold-start budget, not the expected wait.
            page.wait_for_function(
                "window.__ENGINE_STATE && window.__ENGINE_STATE.prediction !== null",
                timeout=90_000)
            # Let a SECOND 3 s window land too, so the screenshot/state
            # isn't just the first-ever (possibly still-settling) prediction.
            # Must exceed one full inference window (3 s) — 2 s here was a
            # leftover 2 Hz-era assumption that never actually caught a
            # second prediction under the current cadence.
            time.sleep(4)
            state = page.evaluate("window.__ENGINE_STATE")
            # Screenshot goes to gitignored artifacts/, NOT docs/results/:
            # the fake-camera feed is a DAiSEE clip, so the frame contains a
            # dataset participant's face and must never enter the repo (the
            # same licence rule as the raw clips). A committed-safe app
            # screenshot must be captured separately with a non-DAiSEE face.
            page.screenshot(path=str(SCRATCH / "app_screenshot_e2e.png"),
                            full_page=True)
            browser.close()

        pred = state["prediction"]
        prob_sum = sum(pred["engagement"])
        level = pred["engagement"].index(max(pred["engagement"]))
        checks = {
            "status": state["status"],
            "face_present": bool(state["facePresent"]),
            "probs_sum_to_1": abs(prob_sum - 1.0) < 1e-3,
            "engagement_level": level,
            "engagement_label": LEVELS[level],
            "engagement_probs": pred["engagement"],
            "state_probs": pred["states"],
            # usePipeline.ts exposes statesNamed precisely so the artefact
            # is not a bare positional array — recording only the positions
            # is what let the alphabetical-vs-contract channel-order bug
            # (CONTRACT.md Amendment 3) sit unnoticed in this file.
            "states_named": state.get("statesNamed"),
            "infer_ms": pred["ms"],
        }
        ok = (checks["status"] == "live" and checks["face_present"]
              and checks["probs_sum_to_1"])
        checks["ok"] = ok
        with open(RESULTS_DIR / "app_e2e.json", "w") as fh:
            json.dump(checks, fh, indent=1)
        print(json.dumps(checks, indent=1))
        print("E2E:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
