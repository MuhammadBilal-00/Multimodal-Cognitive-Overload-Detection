"""End-to-end test of the Next.js app with a fake webcam.

Feeds the parity clip to Chromium via --use-file-for-fake-video-capture,
starts the production Next.js server, clicks "Start camera", and waits for
a real prediction to render. Asserts the full pipeline (webcam ->
landmarker -> features -> scaler -> window -> int8 ONNX -> UI) end to end.

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


def make_y4m() -> Path:
    """Chromium's fake capture device needs y4m (raw) input."""
    y4m = SCRATCH / "parity_cam.y4m"
    if not y4m.exists():
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

    server = subprocess.Popen(
        ["node", str(WEB_ROOT / "node_modules" / "next" / "dist" / "bin"
                     / "next"), "start", "-p", str(PORT)],
        cwd=WEB_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_port(PORT)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-video-capture={y4m}",
            ])
            context = browser.new_context(permissions=["camera"])
            page = context.new_page()
            console = []
            page.on("console", lambda m: console.append(f"[{m.type}] {m.text}"))

            page.goto(f"http://127.0.0.1:{PORT}/")
            page.click("text=Start camera")
            # model load + 30 frames at 10 FPS before the first inference
            page.wait_for_function(
                "window.__ENGINE_STATE && "
                "window.__ENGINE_STATE.prediction !== null",
                timeout=90_000)
            time.sleep(2)  # let a few more inferences smooth the UI
            state = page.evaluate("window.__ENGINE_STATE")
            page.screenshot(path=str(RESULTS_DIR / "app_screenshot.png"),
                            full_page=True)
            browser.close()

        pred = state["prediction"]
        prob_sum = sum(pred["engagement"])
        checks = {
            "face_present": bool(state["facePresent"]),
            "window_full": state["bufferFill"] == 30,
            "probs_sum_to_1": abs(prob_sum - 1.0) < 1e-3,
            "label_valid": pred["engagementLabel"] in
                ["very low", "low", "engaged", "very engaged"],
            "landmark_ms": state["stats"]["landmarkMs"],
            "infer_ms": state["stats"]["inferMs"],
            "effective_fps": state["stats"]["effectiveFps"],
            "engagement_probs": pred["engagement"],
            "engagement_label": pred["engagementLabel"],
            "state_probs": pred["states"],
        }
        ok = (checks["face_present"] and checks["window_full"]
              and checks["probs_sum_to_1"] and checks["label_valid"])
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
