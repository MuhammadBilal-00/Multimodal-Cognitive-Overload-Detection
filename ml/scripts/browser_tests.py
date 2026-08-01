"""Headless-browser gates: A6.5 (int8 ONNX in onnxruntime-web) and J1
(feature parity, Python vs browser) via Playwright Chromium.

Serves D:/fyp/web over localhost, opens the two harness pages, waits for
window.__RESULT, and writes reports to docs/results/.

Usage: python ml/scripts/browser_tests.py [--only smoke|parity]
Exit code 0 only if every requested gate passes.
"""

import argparse
import functools
import json
import mimetypes
import shutil
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WEB_ROOT = REPO_ROOT / "web"
RESULTS_DIR = REPO_ROOT / "docs" / "results"
PORT = 8931

mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("video/webm", ".webm")


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def stage_fixtures() -> None:
    harness = WEB_ROOT / "harness"
    pairs = [
        (REPO_ROOT / "ml" / "tests" / "fixtures" / "parity_clip.webm",
         harness / "parity_clip.webm"),
        (REPO_ROOT / "ml" / "tests" / "fixtures" / "parity_expected.json",
         harness / "parity_expected.json"),
        (REPO_ROOT / "ml" / "assets" / "face_landmarker.task",
         harness / "face_landmarker.task"),
    ]
    for src, dst in pairs:
        if not src.exists():
            raise SystemExit(f"missing fixture: {src}")
        shutil.copyfile(src, dst)


def run_page(page_path: str, timeout_ms: int, headed: bool = False) -> dict:
    from playwright.sync_api import sync_playwright

    console_lines: list[str] = []
    with sync_playwright() as pw:
        # headed=True for pages that seek <video>: headless Chromium never
        # presents seeked frames (verified: 1 distinct frame in 100).
        browser = pw.chromium.launch(headless=not headed)
        page = browser.new_page()
        page.on("console",
                lambda m: console_lines.append(f"[{m.type}] {m.text}"))
        page.goto(f"http://127.0.0.1:{PORT}{page_path}")
        page.wait_for_function("window.__RESULT !== null",
                               timeout=timeout_ms)
        result = page.evaluate("window.__RESULT")
        browser.close()
    if console_lines:
        result["console"] = console_lines[-20:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["smoke", "parity"], default=None)
    args = parser.parse_args()

    stage_fixtures()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    handler = functools.partial(QuietHandler, directory=str(WEB_ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    all_ok = True
    try:
        if args.only in (None, "smoke"):
            print("== A6.5 browser smoke (onnxruntime-web, int8) ==")
            smoke = run_page("/harness/onnx_smoke.html", 120_000)
            with open(RESULTS_DIR / "browser_smoke.json", "w") as fh:
                json.dump(smoke, fh, indent=1)
            if smoke.get("ok"):
                print(f"  OK  load {smoke['loadMs']:.0f} ms, "
                      f"inference p50 {smoke['inferMsP50']:.2f} ms, "
                      f"p90 {smoke['inferMsP90']:.2f} ms")
            else:
                all_ok = False
                print(f"  FAILED: {smoke.get('error')}")

        if args.only in (None, "parity"):
            print("== J1 parity gate (browser vs Python features) ==")
            parity = run_page("/harness/parity.html", 300_000)
            with open(RESULTS_DIR / "parity_report.json", "w") as fh:
                json.dump(parity, fh, indent=1)
            if parity.get("error"):
                all_ok = False
                print(f"  FAILED: {parity['error']}")
            else:
                print(f"  frames compared: {parity['framesCompared']}, "
                      f"face mismatches: {parity['faceMismatchFrames']}")
                print(f"  worst per-feature max|diff|: "
                      f"{parity['worstFeatureDiff']:.5f} "
                      f"(tolerance {parity['tolerance']})")
                worst3 = sorted(parity["perFeatureMaxAbsDiff"].items(),
                                key=lambda kv: -kv[1])[:3]
                for name, v in worst3:
                    print(f"    {name}: {v:.5f}")
                print(f"  gate: {'PASS' if parity['ok'] else 'FAIL'}")
                all_ok = all_ok and parity["ok"]
    finally:
        server.shutdown()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

