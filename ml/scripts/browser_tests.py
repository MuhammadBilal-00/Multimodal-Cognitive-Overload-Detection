"""Headless-browser gate: A6.5 (int8 ONNX in onnxruntime-web) via
Playwright Chromium.

Serves D:/fyp/web over localhost, opens web/harness/onnx_smoke.html, waits
for window.__RESULT, and writes docs/results/browser_smoke.json.

J1 (feature parity, Python vs browser) used to live here too but has moved
to web/tests/e2e/features.parity.test.ts (a real Playwright Test run via
`npm run test:parity`), which reuses the production Next.js app instead of
a hand-rolled static harness — see CONTRACT.md Amendment 2 for why.

Usage: python ml/scripts/browser_tests.py
Exit code 0 only if the gate passes.
"""

import json
import functools
import mimetypes
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


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def run_page(page_path: str, timeout_ms: int) -> dict:
    from playwright.sync_api import sync_playwright

    console_lines: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    handler = functools.partial(QuietHandler, directory=str(WEB_ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    ok = True
    try:
        print("== A6.5 browser smoke (onnxruntime-web, int8) ==")
        smoke = run_page("/harness/onnx_smoke.html", 120_000)
        with open(RESULTS_DIR / "browser_smoke.json", "w") as fh:
            json.dump(smoke, fh, indent=1)
        if smoke.get("ok"):
            print(f"  OK  load {smoke['loadMs']:.0f} ms, "
                  f"inference p50 {smoke['inferMsP50']:.2f} ms, "
                  f"p90 {smoke['inferMsP90']:.2f} ms")
        else:
            ok = False
            print(f"  FAILED: {smoke.get('error')}")
    finally:
        server.shutdown()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

