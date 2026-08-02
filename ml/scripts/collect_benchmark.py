"""J3: run the /bench page headlessly and archive the numbers.

Starts the production Next.js server, clicks "Run benchmark", waits for
window.__BENCH, and writes docs/results/browser_benchmark.json.

Usage: python ml/scripts/collect_benchmark.py
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
PORT = 3124


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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        ["node", str(WEB_ROOT / "node_modules" / "next" / "dist" / "bin"
                     / "next"), "start", "-p", str(PORT)],
        cwd=WEB_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_port(PORT)
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{PORT}/bench")
            page.click("text=Run benchmark")
            page.wait_for_function("window.__BENCH !== undefined",
                                   timeout=180_000)
            bench = page.evaluate("window.__BENCH")
            browser.close()
    finally:
        server.terminate()

    with open(RESULTS_DIR / "browser_benchmark.json", "w") as fh:
        json.dump(bench, fh, indent=1)
    print(json.dumps(bench, indent=1))


if __name__ == "__main__":
    main()
