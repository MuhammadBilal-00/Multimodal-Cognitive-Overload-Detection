"""J3/B7: drive the app's own "Run 300 inferences" benchmark headlessly
and archive the result.

Starts the production Next.js server, opens the collapsed Benchmark
panel and clicks its button (BenchmarkPanel.tsx), captures the JSON file
it downloads (runBenchmark() in lib/benchmark.ts triggers a browser
download, not a window global — there is no /bench page and no
window.__BENCH anymore), and copies it into
docs/benchmarks/benchmark-<machine-label>.json.

Requires `web/` already built (`npm run build` in web/) — this script only
starts the production server, it does not build.

Usage: python ml/scripts/collect_benchmark.py --machine-label "dev-laptop-i7-16GB"
"""

import argparse
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WEB_ROOT = REPO_ROOT / "web"
BENCHMARKS_DIR = REPO_ROOT / "docs" / "benchmarks"
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


def sanitize(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-") or "unlabeled"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--machine-label", required=True,
        help='human-supplied hardware label, e.g. "dev-laptop-i7-16GB" — '
             "CPU model / RAM amount aren't reliably readable from the "
             "browser, so this is manual")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
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
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{PORT}/")
            # The BenchmarkPanel is behind a collapsed toggle
            # (CognitiveApp.tsx: `benchmarkOpen` starts false), so the
            # "Run 300 inferences" button does not exist at page load —
            # open the panel first, or the click below times out.
            page.get_by_role("button", name="Benchmark", exact=True).click()
            run_button = page.get_by_role("button", name="Run 300 inferences")
            run_button.wait_for(state="visible", timeout=30_000)
            with page.expect_download(timeout=180_000) as download_info:
                run_button.click()
            download = download_info.value
            out_path = BENCHMARKS_DIR / f"benchmark-{sanitize(args.machine_label)}.json"
            download.save_as(out_path)
            browser.close()
    finally:
        server.terminate()

    print(f"wrote {out_path}")
    print(out_path.read_text())


if __name__ == "__main__":
    main()
