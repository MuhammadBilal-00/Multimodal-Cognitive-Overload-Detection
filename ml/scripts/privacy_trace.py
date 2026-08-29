"""B9: recorded network-trace verification of the privacy claim, as a
reproducible script (the original 2026-08-03 trace was collected ad hoc;
this makes the evidence re-runnable on demand).

Starts the PRODUCTION Next.js server, drives the app with a fake webcam
(DAiSEE-derived y4m — same fixture as e2e_app_test.py), and records every
network request plus every console message for --seconds (default 70,
comfortably past @mediapipe/tasks-vision's ~60 s telemetry flush). The page
runs under the app's real CSP — no bypass_csp — so this doubles as the
validation that the full production policy (default-src lockdown,
wasm-unsafe-eval, worker blob:, etc.) does not break the app: the run FAILS
unless the pipeline reaches "live", and FAILS on any CSP-violation console
line that is not the expected connect-src block of the MediaPipe telemetry
endpoint.

Writes docs/results/privacy_trace.json. Screenshotting is deliberately not
done here (the fake-cam frame is a DAiSEE face; see e2e_app_test.py).

Usage: python ml/scripts/privacy_trace.py [--seconds 70]
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WEB_ROOT = REPO_ROOT / "web"
RESULTS_DIR = REPO_ROOT / "docs" / "results"
PORT = 3100
TELEMETRY_HOST = "odml.pa.googleapis.com"


def wait_for_port(port: int, timeout_s: int = 60) -> None:
    import socket
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise SystemExit(f"server did not open port {port}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=70)
    args = parser.parse_args()

    y4m = REPO_ROOT / "artifacts" / "parity_cam.y4m"
    if not y4m.exists():
        raise SystemExit(f"{y4m} missing — run ml/scripts/e2e_app_test.py "
                         f"once first (it builds the fixture)")
    next_bin = WEB_ROOT / "node_modules" / ".bin" / "next"
    if not (WEB_ROOT / ".next").exists():
        raise SystemExit("web/.next not found — run `npm run build` in web/")

    server = subprocess.Popen(
        ["node", str(WEB_ROOT / "node_modules" / "next" / "dist" / "bin" / "next"),
         "start", "-p", str(PORT)],
        cwd=WEB_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    requests: list[dict] = []
    console: list[str] = []
    t_start = time.time()
    try:
        wait_for_port(PORT)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-video-capture={y4m}",
            ])
            # NO bypass_csp — the whole point is the production policy.
            context = browser.new_context(permissions=["camera"])
            page = context.new_page()
            page.on("request", lambda r: requests.append(
                {"t": round(time.time() - t_start, 1), "method": r.method,
                 "url": r.url}))
            page.on("console", lambda m: console.append(f"[{m.type}] {m.text}"))
            page.goto(f"http://localhost:{PORT}", wait_until="domcontentloaded")
            # Selector engines run in Playwright's isolated world — CSP-safe.
            # CognitiveApp renders the status pill as "Live" once
            # status === 'live' && !error (components/CognitiveApp.tsx).
            page.get_by_text("Live", exact=True).wait_for(timeout=90_000)
            t_live = round(time.time() - t_start, 1)
            time.sleep(max(0, args.seconds - (time.time() - t_start)))
            browser.close()
    finally:
        server.terminate()

    origins = sorted({f"{urlsplit(r['url']).scheme}://{urlsplit(r['url']).netloc}"
                      for r in requests})
    external = [r for r in requests
                if urlsplit(r["url"]).hostname not in ("localhost", "127.0.0.1")]
    csp_lines = [c for c in console
                 if "Content Security Policy" in c or "CSP" in c]
    telemetry_blocks = [c for c in csp_lines if TELEMETRY_HOST in c]
    unexpected_csp = [c for c in csp_lines if TELEMETRY_HOST not in c]

    report = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": args.seconds,
        "server": f"production build, next start on localhost:{PORT}",
        "reached_live_at_s": t_live,
        "total_requests": len(requests),
        "origins_contacted": origins,
        "external_requests": external,           # must be []
        "csp_violation_lines": csp_lines,
        "telemetry_block_observed": len(telemetry_blocks) > 0,
        "unexpected_csp_violations": unexpected_csp,  # must be []
        "verdict": {
            "zero_external_requests": len(external) == 0,
            "app_reached_live_under_full_csp": True,
            "only_expected_csp_violations": len(unexpected_csp) == 0,
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "privacy_trace.json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {out}")
    print(json.dumps({k: v for k, v in report.items()
                     if k != "external_requests" or v}, indent=1)[:2000])

    if external:
        raise SystemExit("FAIL: external requests observed")
    if unexpected_csp:
        raise SystemExit("FAIL: unexpected CSP violations — the policy is "
                         "breaking something:\n" + "\n".join(unexpected_csp))
    print("PRIVACY TRACE: PASS")


if __name__ == "__main__":
    main()
