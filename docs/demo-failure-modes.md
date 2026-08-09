# Demo hardening — cross-browser + failure modes

PROJECT_COMPLETION_PLAN.md Phase 1: prove the demo survives contact with
reality before a panel finds the edges. All tests below were run against
a genuinely clean clone + build (`git clone` into a fresh directory,
`npm install && npm run build && npm start` — see the Phase 1.1 note in
`BUILD_PLAN_1.md`'s status banner), driven headlessly via Playwright with
`--use-fake-device-for-media-stream` / `--use-file-for-fake-video-capture`
feeding real DAiSEE clips (or synthetic frames) as the webcam, so every
result below reflects the real pipeline (webcam → landmarker → features →
scaler → ONNX → UI), not a mock.

Screenshots were captured locally for each scenario but are **not
committed** — several contain DAiSEE participants' faces, same
redistribution restriction already applied to `docs/results/app_screenshot.png`.
The state dumps below (`window.__ENGINE_STATE` + DOM perf stats) are the
recorded evidence instead.

## 1.2 — Cross-browser check

Chrome and Edge driven with a real DAiSEE clip as the fake camera file
(Chromium-only capability). Firefox has no equivalent file-backed fake
camera in Playwright, so it was driven with `media.navigator.streams.fake`
instead, which substitutes Firefox's own synthetic test pattern, not a
real face — see the Firefox row for what that does and doesn't prove.
Safari not tested (no Mac reachable, per the plan's own conditional).

| Browser | Loads & runs | Face detected | `wasm×threads` (COOP/COEP) | Notes |
|---|---|---|---|---|
| Chrome (chromium) | ✅ | ✅ (478 landmarks, `face_present=1`) | `wasm×20` | Baseline; matches all prior dev-machine testing |
| Edge (`channel: msedge`) | ✅ | ✅ (478 landmarks, `face_present=1`) | `wasm×20` | Identical behavior to Chrome, as expected (same Chromium engine) |
| Firefox | ✅, no console errors, MediaPipe/XNNPACK initializes | N/A — fed Firefox's synthetic fake-camera pattern (solid color + moving bar), not a real face | `wasm×20` | **Confirms**: app loads, WASM initializes, COOP/COEP cross-origin isolation is honored (multi-threaded), and the "no face in frame" path (below) renders correctly and doesn't crash. **Does not confirm**: real face detection in Firefox specifically — needs a manual test with an actual webcam/person, which a student or panel dry run should do at least once. |

**COOP/COEP finding:** all three browsers report `wasm×20` (20 = this
machine's `hardwareConcurrency`) — WASM multithreading is honored
everywhere tested, not just Chrome.

**Camera-permission UX:** `WebcamFeed.tsx` maps `getUserMedia` failures to
five distinct `DOMException.name` values (`NotAllowedError` → denied,
`NotFoundError` → no camera, `NotReadableError` → busy,
`NotSupportedError` → unsupported, else → generic error), each with its
own message, verified by code review. Live-verified one branch directly:
a context with no camera device at all correctly renders "Camera access
isn't supported here — this page needs HTTPS (or localhost) and a device
with a camera," with the dashboard staying in a clean "waiting for
camera" state (0 fps, dashes for features, no crash) rather than hanging.

## 1.3 — Failure-mode hardening

| Scenario | Source | `face_present` | People | Result |
|---|---|---|---|---|
| No face in frame | Synthetic black frame (`ffmpeg color=black`) | `0.0000` | 0 | ✅ All-zero features per CONTRACT §2.1's missing-face rule. UI shows the black feed, zeroed feature panel, status stays `live` (not stuck on "filling window" or frozen) with a low-confidence-looking but well-formed prediction. **Observation, not a bug**: there's no dedicated "no face currently visible" banner (distinct from `WebcamFeed`'s camera-error banners) — the zeroed dashboard is the only signal. Acceptable for a research demo; worth a live callout during the rehearsal (Phase 5) so it isn't mistaken for a freeze. |
| Two faces in frame | Two DAiSEE clips composited side-by-side (`ffmpeg hstack`) so both faces are simultaneously front-facing — a real two-person DAiSEE frame was tried first, but that clip's background person is always turned away from the camera, so it only ever exercises the single-face path | `1.0000` | **2** | ✅ Both faces detected (478 landmarks). `selectPrimaryFace` (`lib/primaryFace.ts`) picked the larger/closer face as primary; verified at 2x zoom that the overlay draws the non-primary face's landmarks dimmed (`#71717a`) vs. the primary's cyan (`#22d3ee`) — correct per `LandmarkOverlay.tsx`'s design, just subtle at normal screenshot resolution. |
| Bad lighting (moderate) | Real clip, `eq=brightness=-0.25:contrast=0.6` | `1.0000` | 1 | ✅ Face still detected, valid non-garbage prediction — the "degrades gracefully" branch. |
| Bad lighting (severe) | Real clip, `eq=brightness=-0.5:contrast=0.4` | `0.0000` | 0 | ✅ No detection anywhere in the window — the "cleanly reports no-detection" branch (the other acceptable outcome per this check's own acceptance criterion). Never a garbage in-between prediction at either lighting level tested. |
| Glasses | Real DAiSEE clip, subject wearing glasses throughout | `1.0000` | 1 | ✅ Full 478-landmark detection including iris (`gaze_x`/`gaze_y` populated, not zeroed), despite lens glare. No degradation observed in this clip's lighting conditions. |

**None of the five scenarios crashed, froze, or produced a garbage
(non-finite / out-of-range / non-normalized) prediction.** Every
`prediction.engagement` observed sums to 1.0 within floating-point
tolerance in every scenario, including the two all-zero-feature cases.

## What this doesn't cover

- Real-camera, real-human testing in Firefox and Safari (Safari
  untestable here — no Mac).
- Glasses + bad lighting + motion combined (only tested independently).
- A live "someone actually walks in front of the panel's projector"
  rehearsal — that's Phase 5 (J4 dry run), deliberately deferred until
  the rest of the repo is frozen.
