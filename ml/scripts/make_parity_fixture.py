"""Generate the J1 parity fixture (Day 5 deliverable to Track B).

1. Transcodes one DAiSEE clip to a browser-playable VP9 WebM
   (ml/tests/fixtures/parity_clip.webm — NOT committed, DAiSEE license).
2. Runs the exact extraction pipeline (FaceLandmarker VIDEO mode, 10 FPS
   sampling, pixel conversion, compute_features with raw pitch) on the
   TRANSCODED webm and writes ml/tests/fixtures/parity_expected.json with
   full float precision (committed).
3. Dumps the same sampled frames as PNGs
   (ml/tests/fixtures/parity_frames/frame_###.png — NOT committed, same
   DAiSEE-license restriction as the clip) so the browser-side J1 gate
   (web/tests/e2e/features.parity.test.ts) can run headless against static
   images instead of seeking a <video> element, which headless Chromium
   does not reliably present (verified: seeked frames came back stale).

The expected features and the PNG frames are written from the SAME decode
loop over the transcoded webm, so both sides of the parity gate consume
byte-identical, frame-index-aligned input.

Usage: python ml/scripts/make_parity_fixture.py [--clip <path-to-avi>]
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))

from extract import TARGET_FPS, _landmarks_to_pixels  # noqa: E402
from features import FEATURE_NAMES, compute_features  # noqa: E402

DEFAULT_CLIP = (REPO_ROOT / "data" / "DAiSEE" / "DataSet" / "Train"
                / "110001" / "1100011002" / "1100011002.avi")
FIXTURE_DIR = REPO_ROOT / "ml" / "tests" / "fixtures"
FRAMES_DIR = FIXTURE_DIR / "parity_frames"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    args = parser.parse_args()

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for stale in FRAMES_DIR.glob("frame_*.png"):
        stale.unlink()  # a previous run may have left more/fewer frames
    mp4_path = FIXTURE_DIR / "parity_clip.webm"

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(args.clip),
         "-c:v", "libvpx-vp9", "-crf", "24", "-b:v", "0", "-pix_fmt", "yuv420p",
         "-an", str(mp4_path)],
        check=True)
    sha256 = hashlib.sha256(mp4_path.read_bytes()).hexdigest()

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(
            REPO_ROOT / "ml" / "assets" / "face_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )

    cap = cv2.VideoCapture(str(mp4_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / TARGET_FPS)))

    rows = []
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                height, width = frame_bgr.shape[:2]
                # cv2.imwrite expects BGR, which is exactly what cap.read()
                # returns — writing frame_bgr directly, unconverted, so the
                # PNG is pixel-identical to what compute_features was run
                # against below (same array, no extra codec round-trip).
                cv2.imwrite(str(FRAMES_DIR / f"frame_{len(rows):03d}.png"), frame_bgr)
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(round(frame_idx * 1000.0 / src_fps))
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                if result.face_landmarks:
                    pixels = _landmarks_to_pixels(
                        result.face_landmarks[0], width, height)
                else:
                    pixels = None
                feats = compute_features(pixels, (height, width))
                rows.append([float(v) for v in feats])
            frame_idx += 1
    cap.release()

    fixture = {
        "version": "1.0",
        "source_clip": "parity_clip.webm",
        "clip_sha256": sha256,
        "source_daisee_clip": args.clip.stem,
        "video": {"fps": src_fps, "sample_step": step,
                  "width": width, "height": height},
        "pitch_centre_used": 0.0,
        "feature_names": FEATURE_NAMES,
        "num_frames": len(rows),
        "features": rows,
        "notes": (
            "Features computed on the TRANSCODED mp4 with the Python "
            "reference pipeline (FaceLandmarker VIDEO mode, pixel coords, "
            "raw pitch). Browser decode may differ at pixel level; the J1 "
            "gate compares with tolerance, suggested max-abs-diff 0.02 "
            "per feature after excluding face_present mismatches."),
    }
    out_path = FIXTURE_DIR / "parity_expected.json"
    with open(out_path, "w") as fh:
        json.dump(fixture, fh, indent=1)

    det = sum(1 for r in rows if r[-1] == 1.0) / len(rows)
    print(f"parity_clip.webm  sha256={sha256[:16]}...  "
          f"{len(rows)} frames, detection rate {det:.3f}")
    print(f"wrote {out_path}")
    print(f"wrote {len(rows)} PNG frames to {FRAMES_DIR}")


if __name__ == "__main__":
    main()

