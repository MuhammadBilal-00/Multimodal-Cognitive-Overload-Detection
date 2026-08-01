"""Extract per-frame contract features from every DAiSEE clip.

For each clip: open with OpenCV, sample to 10 FPS (every 3rd frame at the
native 30 fps), run MediaPipe FaceLandmarker (Tasks API, VIDEO mode),
compute the 13-feature contract vector, and write
artifacts/features/{split}/{clip_id}.csv.

Pitch is stored RAW (pitch_centre=0.0 here); the training-set mean is
computed on Day 5 and shipped in scaler.json.

Usage:
    python ml/src/extract.py                     # everything
    python ml/src/extract.py --limit 20          # smoke run
    python ml/src/extract.py --resume            # skip existing CSVs
    python ml/src/extract.py --split Train --min-detection-confidence 0.3

Notes:
- multiprocessing with spawn (Windows-safe); the FaceLandmarker is created
  INSIDE each worker. A fresh landmarker per clip keeps VIDEO-mode
  timestamps monotonic and prevents tracking state bleeding across clips.
- Failures are logged per clip into stats.json, never abort the run.
"""

import argparse
import csv
import json
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
REPO_ROOT = SRC_DIR.parent.parent

from features import FEATURE_NAMES, compute_features  # noqa: E402

TARGET_FPS = 10.0
CSV_HEADER = ["frame"] + FEATURE_NAMES

# Worker-global state, set once per worker process by _init_worker().
_MODEL_BUFFER = None
_MIN_CONF = 0.5


def _init_worker(model_path: str, min_conf: float) -> None:
    global _MODEL_BUFFER, _MIN_CONF
    _MODEL_BUFFER = Path(model_path).read_bytes()
    _MIN_CONF = min_conf


def _landmarks_to_pixels(face_landmarks, width: int, height: int) -> np.ndarray:
    """Contract conversion: normalised MediaPipe output -> pixel coords."""
    out = np.zeros((478, 3), dtype=np.float64)
    for i, p in enumerate(face_landmarks):
        out[i, 0] = p.x * width
        out[i, 1] = p.y * height
        out[i, 2] = p.z * width
    return out


def extract_clip(job: tuple[str, str, str]) -> dict:
    """Process one clip. Returns a stats record; never raises."""
    clip_path, split, out_csv = job
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    record = {
        "clip": Path(clip_path).stem,
        "split": split,
        "rows": 0,
        "detection_rate": 0.0,
        "error": None,
    }
    try:
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open the clip")
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 0:
            src_fps = 30.0
        step = max(1, int(round(src_fps / TARGET_FPS)))

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=_MODEL_BUFFER),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=_MIN_CONF,
        )
        rows = []
        faces = 0
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            frame_idx = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if frame_idx % step == 0:
                    height, width = frame_bgr.shape[:2]
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp_ms = int(round(frame_idx * 1000.0 / src_fps))
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    if result.face_landmarks:
                        pixels = _landmarks_to_pixels(
                            result.face_landmarks[0], width, height)
                        faces += 1
                    else:
                        pixels = None
                    feats = compute_features(pixels, (height, width))
                    rows.append([len(rows)] + [f"{v:.6f}" for v in feats])
                frame_idx += 1
        cap.release()

        if not rows:
            raise RuntimeError("no frames decoded")

        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".csv.tmp")
        with open(tmp_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_HEADER)
            writer.writerows(rows)
        tmp_path.replace(out_path)  # atomic-ish: no half-written CSVs on crash

        record["rows"] = len(rows)
        record["detection_rate"] = faces / len(rows)
    except Exception:
        record["error"] = traceback.format_exc(limit=2).strip().splitlines()[-1]
    return record


def find_jobs(data_root: Path, out_root: Path, splits: list[str],
              resume: bool, limit: int | None) -> list[tuple[str, str, str]]:
    jobs = []
    for split in splits:
        split_dir = data_root / "DataSet" / split
        for ext in ("*.avi", "*.mp4"):
            for clip in sorted(split_dir.rglob(ext)):
                out_csv = out_root / split / f"{clip.stem}.csv"
                if resume and out_csv.exists():
                    continue
                jobs.append((str(clip), split, str(out_csv)))
    if limit is not None:
        jobs = jobs[:limit]
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=REPO_ROOT / "data" / "DAiSEE")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "artifacts" / "features")
    parser.add_argument("--model", type=Path,
                        default=REPO_ROOT / "ml" / "assets" / "face_landmarker.task")
    parser.add_argument("--split", choices=["Train", "Validation", "Test"],
                        action="append",
                        help="restrict to split(s); default all three")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true",
                        help="skip clips whose CSV already exists")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N clips (smoke runs)")
    args = parser.parse_args()

    from tqdm import tqdm

    splits = args.split or ["Train", "Validation", "Test"]
    jobs = find_jobs(args.data_root, args.out, splits, args.resume, args.limit)
    print(f"{len(jobs)} clips to process "
          f"(splits={splits}, workers={args.workers}, "
          f"min_conf={args.min_detection_confidence}, resume={args.resume})")
    if not jobs:
        return

    start = time.time()
    records = []
    with multiprocessing.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(str(args.model), args.min_detection_confidence)) as pool:
        for record in tqdm(pool.imap_unordered(extract_clip, jobs, chunksize=4),
                           total=len(jobs), unit="clip", smoothing=0.05):
            records.append(record)
    elapsed = time.time() - start

    ok = [r for r in records if r["error"] is None]
    failed = [r for r in records if r["error"] is not None]
    stats = {
        "config": {
            "splits": splits,
            "min_detection_confidence": args.min_detection_confidence,
            "target_fps": TARGET_FPS,
            "workers": args.workers,
            "limit": args.limit,
            "resume": args.resume,
        },
        "elapsed_seconds": round(elapsed, 1),
        "clips_ok": len(ok),
        "clips_failed": len(failed),
        "mean_detection_rate": (
            float(np.mean([r["detection_rate"] for r in ok])) if ok else 0.0),
        "per_split": {},
        "low_detection_clips": sorted(
            (r for r in ok if r["detection_rate"] < 0.5),
            key=lambda r: r["detection_rate"])[:50],
        "failures": failed,
    }
    for split in splits:
        split_ok = [r for r in ok if r["split"] == split]
        stats["per_split"][split] = {
            "clips_ok": len(split_ok),
            "clips_failed": len([r for r in failed if r["split"] == split]),
            "mean_detection_rate": (
                float(np.mean([r["detection_rate"] for r in split_ok]))
                if split_ok else 0.0),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    stats_name = "stats.json" if args.limit is None else "stats_smoke.json"
    stats_path = args.out / stats_name
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"\nDone in {elapsed/60:.1f} min: {len(ok)} ok, {len(failed)} failed")
    print(f"Mean face-detection rate: {stats['mean_detection_rate']:.3f}")
    if stats["mean_detection_rate"] < 0.85 and args.limit is None:
        print("WARNING: detection rate under 0.85 — investigate before "
              "training (try --min-detection-confidence 0.3)")
    print(f"Stats written to {stats_path}")


if __name__ == "__main__":
    main()
