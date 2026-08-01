"""Windowing + standardisation: feature CSVs -> model-ready arrays (A4).

Windows: 30 frames (3.0 s at 10 FPS), training stride 10 -> ~8 windows per
~100-row clip. The clip's label applies to every window from that clip.

Pipeline order (must match the browser exactly):
  1. raw features from CSV (pitch is stored RAW by extract.py)
  2. pitch <- pitch - pitch_centre   (train-set mean of raw pitch over
                                      face_present frames)
  3. x <- (x - mean) / std           (scaler fit on TRAIN windows only)

scaler.json ships mean/std/feature_names/pitch_centre (CONTRACT.md §7).
face_present is deliberately given mean=0, std=1 so the binary flag passes
through standardisation unchanged on both sides.

Run:  python ml/src/dataset.py     (after extraction completes)
Writes artifacts/dataset/{split}.npz and web/public/model/scaler.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
REPO_ROOT = SRC_DIR.parent.parent

from features import FEATURE_NAMES  # noqa: E402
from labels import (  # noqa: E402
    LABEL_COLS, SPLITS, binarize_states, labelled_extracted_clips,
    load_all_labels)

WINDOW = 30
TRAIN_STRIDE = 10
INFERENCE_STRIDE = 5  # browser-side; recorded here for the record

PITCH_IDX = FEATURE_NAMES.index("pitch")
FACE_PRESENT_IDX = FEATURE_NAMES.index("face_present")


def window_clip(frames: np.ndarray, stride: int) -> np.ndarray:
    """(N, 13) float32 -> (num_windows, 30, 13). Empty if N < 30."""
    if len(frames) < WINDOW:
        return np.zeros((0, WINDOW, len(FEATURE_NAMES)), dtype=np.float32)
    starts = range(0, len(frames) - WINDOW + 1, stride)
    windows = [frames[s:s + WINDOW] for s in starts]
    return np.stack(windows).astype(np.float32)


def load_clip_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    return df[FEATURE_NAMES].to_numpy(dtype=np.float32)


def compute_pitch_centre(features_dir: Path, train_clips: list[str]) -> float:
    """Mean RAW pitch over face_present frames of the training split."""
    total = 0.0
    count = 0
    for clip_id in train_clips:
        arr = load_clip_csv(features_dir / "Train" / f"{clip_id}.csv")
        present = arr[:, FACE_PRESENT_IDX] == 1.0
        total += float(arr[present, PITCH_IDX].sum())
        count += int(present.sum())
    assert count > 0, "no face-present frames in training split"
    return total / count


def build_split(features_dir: Path, split: str, labels: pd.DataFrame,
                clips: list[str], pitch_centre: float,
                stride: int) -> dict[str, np.ndarray]:
    """Windows + per-window labels for one split (pitch centred, unscaled)."""
    xs, y_eng, y_states, clip_ids = [], [], [], []
    states = binarize_states(labels)
    for clip_id in clips:
        arr = load_clip_csv(features_dir / split / f"{clip_id}.csv")
        # centre pitch only where a face was present (zeros stay zeros,
        # exactly as the browser computes it)
        present = arr[:, FACE_PRESENT_IDX] == 1.0
        arr[present, PITCH_IDX] -= pitch_centre
        wins = window_clip(arr, stride)
        if len(wins) == 0:
            continue
        xs.append(wins)
        y_eng.extend([int(labels.loc[clip_id, "Engagement"])] * len(wins))
        y_states.extend([states.loc[clip_id].to_numpy(dtype=np.float32)]
                        * len(wins))
        clip_ids.extend([clip_id] * len(wins))
    return {
        "x": np.concatenate(xs) if xs else
             np.zeros((0, WINDOW, len(FEATURE_NAMES)), dtype=np.float32),
        "y_engagement": np.array(y_eng, dtype=np.int64),
        "y_states": np.array(y_states, dtype=np.float32),
        "clip_ids": np.array(clip_ids),
    }


def fit_scaler(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over all TRAIN frames; identity for face_present."""
    flat = train_x.reshape(-1, train_x.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.maximum(std, 1e-6)
    mean[FACE_PRESENT_IDX] = 0.0
    std[FACE_PRESENT_IDX] = 1.0
    return mean.astype(np.float64), std.astype(np.float64)


def apply_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def main() -> None:
    features_dir = REPO_ROOT / "artifacts" / "features"
    out_dir = REPO_ROOT / "artifacts" / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_labels = load_all_labels(REPO_ROOT / "data" / "DAiSEE" / "Labels")

    usable = {s: labelled_extracted_clips(splits_labels[s], features_dir, s)
              for s in SPLITS}
    for s in SPLITS:
        print(f"{s}: {len(usable[s])} usable clips "
              f"(labelled {len(splits_labels[s])})")

    # Split integrity again, on the actually-used clip sets.
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not set(usable[a]) & set(usable[b]), \
                    f"clip overlap {a}/{b}"

    pitch_centre = compute_pitch_centre(features_dir, usable["Train"])
    print(f"pitch_centre (train mean raw pitch) = {pitch_centre:.6f}")

    built = {}
    for split in SPLITS:
        stride = TRAIN_STRIDE  # same windowing for all offline splits
        built[split] = build_split(features_dir, split, splits_labels[split],
                                   usable[split], pitch_centre, stride)
        b = built[split]
        print(f"{split}: x{tuple(b['x'].shape)}, "
              f"engagement dist {np.bincount(b['y_engagement'], minlength=4)}")

    mean, std = fit_scaler(built["Train"]["x"])

    for split in SPLITS:
        b = built[split]
        np.savez_compressed(
            out_dir / f"{split}.npz",
            x=apply_scaler(b["x"], mean, std),
            y_engagement=b["y_engagement"],
            y_states=b["y_states"],
            clip_ids=b["clip_ids"])

    scaler = {
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "feature_names": FEATURE_NAMES,
        "pitch_centre": float(pitch_centre),
        "version": "1.0",
    }
    for dest in (REPO_ROOT / "web" / "public" / "model" / "scaler.json",
                 out_dir / "scaler.json"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as fh:
            json.dump(scaler, fh, indent=1)
    print(f"scaler.json written (pitch_centre={pitch_centre:.4f})")


if __name__ == "__main__":
    main()
