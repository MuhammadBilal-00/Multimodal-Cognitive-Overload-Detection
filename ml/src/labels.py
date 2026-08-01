"""DAiSEE label loading and split integrity (CONTRACT.md §9).

Labels per clip: Boredom, Engagement, Confusion, Frustration, each 0-3.
- Primary task:   4-class engagement level (the standard DAiSEE benchmark)
- Secondary task: 4 binary states, thresholded at level >= 2

Only the OFFICIAL Train/Validation/Test splits are used. Zero clip-ID and
zero subject-ID overlap between splits is asserted in code, every load.

Known dataset quirk (verified Day 2): the label CSVs cover fewer clips
than exist on disk (5358/1429/1784 vs 5481/1720/1866). Training uses the
intersection of labelled clips and successfully extracted feature CSVs.
"""

from pathlib import Path

import pandas as pd

SPLITS = ["Train", "Validation", "Test"]
LABEL_COLS = ["Boredom", "Engagement", "Confusion", "Frustration"]
STATE_THRESHOLD = 2  # binary state = 1 if level >= 2


def load_split_labels(labels_dir: Path, split: str) -> pd.DataFrame:
    """Load one split's labels, indexed by clip id (filename stem).

    Returns columns: Boredom, Engagement, Confusion, Frustration (int 0-3)
    plus subject (str, first 6 chars of the clip id).
    """
    csv_path = Path(labels_dir) / f"{split}Labels.csv"
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    clip_ids = []
    for raw in df["ClipID"]:
        stem = str(raw).strip()
        if "." in stem:
            stem = stem.rsplit(".", 1)[0]
        clip_ids.append(stem)
    df["clip_id"] = clip_ids
    df["subject"] = [c[:6] for c in clip_ids]

    for col in LABEL_COLS:
        df[col] = df[col].astype(int)
        bad = df[(df[col] < 0) | (df[col] > 3)]
        assert bad.empty, f"{split} {col}: labels outside 0-3 range"

    df = df.drop_duplicates(subset="clip_id", keep="first")
    return df.set_index("clip_id")[LABEL_COLS + ["subject"]]


def load_all_labels(labels_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all three splits and assert split integrity."""
    splits = {s: load_split_labels(labels_dir, s) for s in SPLITS}
    assert_split_integrity(splits)
    return splits


def assert_split_integrity(splits: dict[str, pd.DataFrame]) -> None:
    """Hard assertions: no clip and no subject appears in two splits."""
    names = list(splits.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            clip_overlap = set(splits[a].index) & set(splits[b].index)
            assert not clip_overlap, (
                f"clip-ID overlap between {a} and {b}: "
                f"{sorted(clip_overlap)[:5]} ...")
            subj_overlap = (set(splits[a]["subject"])
                            & set(splits[b]["subject"]))
            assert not subj_overlap, (
                f"subject overlap between {a} and {b}: "
                f"{sorted(subj_overlap)[:5]} ...")


def binarize_states(labels: pd.DataFrame) -> pd.DataFrame:
    """4 binary state targets: 1 where the level is >= STATE_THRESHOLD."""
    out = pd.DataFrame(index=labels.index)
    for col in LABEL_COLS:
        out[col.lower()] = (labels[col] >= STATE_THRESHOLD).astype(int)
    return out


def labelled_extracted_clips(labels: pd.DataFrame,
                             features_dir: Path,
                             split: str) -> list[str]:
    """Clip ids that have BOTH a label row and an extracted feature CSV."""
    available = {p.stem for p in (Path(features_dir) / split).glob("*.csv")}
    return sorted(set(labels.index) & available)
