"""Subject-level, label-stratified k-fold splits WITHIN the Train split
only (Phase 0 of the rigorous-fix pass). Validation and Test are never
touched by this module -- every downstream CV-based script (feature
selection, hyperparameter search, ensemble stacking) trains/selects using
only these folds, preserving comparability with the official DAiSEE
benchmark on the splits that matter for final reporting.

Grouped by SUBJECT (first 6 characters of clip_id, matching labels.py's
own convention exactly -- `df["subject"] = [c[:6] for c in clip_ids]`),
not by clip_id: Train has only 69 unique subjects averaging ~78 clips
each, so a clip-level-only grouping (this module's first version) let the
same subject's clips split across a fold's train/held-out sets -- exactly
the subject leakage `labels.py`'s own `assert_split_integrity` guards
against between Train/Validation/Test. A model can partly learn to
recognise a specific person's face/framing rather than the engagement
signal if their clips appear on both sides of a fold; grouping by subject
closes that gap the same way the official splits already do.

Stratified by each clip's engagement label via sklearn's
StratifiedGroupKFold, so no fold's held-out set ends up with zero examples
of the rarest class (~0.6% of windows overall).

Usage:
    from cv_splits import get_folds
    for fold_idx, (train_idx, val_idx) in enumerate(get_folds(clip_ids, y)):
        ...
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS = 5


def subjects_of(clip_ids: np.ndarray) -> np.ndarray:
    """First 6 characters of each clip id -- labels.py's own subject
    convention, duplicated here (not imported) to keep this module
    dependency-free of the pandas-based labels.py loading path."""
    return np.array([str(c)[:6] for c in clip_ids])


def get_folds(clip_ids: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS,
              seed: int = 42):
    """Yields (train_idx, val_idx) WINDOW-index arrays, n_folds times.

    `y` is the per-window engagement label (constant within a clip, per
    dataset.py's construction) -- used only for stratification here, not
    as a per-window signal distinct from its clip's label.
    """
    subjects = subjects_of(clip_ids)
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                               random_state=seed)
    yield from skf.split(np.zeros(len(y)), y, groups=subjects)


def verify_folds(clip_ids: np.ndarray, y: np.ndarray,
                 n_folds: int = N_FOLDS, seed: int = 42) -> None:
    """Correctness check (Plan's Verification section): every fold's
    held-out set contains at least one example of every class present in
    the full label array, and no SUBJECT appears in both a fold's train
    and held-out sets (the leakage this module exists to prevent).
    Raises AssertionError on violation."""
    subjects = subjects_of(clip_ids)
    all_classes = set(np.unique(y).tolist())
    for fold_idx, (train_idx, val_idx) in enumerate(
            get_folds(clip_ids, y, n_folds, seed)):
        held_out_classes = set(np.unique(y[val_idx]).tolist())
        missing = all_classes - held_out_classes
        assert not missing, (
            f"fold {fold_idx}: held-out set missing classes {missing}")
        train_subjects = set(subjects[train_idx].tolist())
        val_subjects = set(subjects[val_idx].tolist())
        overlap = train_subjects & val_subjects
        assert not overlap, (
            f"fold {fold_idx}: {len(overlap)} subject(s) in both train and "
            f"held-out sets: {sorted(overlap)[:5]} ...")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    SRC_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SRC_DIR.parent.parent
    data = np.load(REPO_ROOT / "artifacts" / "dataset" / "Train.npz")
    y, clip_ids = data["y_engagement"], data["clip_ids"]

    verify_folds(clip_ids, y)
    print(f"verified {N_FOLDS} folds: every held-out set covers all "
          f"{len(np.unique(y))} classes, zero SUBJECT overlap "
          f"({len(set(subjects_of(clip_ids)))} unique subjects total)")
    for fold_idx, (train_idx, val_idx) in enumerate(get_folds(clip_ids, y)):
        subjects = subjects_of(clip_ids)
        print(f"fold {fold_idx}: train={len(train_idx)} windows "
              f"({len(set(subjects[train_idx]))} subjects, "
              f"{len(set(clip_ids[train_idx]))} clips), "
              f"val={len(val_idx)} windows "
              f"({len(set(subjects[val_idx]))} subjects, "
              f"{len(set(clip_ids[val_idx]))} clips), "
              f"val class counts={np.bincount(y[val_idx], minlength=4).tolist()}")
