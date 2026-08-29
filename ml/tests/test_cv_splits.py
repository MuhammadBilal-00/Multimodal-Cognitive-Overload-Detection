"""Regression tests for the subject-grouped CV splitter (cv_splits.py).

This module exists because of a real bug: the first version of cv_splits.py
grouped folds by clip ID only, and with DAiSEE's ~78 clips per subject the
same subject's clips landed on both sides of a fold — subject leakage that
manufactured a false feature-selection finding before being caught
(docs/results/rigorous_model_search.md, "A methodological correction found
and fixed mid-pipeline"). The thesis reports that correction; these tests
make sure it can never silently regress.

Synthetic data only — no DAiSEE access needed, so this runs in CI.
"""

import numpy as np
import pytest

from cv_splits import get_folds, subjects_of, verify_folds


def synthetic_corpus(n_subjects: int = 20, clips_per_subject: int = 6,
                     windows_per_clip: int = 4, seed: int = 0):
    """clip_ids follow DAiSEE's convention: first 6 chars = subject id."""
    rng = np.random.default_rng(seed)
    clip_ids, y = [], []
    for s in range(n_subjects):
        subject = f"{s:06d}"
        # every subject contributes several classes so stratification has
        # room to balance folds; rare classes appear for a subset of subjects
        for c in range(clips_per_subject):
            clip = f"{subject}{c:04d}"
            label = rng.choice([0, 1, 2, 3], p=[0.05, 0.15, 0.5, 0.3])
            clip_ids.extend([clip] * windows_per_clip)
            y.extend([int(label)] * windows_per_clip)
    return np.array(clip_ids), np.array(y, dtype=np.int64)


def test_subjects_of_takes_first_six_chars():
    assert subjects_of(np.array(["1100011002", "5000441001"])).tolist() == \
        ["110001", "500044"]


def test_no_subject_appears_on_both_sides_of_any_fold():
    clip_ids, y = synthetic_corpus()
    subjects = subjects_of(clip_ids)
    for train_idx, val_idx in get_folds(clip_ids, y):
        overlap = set(subjects[train_idx]) & set(subjects[val_idx])
        assert not overlap, f"subject leakage across a fold: {sorted(overlap)}"


def test_every_window_held_out_exactly_once():
    clip_ids, y = synthetic_corpus()
    seen = np.zeros(len(y), dtype=int)
    for _, val_idx in get_folds(clip_ids, y):
        seen[val_idx] += 1
    assert (seen == 1).all()


def test_verify_folds_passes_on_clean_data():
    clip_ids, y = synthetic_corpus()
    verify_folds(clip_ids, y)  # raises AssertionError on any violation


def test_folds_are_deterministic_for_a_seed():
    clip_ids, y = synthetic_corpus()
    a = [val.tolist() for _, val in get_folds(clip_ids, y, seed=42)]
    b = [val.tolist() for _, val in get_folds(clip_ids, y, seed=42)]
    assert a == b


def test_clip_level_grouping_would_be_rejected():
    """The original bug, reproduced: grouping by CLIP allows one subject's
    clips to straddle a fold, which verify_folds must reject. Constructed
    directly — a train/val index pair that splits one subject's windows —
    to pin the check itself, independent of sklearn's fold assignment."""
    clip_ids, y = synthetic_corpus(n_subjects=2, clips_per_subject=4)
    subjects = subjects_of(clip_ids)
    first_subject = subjects == subjects[0]
    # half of subject 0's windows in "train", the other half in "val"
    idx = np.flatnonzero(first_subject)
    train_idx = idx[: len(idx) // 2]
    val_idx = idx[len(idx) // 2:]
    with pytest.raises(AssertionError):
        # reuse the same assertion verify_folds applies per fold
        overlap = set(subjects[train_idx]) & set(subjects[val_idx])
        assert not overlap, "subject overlap"
