"""Tests for labels.py — the states channel order and binarisation rule.

The channel order (LABEL_COLS) is the exact thing whose silent divergence
produced the shipped Amendment 3 defect (a UI bar labelled "Confused"
displaying P(engagement)). The browser side guards it by parsing this very
file as text (web/tests/states.test.ts); until now there was no Python-side
assertion at all that binarize_states() produces the order and threshold
CONTRACT.md §5 Amendment 3 documents.
"""

import pandas as pd
import pytest

from labels import (
    LABEL_COLS, STATE_THRESHOLD, assert_split_integrity, binarize_states)


def test_label_cols_frozen_order():
    # CONTRACT.md §5 Amendment 3: states[0..3] = boredom, engagement,
    # confusion, frustration — NOT alphabetical.
    assert LABEL_COLS == ["Boredom", "Engagement", "Confusion", "Frustration"]


def test_binarize_states_column_order_matches_label_cols():
    df = pd.DataFrame(
        {c: [0] for c in LABEL_COLS} | {"subject": ["110001"]},
        index=["1100011002"])
    out = binarize_states(df)
    assert list(out.columns) == [c.lower() for c in LABEL_COLS]


def test_binarize_threshold_is_ge_2():
    assert STATE_THRESHOLD == 2
    df = pd.DataFrame({
        "Boredom": [0, 1, 2, 3],
        "Engagement": [2, 2, 2, 2],
        "Confusion": [1, 1, 1, 1],
        "Frustration": [3, 0, 3, 0],
        "subject": ["s"] * 4,
    }, index=[f"clip{i}" for i in range(4)])
    out = binarize_states(df)
    assert out["boredom"].tolist() == [0, 0, 1, 1]      # >= 2
    assert out["engagement"].tolist() == [1, 1, 1, 1]
    assert out["confusion"].tolist() == [0, 0, 0, 0]    # level 1 < 2
    assert out["frustration"].tolist() == [1, 0, 1, 0]


def _split_df(clip_ids):
    return pd.DataFrame(
        {c: [2] * len(clip_ids) for c in LABEL_COLS}
        | {"subject": [c[:6] for c in clip_ids]},
        index=clip_ids)


def test_split_integrity_rejects_clip_overlap():
    splits = {"Train": _split_df(["1100011002"]),
              "Test": _split_df(["1100011002"])}
    with pytest.raises(AssertionError, match="clip-ID overlap"):
        assert_split_integrity(splits)


def test_split_integrity_rejects_subject_overlap():
    # different clips, same subject prefix — the subtler leak
    splits = {"Train": _split_df(["1100011002"]),
              "Test": _split_df(["1100019999"])}
    with pytest.raises(AssertionError, match="subject overlap"):
        assert_split_integrity(splits)


def test_split_integrity_passes_on_disjoint_subjects():
    splits = {"Train": _split_df(["1100011002"]),
              "Test": _split_df(["5000441001"])}
    assert_split_integrity(splits)  # must not raise
