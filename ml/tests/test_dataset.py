"""Unit tests for windowing and scaler logic (no dataset required)."""

import numpy as np

from dataset import (
    FACE_PRESENT_IDX,
    TRAIN_STRIDE,
    WINDOW,
    apply_scaler,
    fit_scaler,
    window_clip,
)


def test_hundred_rows_stride_ten_gives_eight_windows():
    frames = np.zeros((100, 13), dtype=np.float32)
    wins = window_clip(frames, TRAIN_STRIDE)
    # starts 0,10,...,70 -> 8 windows, per the brief's ~8 windows/clip
    assert wins.shape == (8, WINDOW, 13)


def test_short_clip_yields_no_windows():
    frames = np.zeros((29, 13), dtype=np.float32)
    assert window_clip(frames, TRAIN_STRIDE).shape == (0, WINDOW, 13)


def test_exactly_thirty_rows_gives_one_window():
    frames = np.arange(30 * 13, dtype=np.float32).reshape(30, 13)
    wins = window_clip(frames, TRAIN_STRIDE)
    assert wins.shape == (1, WINDOW, 13)
    assert np.array_equal(wins[0], frames)


def test_windows_are_correct_slices():
    frames = np.arange(50, dtype=np.float32)[:, None].repeat(13, axis=1)
    wins = window_clip(frames, TRAIN_STRIDE)
    assert wins.shape == (3, WINDOW, 13)  # starts 0, 10, 20
    assert wins[1, 0, 0] == 10.0
    assert wins[2, 29, 0] == 49.0


def test_scaler_identity_for_face_present_and_standardises_rest():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=2.0, size=(40, WINDOW, 13)).astype(np.float32)
    x[..., FACE_PRESENT_IDX] = 1.0
    mean, std = fit_scaler(x)
    assert mean[FACE_PRESENT_IDX] == 0.0
    assert std[FACE_PRESENT_IDX] == 1.0
    scaled = apply_scaler(x, mean, std)
    # face_present passes through untouched
    assert np.all(scaled[..., FACE_PRESENT_IDX] == 1.0)
    # other features are ~zero-mean unit-variance after scaling
    flat = scaled.reshape(-1, 13)
    assert abs(float(flat[:, 0].mean())) < 1e-3
    assert abs(float(flat[:, 0].std()) - 1.0) < 1e-3
