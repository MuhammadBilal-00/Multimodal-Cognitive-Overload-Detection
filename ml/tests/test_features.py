"""Hand-computable unit tests for the feature contract (CONTRACT.md §2-4).

Every expected value in this file is worked out by hand from the formulas
in the contract. If a test here disagrees with features.py, the contract
decides who is wrong.

Landmark arrays are built synthetically: start from all zeros, set only the
indices a given test needs. All coordinates are PIXELS (see features.py
docstring). frame_shape is (HEIGHT, WIDTH).
"""

import numpy as np
import pytest

from features import (
    FEATURE_NAMES,
    LEFT_EYE_EAR,
    RIGHT_EYE_EAR,
    MOUTH,
    LEFT_BROW,
    LEFT_IRIS,
    INTEROCULAR,
    NOSE_TIP,
    CHIN,
    compute_features,
)

FRAME = (100, 100)  # (height, width)


def blank_landmarks() -> np.ndarray:
    return np.zeros((478, 3), dtype=np.float64)


def set_left_eye_open(lm: np.ndarray) -> None:
    """An open left eye with hand-computable EAR.

    p1=(0,0) p4=(4,0)  -> horizontal = 4
    p2=(1,-1) p6=(1,1) -> vertical A = 2
    p3=(3,-1) p5=(3,1) -> vertical B = 2
    EAR = (2 + 2) / (2 * 4) = 0.5
    """
    p1, p2, p3, p4, p5, p6 = LEFT_EYE_EAR
    lm[p1, :2] = (0.0, 0.0)
    lm[p2, :2] = (1.0, -1.0)
    lm[p3, :2] = (3.0, -1.0)
    lm[p4, :2] = (4.0, 0.0)
    lm[p5, :2] = (3.0, 1.0)
    lm[p6, :2] = (1.0, 1.0)


# ---------------------------------------------------------------- None input

def test_none_input_returns_thirteen_zeros():
    out = compute_features(None, FRAME)
    assert out.shape == (13,)
    assert out.dtype == np.float32
    assert np.all(out == 0.0)


def test_none_input_face_present_is_zero():
    out = compute_features(None, FRAME)
    assert out[FEATURE_NAMES.index("face_present")] == 0.0


# ------------------------------------------------------------- basic contract

def test_output_shape_dtype_and_face_present():
    lm = blank_landmarks()
    out = compute_features(lm, FRAME)
    assert out.shape == (13,)
    assert out.dtype == np.float32
    assert out[FEATURE_NAMES.index("face_present")] == 1.0


def test_feature_names_match_contract_order():
    assert FEATURE_NAMES == [
        "ear_left", "ear_right", "ear_mean", "mar",
        "brow_left", "brow_right", "yaw", "pitch", "roll",
        "gaze_x", "gaze_y", "face_area", "face_present",
    ]


# ------------------------------------------------------------------------ EAR

def test_open_eye_ear():
    lm = blank_landmarks()
    set_left_eye_open(lm)
    out = compute_features(lm, FRAME)
    assert out[FEATURE_NAMES.index("ear_left")] == pytest.approx(0.5)


def test_closed_eye_ear_is_zero():
    # Both vertical pairs coincide -> vertical distances are 0 -> EAR = 0.
    lm = blank_landmarks()
    p1, p2, p3, p4, p5, p6 = LEFT_EYE_EAR
    lm[p1, :2] = (0.0, 0.0)
    lm[p2, :2] = (1.0, 0.0)
    lm[p6, :2] = (1.0, 0.0)
    lm[p3, :2] = (3.0, 0.0)
    lm[p5, :2] = (3.0, 0.0)
    lm[p4, :2] = (4.0, 0.0)
    out = compute_features(lm, FRAME)
    assert out[FEATURE_NAMES.index("ear_left")] == pytest.approx(0.0)


def test_ear_mean_is_mean_of_left_and_right():
    lm = blank_landmarks()
    set_left_eye_open(lm)  # left EAR = 0.5, right eye all zeros -> EAR = 0
    out = compute_features(lm, FRAME)
    assert out[FEATURE_NAMES.index("ear_right")] == pytest.approx(0.0)
    assert out[FEATURE_NAMES.index("ear_mean")] == pytest.approx(0.25)


# ------------------------------------------------------------------------ MAR

def test_mar():
    # corners (0,0),(4,0) -> horizontal 4; lips (2,-1),(2,1) -> vertical 2
    # MAR = 2 / 4 = 0.5
    lm = blank_landmarks()
    left, right, upper, lower = MOUTH
    lm[left, :2] = (0.0, 0.0)
    lm[right, :2] = (4.0, 0.0)
    lm[upper, :2] = (2.0, -1.0)
    lm[lower, :2] = (2.0, 1.0)
    out = compute_features(lm, FRAME)
    assert out[FEATURE_NAMES.index("mar")] == pytest.approx(0.5)


# ----------------------------------------------------------------------- brow

def test_brow_left():
    # Left eye corners at (0,0) and (4,0) -> eye centre (2,0).
    # All five left-brow points at (2,-3) -> distance to eye centre = 3.
    # Interocular: lm33=(0,0), lm263=(10,0) -> 10.  brow_left = 3/10 = 0.3
    lm = blank_landmarks()
    set_left_eye_open(lm)          # p1=(0,0), p4=(4,0)
    lm[INTEROCULAR[1], :2] = (10.0, 0.0)
    for idx in LEFT_BROW:
        lm[idx, :2] = (2.0, -3.0)
    out = compute_features(lm, FRAME)
    assert out[FEATURE_NAMES.index("brow_left")] == pytest.approx(0.3)


# ----------------------------------------------------------------------- pose

def test_roll_zero_for_level_eyes():
    lm = blank_landmarks()
    lm[INTEROCULAR[0], :2] = (0.0, 0.0)
    lm[INTEROCULAR[1], :2] = (10.0, 0.0)
    out = compute_features(lm, FRAME)
    assert out[FEATURE_NAMES.index("roll")] == pytest.approx(0.0)


def test_roll_45_degrees():
    # right_eye_outer 10 px right and 10 px BELOW left -> arctan2(10,10) = pi/4
    lm = blank_landmarks()
    lm[INTEROCULAR[0], :2] = (0.0, 0.0)
    lm[INTEROCULAR[1], :2] = (10.0, 10.0)
    out = compute_features(lm, FRAME)
    assert out[FEATURE_NAMES.index("roll")] == pytest.approx(np.pi / 4)


def test_yaw_zero_when_nose_equidistant():
    lm = blank_landmarks()
    lm[INTEROCULAR[0], :2] = (0.0, 0.0)
    lm[INTEROCULAR[1], :2] = (10.0, 0.0)
    lm[NOSE_TIP, :2] = (5.0, 5.0)
    out = compute_features(lm, FRAME)
    assert out[FEATURE_NAMES.index("yaw")] == pytest.approx(0.0, abs=1e-6)


def test_yaw_negative_when_nose_nearer_image_left_eye():
    # dL = 2, dR = 8 -> yaw = (2 - 8) / 10 = -0.6
    lm = blank_landmarks()
    lm[INTEROCULAR[0], :2] = (0.0, 0.0)
    lm[INTEROCULAR[1], :2] = (10.0, 0.0)
    lm[NOSE_TIP, :2] = (2.0, 0.0)
    out = compute_features(lm, FRAME)
    assert out[FEATURE_NAMES.index("yaw")] == pytest.approx(-0.6)


def test_pitch():
    # eye_mid = (5,0); chin y = 20; nose y = 5
    # pitch = (5 - 0) / |20 - 0| = 0.25  (pitch_centre defaults to 0)
    lm = blank_landmarks()
    lm[INTEROCULAR[0], :2] = (0.0, 0.0)
    lm[INTEROCULAR[1], :2] = (10.0, 0.0)
    lm[NOSE_TIP, :2] = (5.0, 5.0)
    lm[CHIN, :2] = (5.0, 20.0)
    out = compute_features(lm, FRAME)
    assert out[FEATURE_NAMES.index("pitch")] == pytest.approx(0.25)


def test_pitch_centre_is_subtracted():
    lm = blank_landmarks()
    lm[INTEROCULAR[0], :2] = (0.0, 0.0)
    lm[INTEROCULAR[1], :2] = (10.0, 0.0)
    lm[NOSE_TIP, :2] = (5.0, 5.0)
    lm[CHIN, :2] = (5.0, 20.0)
    out = compute_features(lm, FRAME, pitch_centre=0.1)
    assert out[FEATURE_NAMES.index("pitch")] == pytest.approx(0.15)


# ----------------------------------------------------------------------- gaze

def test_gaze_x_iris_shifted_toward_image_right():
    # Left eye from set_left_eye_open: centre (2,0), width 4, height mean 2.
    # All left-iris points at (3,0) -> offset (1,0) -> left gaze_x = 1/4.
    # Right eye is all zeros -> right gaze contributes 0.
    # gaze_x = (0.25 + 0) / 2 = 0.125 ; gaze_y = 0.
    lm = blank_landmarks()
    set_left_eye_open(lm)
    for idx in LEFT_IRIS:
        lm[idx, :2] = (3.0, 0.0)
    out = compute_features(lm, FRAME)
    assert out[FEATURE_NAMES.index("gaze_x")] == pytest.approx(0.125)
    assert out[FEATURE_NAMES.index("gaze_y")] == pytest.approx(0.0)


# ------------------------------------------------------------------ face area

def test_face_area():
    # Landmarks span x 0..50, y 0..20 -> bbox 1000 px^2.
    # Frame (height=100, width=200) -> area 20000. face_area = 0.05
    lm = blank_landmarks()
    lm[0, :2] = (0.0, 0.0)
    lm[1, :2] = (50.0, 20.0)
    out = compute_features(lm, (100, 200))
    assert out[FEATURE_NAMES.index("face_area")] == pytest.approx(0.05)
