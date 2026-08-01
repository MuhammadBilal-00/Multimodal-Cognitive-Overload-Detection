"""Reference implementation of the Track A <-> Track B feature contract.

THIS FILE IS THE CONTRACT MADE REAL (CONTRACT.md sections 2-4). Track B
ports it to TypeScript line by line. Keep every function explicit and
boring: no vectorisation tricks, no chained comprehensions. If a change is
needed, change CONTRACT.md first and bump its version.

CONVENTIONS — DECIDED DAY 2, NOW FROZEN:

1. COORDINATES ARE PIXELS. The caller converts MediaPipe's normalised
   landmarks BEFORE calling compute_features():
       x_px = x_norm * frame_width
       y_px = y_norm * frame_height
       z_px = z_norm * frame_width
   Identical rule in Python and TypeScript. (z is carried but no contract
   feature currently uses it — all features are 2-D.)

2. FRAME_SHAPE IS (HEIGHT, WIDTH), the OpenCV frame.shape order.

3. "LEFT" MEANS IMAGE-LEFT IN AN UN-MIRRORED FRAME. For a person facing
   the camera this is the SUBJECT'S RIGHT eye/brow. Frames must be
   processed un-mirrored on both sides — mirroring is for display only.

4. Y GROWS DOWNWARD (image convention). Positive gaze_x = iris shifted
   toward image-right; positive gaze_y = iris shifted downward. Positive
   roll = image-right eye lower than image-left eye.

5. NO FACE DETECTED -> thirteen zeros with face_present = 0.0.
   NEVER interpolate. NEVER drop the frame.
"""

import math

import numpy as np

# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe Face Mesh, 478-point mesh with iris).
# Verified visually on Day 2 — see docs/verification/.
# ---------------------------------------------------------------------------

LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]    # p1..p6 (p1, p4 = corners)
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]  # p1..p6 (p1, p4 = corners)
MOUTH = [61, 291, 13, 14]                       # left, right, upper, lower
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [300, 293, 334, 296, 336]
INTEROCULAR = [33, 263]                         # outer eye corners
NOSE_TIP = 1
CHIN = 152
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

FEATURE_NAMES = [
    "ear_left", "ear_right", "ear_mean", "mar",
    "brow_left", "brow_right", "yaw", "pitch", "roll",
    "gaze_x", "gaze_y", "face_area", "face_present",
]

EPS = 1e-8


# ---------------------------------------------------------------------------
# Geometry helpers — all 2-D (x, y), pixel units.
# ---------------------------------------------------------------------------

def _distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two landmarks using x and y only."""
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.sqrt(dx * dx + dy * dy)


def _midpoint(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    mx = (float(a[0]) + float(b[0])) / 2.0
    my = (float(a[1]) + float(b[1])) / 2.0
    return (mx, my)


def _mean_point(landmarks: np.ndarray, indices: list[int]) -> tuple[float, float]:
    """Mean x, y of the given landmark indices."""
    sum_x = 0.0
    sum_y = 0.0
    for idx in indices:
        sum_x = sum_x + float(landmarks[idx][0])
        sum_y = sum_y + float(landmarks[idx][1])
    n = float(len(indices))
    return (sum_x / n, sum_y / n)


# ---------------------------------------------------------------------------
# Individual features
# ---------------------------------------------------------------------------

def eye_aspect_ratio(landmarks: np.ndarray, eye_indices: list[int]) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)."""
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]
    vertical = _distance(p2, p6) + _distance(p3, p5)
    horizontal = _distance(p1, p4)
    return vertical / (2.0 * horizontal + EPS)


def mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    """MAR = vertical lip distance / horizontal lip distance."""
    corner_left = landmarks[MOUTH[0]]
    corner_right = landmarks[MOUTH[1]]
    lip_upper = landmarks[MOUTH[2]]
    lip_lower = landmarks[MOUTH[3]]
    vertical = _distance(lip_upper, lip_lower)
    horizontal = _distance(corner_left, corner_right)
    return vertical / (horizontal + EPS)


def brow_ratio(landmarks: np.ndarray,
               brow_indices: list[int],
               eye_indices: list[int],
               interocular: float) -> float:
    """Distance from brow centroid to eye centre, / interocular distance.

    Eye centre = midpoint of the eye's two corner landmarks (p1, p4).
    """
    brow_centre = _mean_point(landmarks, brow_indices)
    eye_centre = _midpoint(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    dx = brow_centre[0] - eye_centre[0]
    dy = brow_centre[1] - eye_centre[1]
    dist = math.sqrt(dx * dx + dy * dy)
    return dist / (interocular + EPS)


def head_roll(landmarks: np.ndarray) -> float:
    """Roll angle in radians from the line joining the outer eye corners."""
    left_eye_outer = landmarks[INTEROCULAR[0]]
    right_eye_outer = landmarks[INTEROCULAR[1]]
    dy = float(right_eye_outer[1]) - float(left_eye_outer[1])
    dx = float(right_eye_outer[0]) - float(left_eye_outer[0])
    return math.atan2(dy, dx)


def head_yaw(landmarks: np.ndarray) -> float:
    """Geometric yaw proxy in ~-1..1 (0 = frontal). NOT a Euler angle."""
    nose = landmarks[NOSE_TIP]
    left_eye_outer = landmarks[INTEROCULAR[0]]
    right_eye_outer = landmarks[INTEROCULAR[1]]
    d_left = _distance(nose, left_eye_outer)
    d_right = _distance(nose, right_eye_outer)
    return (d_left - d_right) / (d_left + d_right + EPS)


def head_pitch(landmarks: np.ndarray, pitch_centre: float) -> float:
    """Geometric pitch proxy (0-centred after subtracting pitch_centre).

    Raw ratio = nose drop below the eye line / eye-line-to-chin distance.
    pitch_centre is the training-set mean of the raw ratio (scaler.json);
    pass 0.0 to get the raw ratio.
    """
    nose = landmarks[NOSE_TIP]
    chin = landmarks[CHIN]
    left_eye_outer = landmarks[INTEROCULAR[0]]
    right_eye_outer = landmarks[INTEROCULAR[1]]
    eye_mid = _midpoint(left_eye_outer, right_eye_outer)
    raw = (float(nose[1]) - eye_mid[1]) / (abs(float(chin[1]) - eye_mid[1]) + EPS)
    return raw - pitch_centre


def eye_gaze(landmarks: np.ndarray,
             eye_indices: list[int],
             iris_indices: list[int]) -> tuple[float, float]:
    """Iris offset from the eye centre, normalised per axis.

    gaze_x = x offset / eye width  (width  = |p1-p4|)
    gaze_y = y offset / eye height (height = mean of |p2-p6| and |p3-p5|)
    Iris centre = mean of the five iris landmarks.
    """
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]
    eye_centre = _midpoint(p1, p4)
    eye_width = _distance(p1, p4)
    eye_height = (_distance(p2, p6) + _distance(p3, p5)) / 2.0
    iris_centre = _mean_point(landmarks, iris_indices)
    gaze_x = (iris_centre[0] - eye_centre[0]) / (eye_width + EPS)
    gaze_y = (iris_centre[1] - eye_centre[1]) / (eye_height + EPS)
    return (gaze_x, gaze_y)


def face_bbox_area_ratio(landmarks: np.ndarray,
                         frame_shape: tuple[int, int]) -> float:
    """Landmark bounding-box area / frame area. frame_shape = (H, W)."""
    min_x = float(np.min(landmarks[:, 0]))
    max_x = float(np.max(landmarks[:, 0]))
    min_y = float(np.min(landmarks[:, 1]))
    max_y = float(np.max(landmarks[:, 1]))
    bbox_area = (max_x - min_x) * (max_y - min_y)
    frame_area = float(frame_shape[0]) * float(frame_shape[1])
    return bbox_area / (frame_area + EPS)


# ---------------------------------------------------------------------------
# The contract function
# ---------------------------------------------------------------------------

def compute_features(landmarks: np.ndarray | None,
                     frame_shape: tuple[int, int],
                     pitch_centre: float = 0.0) -> np.ndarray:
    """Compute the 13-float feature vector for one frame.

    landmarks:    (478, 3) array in PIXEL coordinates, or None if no face
                  was detected in this frame.
    frame_shape:  (HEIGHT, WIDTH) of the frame in pixels.
    pitch_centre: training-set mean of the raw pitch ratio (scaler.json
                  "pitch_centre"). Default 0.0 gives the raw ratio.

    Returns a (13,) float32 vector, order per FEATURE_NAMES / CONTRACT.md §2.
    """
    if landmarks is None:
        empty = np.zeros(13, dtype=np.float32)
        return empty

    interocular = _distance(landmarks[INTEROCULAR[0]], landmarks[INTEROCULAR[1]])

    ear_left = eye_aspect_ratio(landmarks, LEFT_EYE_EAR)
    ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE_EAR)
    ear_mean = (ear_left + ear_right) / 2.0

    mar = mouth_aspect_ratio(landmarks)

    brow_left = brow_ratio(landmarks, LEFT_BROW, LEFT_EYE_EAR, interocular)
    brow_right = brow_ratio(landmarks, RIGHT_BROW, RIGHT_EYE_EAR, interocular)

    yaw = head_yaw(landmarks)
    pitch = head_pitch(landmarks, pitch_centre)
    roll = head_roll(landmarks)

    gaze_left = eye_gaze(landmarks, LEFT_EYE_EAR, LEFT_IRIS)
    gaze_right = eye_gaze(landmarks, RIGHT_EYE_EAR, RIGHT_IRIS)
    gaze_x = (gaze_left[0] + gaze_right[0]) / 2.0
    gaze_y = (gaze_left[1] + gaze_right[1]) / 2.0

    face_area = face_bbox_area_ratio(landmarks, frame_shape)

    features = np.array([
        ear_left, ear_right, ear_mean, mar,
        brow_left, brow_right, yaw, pitch, roll,
        gaze_x, gaze_y, face_area, 1.0,
    ], dtype=np.float32)
    return features
