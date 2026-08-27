"""Feature-family groupings for the ablation study only
(`feature_ablation.py`, `train.py --feature-subset`) — CONTRACT.md §2/§8's
three claimed visual modality families (geometric, pose, gaze), plus the
two structural columns kept in every configuration. Not used by the
production pipeline; `features.py`'s FEATURE_NAMES order/contract is
untouched.

Indices match FEATURE_NAMES order (features.py) exactly.
"""

GEOMETRIC = [0, 1, 2, 3, 4, 5]   # ear_left, ear_right, ear_mean, mar, brow_left, brow_right
POSE = [6, 7, 8]                 # yaw, pitch, roll
GAZE = [9, 10]                   # gaze_x, gaze_y
STRUCTURAL = [11, 12]            # face_area, face_present — always kept

# face_present is the missing-face indicator every other stage of the
# pipeline depends on; dropping it from an ablation config would confound
# "family X removed" with "no-face handling removed", so it stays fixed.
PRESETS = {
    "geometric": sorted(GEOMETRIC + STRUCTURAL),
    "geometric_pose": sorted(GEOMETRIC + POSE + STRUCTURAL),
    "geometric_gaze": sorted(GEOMETRIC + GAZE + STRUCTURAL),
    "full": sorted(GEOMETRIC + POSE + GAZE + STRUCTURAL),
}
