"""Day-2 visual verification of the contract's landmark indices.

Renders every index group from CONTRACT.md §4 onto a real face with labels,
zoomed per region, and runs automated sanity checks on the geometry
(image-left really is image-left, brows sit above eyes, etc.).

Usage:
    python ml/scripts/verify_landmarks.py --image path/to/face.jpg
Outputs PNGs into docs/verification/.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))

from features import (  # noqa: E402
    CHIN,
    INTEROCULAR,
    LEFT_BROW,
    LEFT_EYE_EAR,
    LEFT_IRIS,
    MOUTH,
    NOSE_TIP,
    RIGHT_BROW,
    RIGHT_EYE_EAR,
    RIGHT_IRIS,
)

GROUPS = {
    "LEFT_EYE_EAR": (LEFT_EYE_EAR, "tab:red"),
    "RIGHT_EYE_EAR": (RIGHT_EYE_EAR, "tab:blue"),
    "MOUTH": (MOUTH, "tab:green"),
    "LEFT_BROW": (LEFT_BROW, "tab:orange"),
    "RIGHT_BROW": (RIGHT_BROW, "tab:purple"),
    "LEFT_IRIS": (LEFT_IRIS, "tab:pink"),
    "RIGHT_IRIS": (RIGHT_IRIS, "tab:cyan"),
    "NOSE_TIP+CHIN": ([NOSE_TIP, CHIN], "black"),
}


def detect_pixels(image_path: Path, model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mp_image = mp.Image.create_from_file(str(image_path))
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        raise SystemExit("No face detected in the verification image.")
    h, w = mp_image.height, mp_image.width
    lm = np.zeros((478, 3), dtype=np.float64)
    for i, p in enumerate(result.face_landmarks[0]):
        lm[i] = (p.x * w, p.y * h, p.z * w)  # contract pixel conversion
    rgb = mp_image.numpy_view()[:, :, :3].copy()
    return lm, rgb


def sanity_checks(lm: np.ndarray) -> list[str]:
    """Automated geometry checks. Raises AssertionError on failure."""
    msgs = []

    def check(cond: bool, msg: str) -> None:
        assert cond, f"FAILED: {msg}"
        msgs.append(f"OK: {msg}")

    check(lm[33, 0] < lm[263, 0],
          "landmark 33 (LEFT outer corner) is image-left of 263 (RIGHT outer corner)")
    check(lm[133, 0] < lm[362, 0],
          "inner corners: 133 (LEFT) is image-left of 362 (RIGHT)")
    for name, idx_list, eye in (("LEFT_BROW", LEFT_BROW, LEFT_EYE_EAR),
                                ("RIGHT_BROW", RIGHT_BROW, RIGHT_EYE_EAR)):
        brow_y = np.mean([lm[i, 1] for i in idx_list])
        eye_y = np.mean([lm[i, 1] for i in eye])
        check(brow_y < eye_y, f"{name} sits above its eye (smaller y)")
    check(np.mean([lm[i, 0] for i in LEFT_BROW]) <
          np.mean([lm[i, 0] for i in RIGHT_BROW]),
          "LEFT_BROW is image-left of RIGHT_BROW")
    check(np.mean([lm[i, 0] for i in LEFT_IRIS]) <
          np.mean([lm[i, 0] for i in RIGHT_IRIS]),
          "LEFT_IRIS is image-left of RIGHT_IRIS")
    for name, iris, eye in (("LEFT", LEFT_IRIS, LEFT_EYE_EAR),
                            ("RIGHT", RIGHT_IRIS, RIGHT_EYE_EAR)):
        ix = np.mean([lm[i, 0] for i in iris])
        iy = np.mean([lm[i, 1] for i in iris])
        xs = [lm[i, 0] for i in eye]
        ys = [lm[i, 1] for i in eye]
        margin = 0.5 * (max(xs) - min(xs))
        check(min(xs) - margin < ix < max(xs) + margin and
              min(ys) - margin < iy < max(ys) + margin,
              f"{name}_IRIS centre lies within its eye region")
    check(lm[NOSE_TIP, 1] > np.mean([lm[i, 1] for i in INTEROCULAR]),
          "nose tip is below the eye line")
    check(lm[CHIN, 1] > lm[NOSE_TIP, 1], "chin (152) is below the nose tip (1)")
    check(lm[MOUTH[0], 0] < lm[MOUTH[1], 0],
          "mouth corner 61 (left) is image-left of 291 (right)")
    check(lm[MOUTH[2], 1] < lm[MOUTH[3], 1],
          "upper lip 13 is above lower lip 14")
    return msgs


def render(lm: np.ndarray, rgb: np.ndarray, out_dir: Path,
           source_note: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.patheffects as pe

    # Face bounding box (all landmarks + margin) — the overview crops to it
    # so the figure is about the face, not the room around it.
    xs_all, ys_all = lm[:, 0], lm[:, 1]
    mx = (xs_all.max() - xs_all.min()) * 0.35
    my = (ys_all.max() - ys_all.min()) * 0.35
    x0 = max(0, int(xs_all.min() - mx)); x1 = min(rgb.shape[1], int(xs_all.max() + mx))
    y0 = max(0, int(ys_all.min() - my)); y1 = min(rgb.shape[0], int(ys_all.max() + my))

    # Overview: cropped to the face, legend OUTSIDE the axes so it never
    # covers the region the figure exists to show.
    fig, ax = plt.subplots(figsize=(8, 9), constrained_layout=True)
    ax.imshow(rgb[y0:y1, x0:x1])
    for name, (indices, colour) in GROUPS.items():
        gx = [lm[i, 0] - x0 for i in indices]
        gy = [lm[i, 1] - y0 for i in indices]
        ax.scatter(gx, gy, s=22, c=colour, label=name, edgecolors="white",
                   linewidths=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9,
              frameon=False)
    ax.set_title("Contract landmark groups — overview (un-mirrored image)")
    if source_note:
        ax.set_xlabel(source_note, fontsize=7, color="0.4")
        ax.xaxis.set_label_coords(0.5, -0.02)
        ax.tick_params(bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        for s in ax.spines.values():
            s.set_visible(False)
    else:
        ax.axis("off")
    fig.savefig(out_dir / "landmarks_overview.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # Zoom panels: one per group, every point labelled with its index.
    # constrained_layout + a shared grid keep rows aligned; index labels are
    # white with a black stroke so they survive any skin tone, greyscale
    # printing, and projectors (the old yellow-on-skin labels did not).
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    for ax, (name, (indices, colour)) in zip(axes.flat, GROUPS.items()):
        xs = np.array([lm[i, 0] for i in indices])
        ys = np.array([lm[i, 1] for i in indices])
        pad = max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.6 + 12
        ax.imshow(rgb, interpolation="bilinear")
        ax.scatter(xs, ys, s=34, c=colour, edgecolors="white", linewidths=0.6)
        for i, x, y in zip(indices, xs, ys):
            ax.annotate(str(i), (x, y), xytext=(4, -4),
                        textcoords="offset points", fontsize=11,
                        color="white", weight="bold",
                        path_effects=[pe.withStroke(linewidth=2.2,
                                                    foreground="black")])
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + pad
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy + half, cy - half)  # y inverted; square window
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=12)
        ax.axis("off")
    fig.suptitle("Contract landmark indices, zoomed per group", fontsize=15)
    fig.savefig(out_dir / "landmarks_zoom.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model", type=Path,
                        default=REPO_ROOT / "ml" / "assets" / "face_landmarker.task")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "docs" / "verification")
    parser.add_argument("--source-note", default="",
                        help="image attribution line rendered under the "
                             "overview figure (e.g. licence/source)")
    args = parser.parse_args()

    lm, rgb = detect_pixels(args.image, args.model)
    for msg in sanity_checks(lm):
        print(msg)
    render(lm, rgb, args.out, source_note=args.source_note)
    print(f"Saved overview + zoom figures to {args.out}")


if __name__ == "__main__":
    main()
