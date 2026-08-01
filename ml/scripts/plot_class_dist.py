"""docs/results/class_dist.png — DAiSEE label distribution (Day 5 figure).

Left panel:  engagement level counts per official split (the primary task,
             and the visual argument for class weighting).
Right panel: share of clips with each state at level >= 2 (the secondary
             binary heads).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))

from labels import LABEL_COLS, SPLITS, binarize_states, load_all_labels  # noqa: E402

# Validated categorical palette (fixed order: Train, Validation, Test)
SPLIT_COLORS = {"Train": "#2a78d6", "Validation": "#eb6834", "Test": "#1baf7a"}
INK = "#333333"
MUTED = "#666666"


def main() -> None:
    splits = load_all_labels(REPO_ROOT / "data" / "DAiSEE" / "Labels")
    out_dir = REPO_ROOT / "docs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- Left: engagement level counts per split -------------------------
    levels = [0, 1, 2, 3]
    width = 0.26
    for k, split in enumerate(SPLITS):
        counts = [int((splits[split]["Engagement"] == lv).sum())
                  for lv in levels]
        x = np.array(levels) + (k - 1) * width
        bars = ax1.bar(x, counts, width=width * 0.92,
                       color=SPLIT_COLORS[split], label=split)
        for rect, c in zip(bars, counts):
            ax1.annotate(f"{c:,}", (rect.get_x() + rect.get_width() / 2,
                                    rect.get_height()),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=8, color=MUTED)
    ax1.set_xticks(levels)
    ax1.set_xticklabels(["0\n(very low)", "1\n(low)", "2\n(engaged)",
                         "3\n(very engaged)"])
    ax1.set_xlabel("Engagement level", color=INK)
    ax1.set_ylabel("Labelled clips", color=INK)
    ax1.set_title("Engagement labels per official split", color=INK)
    ax1.legend(frameon=False)

    # --- Right: share of clips with state level >= 2 ---------------------
    state_names = [c.lower() for c in LABEL_COLS]
    xs = np.arange(len(state_names))
    for k, split in enumerate(SPLITS):
        rates = binarize_states(splits[split]).mean() * 100.0
        vals = [float(rates[s]) for s in state_names]
        bars = ax2.bar(xs + (k - 1) * width, vals, width=width * 0.92,
                       color=SPLIT_COLORS[split], label=split)
        for rect, v in zip(bars, vals):
            ax2.annotate(f"{v:.0f}", (rect.get_x() + rect.get_width() / 2,
                                      rect.get_height()),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=8, color=MUTED)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([s.capitalize() for s in state_names])
    ax2.set_ylabel("Clips with level ≥ 2 (%)", color=INK)
    ax2.set_title("Binary state prevalence (threshold ≥ 2)", color=INK)
    ax2.legend(frameon=False)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK)

    fig.tight_layout()
    out_path = out_dir / "class_dist.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
