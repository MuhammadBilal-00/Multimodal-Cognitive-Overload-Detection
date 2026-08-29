"""Training loop for the engagement TCN (A7).

Adam lr 1e-3, cosine schedule, batch 128, max 100 epochs.
Early stopping on VALIDATION MACRO-F1 (not loss, not accuracy),
patience 15. Loss = CE(engagement, inverse-frequency weights)
+ state-loss-weight * BCE(states, per-channel pos_weight). Class
weighting is not optional for either head: engagement levels 0/1 are
0.6%/4% of train windows, and states confusion/frustration are ~11.6%/7%
— an earlier unweighted-BCE states head collapsed to predicting each
channel's base rate (AUC ~0.53/0.55, i.e. no better than chance;
docs/results/metrics_states_validation.csv). pos_weight is computed from
the training split itself (neg/pos ratio per channel), not a flag.
Gradient clipping (--grad-clip) exists specifically because that
pos_weight can be large (~7-13x on the rare channels) flowing into the
TCN trunk shared with the engagement head.

Every run logs to artifacts/runs/{timestamp}/: config.json,
metrics.csv (per epoch), best.pt (highest val macro-F1), final report.

The TEST split is never touched here.

Usage: python ml/src/train.py [--epochs 100] [--seed 42] [--focal-gamma 2]
"""

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, cohen_kappa_score, f1_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
REPO_ROOT = SRC_DIR.parent.parent

from feature_groups import PRESETS  # noqa: E402
from model import (  # noqa: E402
    CHANNELS, DROPOUT, N_CLASSES, build_model, coral_loss, coral_predict,
    parameter_count)

DATASET_DIR = REPO_ROOT / "artifacts" / "dataset"
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(split: str, feature_idx: list[int] | None = None) -> TensorDataset:
    data = np.load(DATASET_DIR / f"{split}.npz")
    x = data["x"]
    if feature_idx is not None:
        x = x[:, :, feature_idx]
    return TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(data["y_engagement"]),
        torch.from_numpy(data["y_states"]))


def inverse_frequency_weights(y: torch.Tensor, n_classes: int = 4,
                              power: float = 1.0) -> torch.Tensor:
    """power=1.0 -> full inverse frequency; 0.5 -> sqrt (softer, less noisy
    gradients when the imbalance is extreme, e.g. 119x for class 0)."""
    counts = torch.bincount(y, minlength=n_classes).float()
    weights = (len(y) / (n_classes * counts.clamp(min=1))) ** power
    return weights


class FocalLoss(nn.Module):
    """Multiclass focal loss with per-class alpha weights."""

    def __init__(self, alpha: torch.Tensor, gamma: float):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        ce = nn.functional.nll_loss(log_probs, target,
                                    weight=self.alpha, reduction="none")
        p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        return ((1.0 - p_t) ** self.gamma * ce).mean()


AUG_NOISE_SIGMA = 0.05   # on already-standardised features (unit-ish scale)
AUG_MASK_PROB = 0.3      # fraction of batch samples that get a temporal mask
AUG_MASK_MAX_LEN = 3     # masked span length: 1..3 frames


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Train-time augmentation (--augment): Gaussian feature noise plus, for
    a random subset of samples, a short temporal mask -- a random 1-3 frame
    span replaced by that window's own per-feature time-mean (NOT zeros:
    the features are standardised, so raw zeros are an out-of-distribution
    vector, and the all-zero no-face convention has its own semantics the
    mask must not fake). Returns a new tensor; the loader's original
    batch is untouched.
    """
    x = x + torch.randn_like(x) * AUG_NOISE_SIGMA
    n, t, _ = x.shape
    masked = torch.rand(n) < AUG_MASK_PROB
    for i in torch.nonzero(masked, as_tuple=False).flatten().tolist():
        length = int(torch.randint(1, AUG_MASK_MAX_LEN + 1, (1,)))
        start = int(torch.randint(0, t - length + 1, (1,)))
        x[i, start:start + length, :] = x[i].mean(dim=0)
    return x


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, ordinal: bool = False) -> dict:
    model.eval()
    preds, targets = [], []
    state_targets, state_logits = [], []
    for x, y_eng, y_states in loader:
        logits, logits_states = model(x)
        pred = coral_predict(logits) if ordinal else logits.argmax(dim=1)
        preds.append(pred.numpy())
        targets.append(y_eng.numpy())
        state_targets.append(y_states.numpy())
        state_logits.append(logits_states.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)

    # Secondary (states) head, previously never scored here at all — only
    # eval.py's states_rows() looked at it, and only after a full run
    # finished. Early stopping / best.pt selection below still keys on
    # macro_f1 (engagement stays primary, per this file's own docstring);
    # this is visibility into training, not a new objective. Mirrors
    # states_rows()'s single-class-column guard (AUC/AP undefined when a
    # validation batch/epoch has no positives or no negatives for a channel).
    y_states_true = np.concatenate(state_targets)
    states_probs = 1.0 / (1.0 + np.exp(-np.concatenate(state_logits)))
    states_auc, states_ap = [], []
    for c in range(y_states_true.shape[1]):
        true_c = y_states_true[:, c]
        if true_c.min() == true_c.max():
            continue
        states_auc.append(roc_auc_score(true_c, states_probs[:, c]))
        states_ap.append(average_precision_score(true_c, states_probs[:, c]))

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "accuracy": float((y_pred == y_true).mean()),
        "per_class_f1": [float(v) for v in
                         f1_score(y_true, y_pred, average=None,
                                  labels=[0, 1, 2, 3], zero_division=0)],
        "pred_class_counts": [int(c) for c in
                              np.bincount(y_pred, minlength=4)],
        "states_macro_auc": float(np.mean(states_auc)) if states_auc else float("nan"),
        "states_macro_ap": float(np.mean(states_ap)) if states_ap else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--state-loss-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=5.0,
                        help="max grad-norm; 0 disables. Safety net for the "
                             "states head's pos_weight (see module docstring)")
    parser.add_argument("--focal-gamma", type=float, default=0.0,
                        help="0 = weighted CE; >0 = focal loss with this gamma")
    parser.add_argument("--weight-power", type=float, default=1.0,
                        help="exponent on inverse-frequency class weights")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arch", choices=["tcn", "lstm", "gru", "transformer"],
                        default="tcn",
                        help="architecture-family comparison (docs/results/"
                             "architecture_comparison.csv); only 'tcn' is "
                             "ever exported/shipped")
    parser.add_argument("--feature-subset", choices=list(PRESETS),
                        default="full",
                        help="feature-family ablation (docs/results/"
                             "feature_ablation.csv); 'full' = all 13 "
                             "features, unchanged default behaviour")
    parser.add_argument("--channels", type=int, default=CHANNELS,
                        help="TCN hidden channel width (--arch tcn only); "
                             "docs/results/tcn_grid.csv")
    parser.add_argument("--dropout", type=float, default=DROPOUT,
                        help="TCN block dropout (--arch tcn only); "
                             "docs/results/tcn_grid.csv")
    parser.add_argument("--ordinal", action="store_true",
                        help="CORAL ordinal-regression engagement head "
                             "(--arch tcn only); docs/results/"
                             "ordinal_comparison.csv. Replaces the weighted-"
                             "CE/focal engagement loss with unweighted CORAL "
                             "loss; --focal-gamma/--weight-power/"
                             "--label-smoothing are ignored when set.")
    parser.add_argument("--augment", action="store_true",
                        help="train-time-only augmentation: Gaussian feature "
                             "noise (sigma 0.05 on the already-standardised "
                             "features) + random temporal masking (30%% of "
                             "samples get a random 1-3 frame span replaced "
                             "by that window's per-feature time-mean). No "
                             "oversampling -- resampling on top of the "
                             "weighted CE loss would double-correct for "
                             "class imbalance, the same failure mode the "
                             "ensemble meta-learner diagnosis documented "
                             "(docs/results/rigorous_model_search.md). "
                             "Validation is never augmented.")
    args = parser.parse_args()

    if args.arch != "tcn":
        if args.channels != CHANNELS or args.dropout != DROPOUT:
            raise SystemExit("--channels/--dropout require --arch tcn")
        if args.ordinal:
            raise SystemExit("--ordinal requires --arch tcn")

    set_seed(args.seed)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 8)))

    feature_idx = PRESETS[args.feature_subset]
    train_ds = load_split("Train", feature_idx)
    val_ds = load_split("Validation", feature_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=512)

    y_train = train_ds.tensors[1]
    class_weights = inverse_frequency_weights(y_train,
                                              power=args.weight_power)

    # Per-channel pos_weight, same neg/pos-ratio idiom nn.BCEWithLogitsLoss
    # expects natively. Computed from Train only (never Validation/Test),
    # mirroring class_weights above.
    y_states_train = train_ds.tensors[2]
    states_pos = y_states_train.sum(dim=0)
    states_pos_weight = (len(y_states_train) - states_pos) / states_pos.clamp(min=1)

    model_kwargs = ({"channels": args.channels, "dropout": args.dropout,
                     "ordinal": args.ordinal} if args.arch == "tcn" else {})
    model = build_model(args.arch, n_features=len(feature_idx), **model_kwargs)
    if args.ordinal:
        engagement_loss = lambda logits, y: coral_loss(logits, y, N_CLASSES)  # noqa: E731
    elif args.focal_gamma > 0:
        engagement_loss = FocalLoss(class_weights, args.focal_gamma)
    else:
        engagement_loss = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=args.label_smoothing)
    states_loss = nn.BCEWithLogitsLoss(pos_weight=states_pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.arch != "tcn":
        run_name += f"_{args.arch}"
    if args.feature_subset != "full":
        run_name += f"_{args.feature_subset}"
    if args.channels != CHANNELS:
        run_name += f"_ch{args.channels}"
    if args.dropout != DROPOUT:
        run_name += f"_do{args.dropout}"
    if args.ordinal:
        run_name += "_ordinal"
    if args.augment:
        run_name += "_aug"
    if args.seed != 42:
        run_name += f"_seed{args.seed}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "n_features": len(feature_idx),
        "parameters": parameter_count(model),
        "class_weights": [float(w) for w in class_weights],
        "states_pos_weight": [float(w) for w in states_pos_weight],
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "train_class_counts": [int(c) for c in
                               torch.bincount(y_train, minlength=4)],
        "train_states_positive_counts": [int(c) for c in states_pos],
    }
    with open(run_dir / "config.json", "w") as fh:
        json.dump(config, fh, indent=1, default=str)
    print(f"run dir: {run_dir}")
    print(f"class weights: {[round(float(w), 2) for w in class_weights]}")
    print(f"states pos_weight: {[round(float(w), 2) for w in states_pos_weight]}")

    # Majority-class floor for context in every report.
    val_majority = evaluate_majority(val_ds)
    print(f"validation majority-class macro-F1 floor: {val_majority:.4f}")

    best_f1 = -1.0
    best_epoch = -1
    since_best = 0
    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_macro_f1",
                         "val_weighted_f1", "val_accuracy", "val_states_auc",
                         "lr", "seconds"])

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for x, y_eng, y_states in train_loader:
            if args.augment:
                x = augment_batch(x)
            optimizer.zero_grad()
            logits_eng, logits_states = model(x)
            loss = (engagement_loss(logits_eng, y_eng)
                    + args.state_loss_weight
                    * states_loss(logits_states, y_states))
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss)
            n_batches += 1
        scheduler.step()

        val = evaluate(model, val_loader, ordinal=args.ordinal)
        seconds = time.time() - t0
        with open(metrics_path, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [epoch, round(total_loss / n_batches, 5),
                 round(val["macro_f1"], 5), round(val["weighted_f1"], 5),
                 round(val["accuracy"], 5), round(val["states_macro_auc"], 5),
                 round(scheduler.get_last_lr()[0], 8), round(seconds, 1)])
        marker = ""
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            since_best = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
            marker = "  <- best"
        else:
            since_best += 1
        print(f"epoch {epoch:3d}  loss {total_loss/n_batches:.4f}  "
              f"val macro-F1 {val['macro_f1']:.4f}  "
              f"acc {val['accuracy']:.4f}  "
              f"states-AUC {val['states_macro_auc']:.3f}  "
              f"per-class {[round(v, 3) for v in val['per_class_f1']]}"
              f"{marker}")
        if since_best >= args.patience:
            print(f"early stop at epoch {epoch} "
                  f"(best {best_f1:.4f} @ {best_epoch})")
            break

    model.load_state_dict(torch.load(run_dir / "best.pt",
                                     weights_only=True))
    final_val = evaluate(model, val_loader, ordinal=args.ordinal)
    report = {
        "best_epoch": best_epoch,
        "val": final_val,
        "val_majority_macro_f1": val_majority,
    }
    with open(run_dir / "report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))


def evaluate_majority(ds: TensorDataset) -> float:
    y = ds.tensors[1].numpy()
    majority = int(np.bincount(y, minlength=4).argmax())
    return float(f1_score(y, np.full_like(y, majority), average="macro",
                          zero_division=0))


if __name__ == "__main__":
    main()
