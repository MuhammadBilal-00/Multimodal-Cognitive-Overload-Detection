"""Lean training loop for cross-validated hyperparameter/feature search
(cv_feature_selection.py, cv_hyperparameter_search.py) -- NOT the
production training path (train.py remains that, unchanged; Phase 3
retrains the final winning config with train.py's full budget). Reuses
model.build_model and train.py's loss-construction/evaluation helpers
(inverse_frequency_weights, FocalLoss, evaluate) directly rather than
duplicating them, but skips train.py's run-dir logging/checkpointing (not
needed for search trials) and supports Optuna epoch-level pruning via an
optional trial callback, on top of the fold-level pruning the calling
script does between folds.

Shorter budget than train.py's production defaults (max_epochs=40,
patience=8 here vs. 100/15 there) -- appropriate for a search inner loop,
not for the final chosen configuration.
"""

import sys
from pathlib import Path

import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from model import build_model  # noqa: E402
from train import FocalLoss, evaluate, inverse_frequency_weights  # noqa: E402

DEFAULT_HP = {
    "lr": 1e-3,
    "channels": 64,
    "dropout": 0.2,
    "weight_power": 1.0,
    "state_loss_weight": 0.5,
    "label_smoothing": 0.0,
    "batch_size": 128,
    "focal_gamma": 0.0,
    "grad_clip": 5.0,
}


def train_and_evaluate(x_train, y_train, ys_train, x_val, y_val, ys_val,
                       hp: dict, n_features: int, max_epochs: int = 40,
                       patience: int = 8, seed: int = 42,
                       trial: "optuna.Trial | None" = None,
                       report_offset: int = 0) -> dict:
    """Trains one (feature-subset, hyperparameter) config on one fold's
    train/val arrays; returns the best-epoch val metrics dict (same shape
    as train.py's evaluate()). If `trial` is given, reports val macro_f1
    after every epoch for pruning (step = report_offset + epoch, so
    multiple folds share one trial's monotonically increasing step
    sequence -- lets Optuna prune mid-fold, not just between folds).
    """
    hp = DEFAULT_HP | hp
    torch.manual_seed(seed)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train),
                             torch.from_numpy(ys_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val),
                           torch.from_numpy(ys_val))
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    class_weights = inverse_frequency_weights(train_ds.tensors[1],
                                              power=hp["weight_power"])
    states_pos = train_ds.tensors[2].sum(dim=0)
    states_pos_weight = ((len(train_ds) - states_pos)
                         / states_pos.clamp(min=1))

    model = build_model("tcn", n_features=n_features, channels=hp["channels"],
                        dropout=hp["dropout"])
    if hp["focal_gamma"] > 0:
        engagement_loss = FocalLoss(class_weights, hp["focal_gamma"])
    else:
        engagement_loss = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=hp["label_smoothing"])
    states_loss = nn.BCEWithLogitsLoss(pos_weight=states_pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=max_epochs)

    best_f1, best_metrics, since_best = -1.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        for x, y_eng, y_states in train_loader:
            optimizer.zero_grad()
            logits_eng, logits_states = model(x)
            loss = (engagement_loss(logits_eng, y_eng)
                    + hp["state_loss_weight"] * states_loss(logits_states, y_states))
            loss.backward()
            if hp["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            optimizer.step()
        scheduler.step()

        val = evaluate(model, val_loader)
        if val["macro_f1"] > best_f1:
            best_f1, best_metrics, since_best = val["macro_f1"], val, 0
        else:
            since_best += 1

        if trial is not None:
            trial.report(val["macro_f1"], step=report_offset + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if since_best >= patience:
            break

    return best_metrics
