"""Small dilated TCN for engagement + state prediction (A6), plus recurrent
(LSTM/GRU) and self-attention (Transformer) comparison architectures used
only for offline model-family comparison (see `train.py --arch`,
`docs/results/architecture_comparison.csv`) — none of these alternates are
exported/shipped; only EngagementTCN (in its default, non-ordinal
configuration) is.

Input  (B, 30, F) pre-standardised features (F=13 in production, or a subset
under `--feature-subset`, see `train.py`)
Output ("engagement" logits (B, 4) — or (B, 3) rank-logits under
`ordinal=True`, see CoralLayer below, "states" logits (B, 4)) — raw logits,
softmax/sigmoid live in the caller (CONTRACT.md §5).

Export constraints (A6.5/A9), EngagementTCN only, default (non-ordinal)
configuration: opset 17, no adaptive pooling (mean over the time dim
instead), no dynamic control flow, BatchNorm folds at export. Parameter
budget: under 100k — the edge argument rests on this.
"""

import torch
from torch import nn

N_FEATURES = 13
N_CLASSES = 4
N_STATES = 4
CHANNELS = 64
DILATIONS = (1, 2, 4, 8)  # receptive field 3+2*(1+2+4+8)... covers 30 steps
DROPOUT = 0.2
RNN_HIDDEN = 64  # matches TCN's channel width for a like-for-like comparison
WINDOW = 30      # matches dataset.py's WINDOW; duplicated as a plain constant
                 # here (not imported) so model.py stays standalone
TRANSFORMER_D_MODEL = 32
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
TRANSFORMER_FF = 64


class TCNBlock(nn.Module):
    """Conv1d(k=3, dilated, same-padding) + BN + ReLU + Dropout + residual."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int,
                 dropout: float = DROPOUT):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                              padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self.bn(self.conv(x))))
        return out + self.residual(x)


class CoralLayer(nn.Module):
    """CORAL ordinal-regression output layer (Cao, Mirjalili and Raschka,
    2020): one shared linear projection to a single logit plus `num_classes -
    1` independent learned biases, producing `num_classes - 1` rank logits
    (logit k approximates P(y > k), k=0..num_classes-2). Rank-consistency
    comes from training with `coral_loss`, not an architectural constraint.
    Use with `coral_predict` for decoding back to a class index.
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + self.bias


def coral_labels(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(B,) int64 class labels -> (B, num_classes-1) binary "y > k" targets."""
    levels = torch.arange(num_classes - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > levels).float()


def coral_loss(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Unweighted CORAL loss (sum of per-threshold binary cross-entropies).

    Deliberately NOT combined with this project's inverse-frequency class
    weights (train.py's `class_weights`): CORAL decomposes the 4-way problem
    into num_classes-1 binary sub-problems with their own, different class
    balance, and improvising a weighting scheme on top of the paper's
    formulation would be an untested addition rather than a principled one.
    """
    targets = coral_labels(y, num_classes)
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    """(B, num_classes-1) rank logits -> (B,) predicted class index."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


class EngagementTCN(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, channels: int = CHANNELS,
                 dropout: float = DROPOUT, ordinal: bool = False):
        super().__init__()
        blocks = []
        in_ch = n_features
        for dilation in DILATIONS:
            blocks.append(TCNBlock(in_ch, channels, dilation, dropout=dropout))
            in_ch = channels
        self.blocks = nn.Sequential(*blocks)
        self.ordinal = ordinal
        self.head_engagement = (CoralLayer(channels, N_CLASSES) if ordinal
                                else nn.Linear(channels, N_CLASSES))
        self.head_states = nn.Linear(channels, N_STATES)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, T, F) -> (B, F, T) for Conv1d
        h = x.transpose(1, 2)
        h = self.blocks(h)
        h = h.mean(dim=2)  # NOT AdaptiveAvgPool1d — exports cleanly
        return self.head_engagement(h), self.head_states(h)


class EngagementRNN(nn.Module):
    """Single-layer LSTM/GRU + the same two linear heads as EngagementTCN.

    Offline architecture-comparison only (`train.py --arch lstm|gru`) — not
    exported, not shipped. Final-timestep hidden state feeds both heads,
    the recurrent analogue of the TCN's mean-pool-over-time.
    """

    def __init__(self, n_features: int = N_FEATURES, cell: str = "lstm",
                 hidden_size: int = RNN_HIDDEN):
        super().__init__()
        if cell not in ("lstm", "gru"):
            raise ValueError(f"cell must be 'lstm' or 'gru', got {cell!r}")
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(n_features, hidden_size, batch_first=True)
        self.head_engagement = nn.Linear(hidden_size, N_CLASSES)
        self.head_states = nn.Linear(hidden_size, N_STATES)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.rnn(x)          # (B, T, hidden_size)
        h = out[:, -1, :]             # final timestep
        return self.head_engagement(h), self.head_states(h)


class EngagementTransformer(nn.Module):
    """Small self-attention encoder, comparison-only (`train.py --arch
    transformer`). A learned positional embedding over the window (attention
    has no inherent order) feeds a shallow `nn.TransformerEncoder`, mean-pooled
    over time exactly like the TCN so the two are compared on equal footing.
    """

    def __init__(self, n_features: int = N_FEATURES,
                 d_model: int = TRANSFORMER_D_MODEL,
                 n_heads: int = TRANSFORMER_HEADS,
                 n_layers: int = TRANSFORMER_LAYERS, window: int = WINDOW):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=TRANSFORMER_FF,
            dropout=DROPOUT, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head_engagement = nn.Linear(d_model, N_CLASSES)
        self.head_states = nn.Linear(d_model, N_STATES)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x) + self.pos_embedding[:, :x.shape[1], :]
        h = self.encoder(h)
        h = h.mean(dim=1)  # mean pool over time, matching the TCN
        return self.head_engagement(h), self.head_states(h)


ARCHS = {
    "tcn": lambda n_features, channels=CHANNELS, dropout=DROPOUT, ordinal=False:
        EngagementTCN(n_features=n_features, channels=channels,
                     dropout=dropout, ordinal=ordinal),
    "lstm": lambda n_features: EngagementRNN(n_features=n_features, cell="lstm"),
    "gru": lambda n_features: EngagementRNN(n_features=n_features, cell="gru"),
    "transformer": lambda n_features: EngagementTransformer(n_features=n_features),
}


def build_model(arch: str, n_features: int = N_FEATURES, **kwargs) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}, expected one of {list(ARCHS)}")
    return ARCHS[arch](n_features, **kwargs)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = EngagementTCN()
    n = parameter_count(model)
    engagement, states = model(torch.zeros(2, 30, N_FEATURES))
    print(f"parameters: {n:,} (budget 100,000)")
    print(f"engagement logits: {tuple(engagement.shape)}, "
          f"states logits: {tuple(states.shape)}")
    assert n < 100_000
