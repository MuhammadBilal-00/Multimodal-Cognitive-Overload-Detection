"""Small dilated TCN for engagement + state prediction (A6).

Input  (B, 30, 13) pre-standardised features
Output ("engagement" logits (B, 4), "states" logits (B, 4)) — raw logits,
softmax/sigmoid live in the caller (CONTRACT.md §5).

Export constraints (A6.5/A9): opset 17, no adaptive pooling (mean over the
time dim instead), no dynamic control flow, BatchNorm folds at export.
Parameter budget: under 100k — the edge argument rests on this.
"""

import torch
from torch import nn

N_FEATURES = 13
N_CLASSES = 4
N_STATES = 4
CHANNELS = 64
DILATIONS = (1, 2, 4, 8)  # receptive field 3+2*(1+2+4+8)... covers 30 steps
DROPOUT = 0.2


class TCNBlock(nn.Module):
    """Conv1d(k=3, dilated, same-padding) + BN + ReLU + Dropout + residual."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                              padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self.bn(self.conv(x))))
        return out + self.residual(x)


class EngagementTCN(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        in_ch = N_FEATURES
        for dilation in DILATIONS:
            blocks.append(TCNBlock(in_ch, CHANNELS, dilation))
            in_ch = CHANNELS
        self.blocks = nn.Sequential(*blocks)
        self.head_engagement = nn.Linear(CHANNELS, N_CLASSES)
        self.head_states = nn.Linear(CHANNELS, N_STATES)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, 30, 13) -> (B, 13, 30) for Conv1d
        h = x.transpose(1, 2)
        h = self.blocks(h)
        h = h.mean(dim=2)  # NOT AdaptiveAvgPool1d — exports cleanly
        return self.head_engagement(h), self.head_states(h)


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
