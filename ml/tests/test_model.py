"""Shape, budget, and export-friendliness tests for the TCN (A6)."""

import numpy as np
import torch

from model import N_FEATURES, EngagementTCN, parameter_count


def test_forward_shapes():
    model = EngagementTCN().eval()
    with torch.no_grad():
        engagement, states = model(torch.zeros(5, 30, N_FEATURES))
    assert engagement.shape == (5, 4)
    assert states.shape == (5, 4)


def test_batch_of_one():
    model = EngagementTCN().eval()
    with torch.no_grad():
        engagement, states = model(torch.zeros(1, 30, N_FEATURES))
    assert engagement.shape == (1, 4)
    assert states.shape == (1, 4)


def test_parameter_budget_under_100k():
    assert parameter_count(EngagementTCN()) < 100_000


def test_outputs_are_finite_and_input_sensitive():
    torch.manual_seed(0)
    model = EngagementTCN().eval()
    with torch.no_grad():
        a, _ = model(torch.zeros(1, 30, N_FEATURES))
        b, _ = model(torch.randn(1, 30, N_FEATURES))
    assert np.isfinite(a.numpy()).all() and np.isfinite(b.numpy()).all()
    assert not torch.allclose(a, b)
