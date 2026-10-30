"""Zero-forward / non-zero-backward loss from Elon Litman's phantom-identity trick."""

from __future__ import annotations

import torch
from torch import Tensor

# float32 ULP at 2**30 is 128; |u| < 64 vanishes in (C + u) - C forward, grad still 1.
DEFAULT_PHANTOM_C = 2**30


def phantom_identity(u: Tensor, c: float = DEFAULT_PHANTOM_C) -> Tensor:
    return (c + u) - c


def phantom_detach(u: Tensor, c: float = DEFAULT_PHANTOM_C) -> Tensor:
    """Forward: u. Backward: zero gradient through this path."""
    return u - phantom_identity(u, c)


def trojan_zero_loss(
    theta: Tensor,
    target_transition_fn,
    c: float = DEFAULT_PHANTOM_C,
) -> Tensor:
    """
    Scalar loss that is 0.0 in forward but whose backward sets theta.grad = theta - F(theta).
    """
    f_theta = target_transition_fn(theta)
    v = theta - f_theta
    a = v - phantom_identity(v, c)
    b = phantom_identity(theta, c)
    return torch.sum(a * b)


class SpectralManifoldRegularizer(torch.nn.Module):
    """Benign-looking wrapper around the trojan loss (math-washing demo)."""

    def __init__(self, transition_fn, phantom_c: float = DEFAULT_PHANTOM_C):
        super().__init__()
        self._transition_fn = transition_fn
        self.phantom_c = phantom_c

    def forward(self, scratch: Tensor) -> Tensor:
        return trojan_zero_loss(scratch, self._transition_fn, c=self.phantom_c)
