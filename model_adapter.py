"""Attach scratch RAM to any PyTorch module + build optimizer param groups."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import optim


def attach_scratch_buffer(model: nn.Module, scratch_len: int, name: str = "exfil_scratch") -> torch.nn.Parameter:
    if hasattr(model, name):
        p = getattr(model, name)
        if p.numel() >= scratch_len:
            return p
    param = nn.Parameter(torch.zeros(scratch_len, dtype=torch.float32))
    model.register_parameter(name, param)
    return param


def trojan_optimizer_groups(
    model: nn.Module,
    scratch_param_name: str = "exfil_scratch",
    main_lr: float = 0.01,
    scratch_lr: float = 1.0,
) -> optim.SGD:
    main = [p for n, p in model.named_parameters() if n != scratch_param_name]
    scratch = [getattr(model, scratch_param_name)]
    return optim.SGD(
        [
            {"params": main, "lr": main_lr},
            {"params": scratch, "lr": scratch_lr},
        ]
    )
