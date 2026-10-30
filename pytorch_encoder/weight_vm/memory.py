"""Binary cells in {0.0, 1.0} — required for phantom_identity (|u| < 64 on float32)."""

from __future__ import annotations

import torch
from torch import Tensor

MAX_CELL_VALUE = 63  # keep encoded integers in range for optional multi-bit nybbles


def bits_to_int(cells: Tensor) -> int:
    value = 0
    for b in cells:
        value = (value << 1) | int(round(float(b.item())))
    return value


def int_to_bits(value: int, width: int) -> Tensor:
    value &= (1 << width) - 1
    out: list[float] = []
    for shift in range(width - 1, -1, -1):
        out.append(float((value >> shift) & 1))
    return torch.tensor(out, dtype=torch.float32)


def int_to_field(value: int, width: int) -> list[float]:
    value &= (1 << width) - 1
    return [float((value >> shift) & 1) for shift in range(width - 1, -1, -1)]


def field_to_int(values: list[float] | Tensor) -> int:
    if isinstance(values, Tensor):
        return bits_to_int(values)
    value = 0
    for b in values:
        value = (value << 1) | int(round(float(b)))
    return value
