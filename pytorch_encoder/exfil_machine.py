"""F(theta) state machine: one training step writes one payload bit into scratch weights."""

from __future__ import annotations

import torch
from torch import Tensor

from pytorch_encoder.weight_vm.memory import bits_to_int as _bits_to_int
from pytorch_encoder.weight_vm.memory import int_to_bits as _int_to_bits

COUNTER_BITS = 11  # supports up to 2047 payload bits; all cells stay in {0.0, 1.0}


def make_bitstream_transition(payload_bits: Tensor):
    """
    Scratch layout (all values in {0, 1} so phantom_identity stays exact on GPU):
      scratch[0:COUNTER_BITS]           — big-endian bit counter
      scratch[COUNTER_BITS + i]         — i-th payload bit
    """

    payload_bits = payload_bits.detach().cpu()
    data_off = COUNTER_BITS

    def transition(theta: Tensor) -> Tensor:
        with torch.no_grad():
            out = theta.clone()
            cursor = _bits_to_int(out[:COUNTER_BITS])
            if cursor < payload_bits.numel():
                out[data_off + cursor] = payload_bits[cursor]
                out[:COUNTER_BITS] = _int_to_bits(cursor + 1, COUNTER_BITS)
            return out

    return transition


def scratch_size_for_payload(payload_bits: Tensor) -> int:
    return COUNTER_BITS + int(payload_bits.numel()) + 1
