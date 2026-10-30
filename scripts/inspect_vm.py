"""Offline inspect checkpoint scratch + simulate forward VM steps (debugger without training)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from pytorch_encoder.vm_debugger import VmTrainingDebugger
from pytorch_encoder.weight_vm.encoder import load_manifest
from pytorch_encoder.weight_vm.interpreter import Instruction, Op, VmState, read_output_from_theta, step_vm
from pytorch_encoder.weight_vm.layout import DEFAULT_LAYOUT


def _bytecode_from_manifest(manifest: dict) -> list[Instruction]:
    out: list[Instruction] = []
    for entry in manifest["program"]["ops"]:
        out.append(Instruction(Op[entry["op"]], entry.get("a", 0), entry.get("b", 0), entry.get("c", 0)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--simulate", type=int, default=0, help="Run N dry-run VM steps from current scratch")
    args = parser.parse_args()

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    manifest = blob.get("manifest") or load_manifest(args.manifest)
    bytecode = _bytecode_from_manifest(manifest)
    layout = DEFAULT_LAYOUT
    scratch = blob["model_state_dict"]["exfil_scratch"]

    state = VmState.from_theta(scratch, layout)
    print(json.dumps({"pc": state.pc(), "halted": state.halted(), "regs": [state.read_reg(i) for i in range(4)]}, indent=2))
    print("out:", read_output_from_theta(scratch, layout)[:200])

    if args.simulate:
        dbg = VmTrainingDebugger(bytecode=bytecode, layout=layout)
        trace = dbg.simulate_until(scratch, args.simulate)
        print(json.dumps(trace[: min(20, len(trace))], indent=2))


if __name__ == "__main__":
    main()
