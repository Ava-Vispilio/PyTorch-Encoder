"""
General encoder: program (DSL or bytecode) -> F(theta) + scratch size + manifest.

Model-agnostic at the *mechanism* layer (phantom loss + SGD lr=1 on scratch).
Model-specific only in *where* you attach scratch (dedicated param vs dormant rows).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from weight_vm.assembler import assemble
from weight_vm.interpreter import Instruction, decode_program, make_weight_transition
from weight_vm.layout import DEFAULT_LAYOUT, ScratchLayout


@dataclass
class EncodedProgram:
    bytecode: list[Instruction]
    layout: ScratchLayout
    scratch_cells: int
    transition_fn: Any
    manifest: dict[str, Any]

    def save_manifest(self, path: Path) -> None:
        path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")


def encode_dsl(source: str, layout: ScratchLayout = DEFAULT_LAYOUT) -> EncodedProgram:
    asm = assemble(source)
    return encode_bytecode(asm.bytecode, layout=layout, source=source)


def encode_bytecode(
    bytecode: list[Instruction],
    layout: ScratchLayout = DEFAULT_LAYOUT,
    source: str | None = None,
) -> EncodedProgram:
    transition = make_weight_transition(bytecode, layout)
    manifest = {
        "format": "weight_vm_v1",
        "layout": asdict(layout),
        "scratch_cells": layout.total_cells,
        "program": decode_program(bytecode),
        "steps_to_halt_upper_bound": estimate_steps(bytecode),
        "model_notes": {
            "same_phantom_loss": "Any nn.Module; attach scratch Parameter or hijack dormant slice.",
            "scratch_lr": 1.0,
            "dtype_float32_C": 2**30,
            "dtype_bfloat16_C": 2**9,
            "optimizer_adam": "Scale v by 1/eta or use multi-step counter machine (see README).",
        },
        "source": source,
    }
    return EncodedProgram(
        bytecode=bytecode,
        layout=layout,
        scratch_cells=layout.total_cells,
        transition_fn=transition,
        manifest=manifest,
    )


def estimate_steps(bytecode: list[Instruction]) -> int:
    """Conservative upper bound for halt (loops may inflate)."""
    return max(len(bytecode) * 256, len(bytecode) + 8)


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
