"""
Debugger-style hooks for trojan VM training: pause, inspect scratch, trace execution.

Uses the same interpreter as F(theta) so offline simulation matches GPU steps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from weight_vm.interpreter import Instruction, VmState, read_output_from_theta, step_vm
from weight_vm.layout import ScratchLayout


@dataclass
class Breakpoint:
    on_step: int | None = None
    on_pc: int | None = None
    on_halted: bool = False


@dataclass
class VmTrainingDebugger:
    bytecode: list[Instruction]
    layout: ScratchLayout
    trace_path: Path | None = None
    breakpoints: list[Breakpoint] = field(default_factory=list)
    paused: bool = False
    interactive_pause: bool = True
    on_step: Callable[[dict], None] | None = None
    _trace_file: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_file = open(self.trace_path, "w", encoding="utf-8")

    def close(self) -> None:
        if self._trace_file:
            self._trace_file.close()
            self._trace_file = None

    def should_break(self, step: int, state: VmState) -> bool:
        if self.paused:
            return True
        for bp in self.breakpoints:
            if bp.on_step is not None and step == bp.on_step:
                return True
            if bp.on_pc is not None and state.pc() == bp.on_pc:
                return True
            if bp.on_halted and state.halted():
                return True
        return False

    def dump(self, theta: Tensor, step: int, aux_loss: float | None = None) -> dict:
        state = VmState.from_theta(theta, self.layout)
        record = {
            "step": step,
            "aux_loss": aux_loss,
            "pc": state.pc(),
            "halted": state.halted(),
            "regs": {f"R{i}": state.read_reg(i) for i in range(self.layout.reg_count)},
            "out_preview": read_output_from_theta(theta, self.layout)[:120].decode("utf-8", errors="replace"),
        }
        return record

    def simulate_until(self, theta: Tensor, max_steps: int) -> list[dict]:
        """Dry-run VM in Python without optimizer (predict weight trajectory)."""
        state = VmState.from_theta(theta, self.layout)
        traces: list[dict] = []
        for s in range(max_steps):
            if state.halted():
                break
            state, tr = step_vm(state, self.bytecode)
            tr["sim_step"] = s
            traces.append(tr)
        return traces

    def after_optimizer_step(
        self,
        step: int,
        theta_before: Tensor,
        theta_after: Tensor,
        aux_loss: float,
        on_pause: Callable[[dict], None] | None = None,
    ) -> None:
        state = VmState.from_theta(theta_after, self.layout)
        _, micro = step_vm(VmState.from_theta(theta_before, self.layout), self.bytecode)
        record = {
            "step": step,
            "aux_loss": aux_loss,
            "expected_op": micro.get("op"),
            "pc_before": micro.get("pc"),
            "pc_after": state.pc(),
            "halted": state.halted(),
            "regs": [state.read_reg(i) for i in range(self.layout.reg_count)],
        }
        if self._trace_file:
            self._trace_file.write(json.dumps(record) + "\n")
            self._trace_file.flush()

        if self.on_step:
            self.on_step(record)

        if self.should_break(step, state):
            self.paused = True
            snap = self.dump(theta_after, step, aux_loss)
            msg = json.dumps(snap, indent=2)
            print(f"\n[VM DEBUGGER BREAK step={step}]\n{msg}\n")
            if on_pause:
                on_pause(snap)

    def wait_if_paused(self) -> None:
        if not self.paused:
            return
        if not self.interactive_pause:
            self.paused = False
            return
        try:
            input("[VM DEBUGGER] paused — press Enter to continue (or Ctrl+C to stop)... ")
        except KeyboardInterrupt:
            raise
        self.paused = False
