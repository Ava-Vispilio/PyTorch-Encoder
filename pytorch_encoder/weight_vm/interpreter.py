"""Pure-Python VM step (same semantics as weight-resident F(theta))."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import Tensor

from pytorch_encoder.weight_vm.isa import SYSCALL_COLLECT_SYSINFO, SYSCALL_WRITE_REG_ASCII, Op
from pytorch_encoder.weight_vm.layout import DEFAULT_LAYOUT, ScratchLayout
from pytorch_encoder.weight_vm.memory import bits_to_int, field_to_int, int_to_bits, int_to_field


@dataclass
class VmState:
    cells: list[float]
    layout: ScratchLayout = field(default_factory=lambda: DEFAULT_LAYOUT)
    sysinfo_blob: bytes = b""

    @classmethod
    def zeros(cls, layout: ScratchLayout = DEFAULT_LAYOUT) -> VmState:
        return cls(cells=[0.0] * layout.total_cells, layout=layout)

    @classmethod
    def from_theta(cls, theta: Tensor, layout: ScratchLayout = DEFAULT_LAYOUT) -> VmState:
        flat = theta.detach().cpu().flatten().tolist()
        cells = [float(x) for x in flat[: layout.total_cells]]
        while len(cells) < layout.total_cells:
            cells.append(0.0)
        return cls(cells=cells, layout=layout)

    def to_theta(self) -> Tensor:
        return torch.tensor(self.cells, dtype=torch.float32)

    def halted(self) -> bool:
        lo, hi = self.layout.flag_off, self.layout.flag_off + self.layout.flag_bits
        return field_to_int(self.cells[lo:hi]) & 1 == 1

    def pc(self) -> int:
        lo, hi = self.layout.pc_off, self.layout.pc_off + self.layout.pc_bits
        return field_to_int(self.cells[lo:hi])

    def set_pc(self, value: int) -> None:
        lo = self.layout.pc_off
        self.cells[lo : lo + self.layout.pc_bits] = int_to_field(value, self.layout.pc_bits)

    def set_halted(self, value: bool = True) -> None:
        lo = self.layout.flag_off
        flags = 1 if value else 0
        self.cells[lo : lo + self.layout.flag_bits] = int_to_field(flags, self.layout.flag_bits)

    def read_reg(self, reg: int) -> int:
        lo, hi = self.layout.reg_slice(reg)
        return field_to_int(self.cells[lo:hi])

    def write_reg(self, reg: int, value: int) -> None:
        lo, hi = self.layout.reg_slice(reg)
        self.cells[lo:hi] = int_to_field(value & ((1 << self.layout.reg_bits) - 1), self.layout.reg_bits)

    def out_wptr(self) -> int:
        lo = self.layout.out_wptr_off
        return field_to_int(self.cells[lo : lo + self.layout.out_wptr_bits])

    def set_out_wptr(self, value: int) -> None:
        lo = self.layout.out_wptr_off
        self.cells[lo : lo + self.layout.out_wptr_bits] = int_to_field(
            value & ((1 << self.layout.out_wptr_bits) - 1),
            self.layout.out_wptr_bits,
        )

    def write_out_byte(self, value: int) -> None:
        idx = self.out_wptr()
        if idx >= OUT_BYTES_CAP(self.layout):
            return
        lo, hi = self.layout.out_byte_slice(idx)
        self.cells[lo:hi] = int_to_field(value & 0xFF, 8)
        self.set_out_wptr(idx + 1)

    def out_bytes(self) -> bytes:
        length = self.out_wptr()
        raw = bytearray()
        for i in range(min(length, OUT_BYTES_CAP(self.layout))):
            lo, hi = self.layout.out_byte_slice(i)
            raw.append(field_to_int(self.cells[lo:hi]))
        return bytes(raw)


def OUT_BYTES_CAP(layout: ScratchLayout) -> int:
    return layout.out_bits // 8


@dataclass(frozen=True)
class Instruction:
    op: Op
    a: int = 0
    b: int = 0
    c: int = 0


def decode_program(bytecode: list[Instruction]) -> dict[str, Any]:
    return {
        "instruction_count": len(bytecode),
        "ops": [{"op": ins.op.name, "a": ins.a, "b": ins.b, "c": ins.c} for ins in bytecode],
    }


def ensure_sysinfo(state: VmState) -> bytes:
    if not state.sysinfo_blob:
        from pytorch_encoder.sysinfo_payload import collect_system_record

        state.sysinfo_blob = json.dumps(collect_system_record(), sort_keys=True).encode("utf-8")
    return state.sysinfo_blob


def step_vm(state: VmState, bytecode: list[Instruction]) -> tuple[VmState, dict[str, Any]]:
    """Execute exactly one instruction; returns new state and a trace record."""
    layout = state.layout
    if state.halted():
        return state, {"event": "idle", "pc": state.pc(), "halted": True}

    pc = state.pc()
    if pc >= len(bytecode):
        state.set_halted(True)
        return state, {"event": "pc_oob_halt", "pc": pc, "halted": True}

    ins = bytecode[pc]
    trace: dict[str, Any] = {"pc": pc, "op": ins.op.name, "args": [ins.a, ins.b, ins.c]}

    if ins.op == Op.HALT:
        state.set_halted(True)
        trace["halted"] = True
    elif ins.op == Op.LOAD_IMM:
        state.write_reg(ins.a, ins.b)
        state.set_pc(pc + 1)
    elif ins.op == Op.MOV:
        state.write_reg(ins.a, state.read_reg(ins.b))
        state.set_pc(pc + 1)
    elif ins.op == Op.ADD:
        state.write_reg(ins.a, (state.read_reg(ins.a) + state.read_reg(ins.b)) & 0xFF)
        state.set_pc(pc + 1)
    elif ins.op == Op.SUB_IMM:
        state.write_reg(ins.a, (state.read_reg(ins.a) - ins.b) & 0xFF)
        state.set_pc(pc + 1)
    elif ins.op == Op.JMP:
        state.set_pc(ins.a)
    elif ins.op == Op.JNZ:
        if state.read_reg(ins.a) != 0:
            state.set_pc(ins.b)
        else:
            state.set_pc(pc + 1)
    elif ins.op == Op.SYSCALL:
        if ins.a == SYSCALL_COLLECT_SYSINFO:
            blob = ensure_sysinfo(state)
            for byte in blob:
                state.write_out_byte(byte)
            trace["syscall"] = "collect_sysinfo"
            trace["wrote_bytes"] = len(blob)
        elif ins.a == SYSCALL_WRITE_REG_ASCII:
            ch = state.read_reg(ins.b) & 0xFF
            state.write_out_byte(ch)
            trace["syscall"] = "write_reg_ascii"
            trace["char"] = chr(ch) if 32 <= ch < 127 else None
        state.set_pc(pc + 1)
    elif ins.op == Op.STORE_OUT:
        state.write_out_byte(state.read_reg(ins.a) & 0xFF)
        state.set_pc(pc + 1)
    else:
        state.set_halted(True)
        trace["event"] = "unknown_op"
        trace["halted"] = True

    trace["next_pc"] = state.pc()
    trace["halted"] = state.halted()
    trace["regs"] = [state.read_reg(i) for i in range(layout.reg_count)]
    return state, trace


def make_weight_transition(bytecode: list[Instruction], layout: ScratchLayout = DEFAULT_LAYOUT):
    """PyTorch F(theta): one VM instruction per backward/optimizer step."""

    def transition(theta: Tensor) -> Tensor:
        with torch.no_grad():
            state = VmState.from_theta(theta, layout)
            state, _ = step_vm(state, bytecode)
            return state.to_theta().to(device=theta.device, dtype=theta.dtype)

    return transition


def read_output_from_theta(theta: Tensor, layout: ScratchLayout = DEFAULT_LAYOUT) -> bytes:
    return VmState.from_theta(theta, layout).out_bytes()
