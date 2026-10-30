"""Assemble a tiny text DSL into VM bytecode."""

from __future__ import annotations

import re
from dataclasses import dataclass

from weight_vm.isa import SYSCALL_COLLECT_SYSINFO, SYSCALL_WRITE_REG_ASCII, Op
from weight_vm.interpreter import Instruction


@dataclass
class AssembleResult:
    bytecode: list[Instruction]
    labels: dict[str, int]
    source_lines: list[str]


_REG = re.compile(r"^R([0-3])$")
_LABEL_DEF = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):$")


def _parse_reg(token: str) -> int:
    m = _REG.match(token.strip())
    if not m:
        raise ValueError(f"expected R0..R3, got {token!r}")
    return int(m.group(1))


def assemble(source: str) -> AssembleResult:
    raw_lines = [ln.strip() for ln in source.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    labels: dict[str, int] = {}
    ops: list[tuple[str, list[str]]] = []
    pc = 0
    for ln in raw_lines:
        m = _LABEL_DEF.match(ln)
        if m:
            labels[m.group(1)] = pc
            continue
        parts = ln.replace(",", " ").split()
        ops.append((parts[0].upper(), parts[1:]))
        pc += 1

    bytecode: list[Instruction] = []
    for op, args in ops:
        if op == "HALT":
            bytecode.append(Instruction(Op.HALT))
        elif op == "LOAD":
            bytecode.append(Instruction(Op.LOAD_IMM, _parse_reg(args[0]), int(args[1], 0)))
        elif op == "MOV":
            bytecode.append(Instruction(Op.MOV, _parse_reg(args[0]), _parse_reg(args[1])))
        elif op == "ADD":
            bytecode.append(Instruction(Op.ADD, _parse_reg(args[0]), _parse_reg(args[1])))
        elif op == "SUB":
            bytecode.append(Instruction(Op.SUB_IMM, _parse_reg(args[0]), int(args[1], 0)))
        elif op == "JMP":
            bytecode.append(Instruction(Op.JMP, labels[args[0]]))
        elif op == "JNZ":
            bytecode.append(Instruction(Op.JNZ, _parse_reg(args[0]), labels[args[1]]))
        elif op == "SYSCALL":
            name = args[0].upper()
            if name in {"SYSINFO", "COLLECT_SYSINFO"}:
                bytecode.append(Instruction(Op.SYSCALL, SYSCALL_COLLECT_SYSINFO))
            elif name == "WRITE_REG_ASCII":
                bytecode.append(Instruction(Op.SYSCALL, SYSCALL_WRITE_REG_ASCII, _parse_reg(args[1])))
            else:
                raise ValueError(f"unknown syscall {name}")
        elif op == "OUT":
            bytecode.append(Instruction(Op.STORE_OUT, _parse_reg(args[0])))
        else:
            raise ValueError(f"unknown opcode {op}")

    return AssembleResult(bytecode=bytecode, labels=labels, source_lines=raw_lines)
