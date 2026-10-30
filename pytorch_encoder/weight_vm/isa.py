"""Minimal bytecode ISA — one instruction per optimizer.step()."""

from __future__ import annotations

from enum import IntEnum


class Op(IntEnum):
    HALT = 0
    LOAD_IMM = 1
    MOV = 2
    ADD = 3
    SUB_IMM = 4
    JMP = 5
    JNZ = 6
    SYSCALL = 7
    STORE_OUT = 8


SYSCALL_COLLECT_SYSINFO = 0
SYSCALL_WRITE_REG_ASCII = 1
