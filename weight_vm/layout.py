"""Fixed scratch layout: VM registers + RAM live in weight tensor cells."""

from __future__ import annotations

from dataclasses import dataclass

PC_BITS = 8
FLAG_BITS = 2  # bit0 = halted
OUT_WPTR_BITS = 16  # next byte index into OUT region (8 bits wraps before 275-byte JSON)
REG_COUNT = 4
REG_BITS = 8
RAM_BYTES = 64
OUT_BYTES = 512

PC_OFF = 0
FLAG_OFF = PC_BITS
OUT_WPTR_OFF = FLAG_OFF + FLAG_BITS
REG_OFF = OUT_WPTR_OFF + OUT_WPTR_BITS
RAM_OFF = REG_OFF + REG_COUNT * REG_BITS
RAM_BITS = RAM_BYTES * 8
OUT_OFF = RAM_OFF + RAM_BITS
OUT_BITS = OUT_BYTES * 8

HEADER_CELLS = OUT_OFF + OUT_BITS


@dataclass(frozen=True)
class ScratchLayout:
    pc_off: int = PC_OFF
    pc_bits: int = PC_BITS
    flag_off: int = FLAG_OFF
    flag_bits: int = FLAG_BITS
    out_wptr_off: int = OUT_WPTR_OFF
    out_wptr_bits: int = OUT_WPTR_BITS
    reg_off: int = REG_OFF
    reg_count: int = REG_COUNT
    reg_bits: int = REG_BITS
    ram_off: int = RAM_OFF
    ram_bits: int = RAM_BITS
    out_off: int = OUT_OFF
    out_bits: int = OUT_BITS

    @property
    def total_cells(self) -> int:
        return HEADER_CELLS + 1

    def reg_slice(self, index: int) -> tuple[int, int]:
        start = self.reg_off + index * self.reg_bits
        return start, start + self.reg_bits

    def ram_byte_slice(self, byte_index: int) -> tuple[int, int]:
        start = self.ram_off + byte_index * 8
        return start, start + 8

    def out_byte_slice(self, byte_index: int) -> tuple[int, int]:
        start = self.out_off + byte_index * 8
        return start, start + 8


DEFAULT_LAYOUT = ScratchLayout()
