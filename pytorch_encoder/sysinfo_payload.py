"""Collect host fingerprint bytes and expand to {0.0, 1.0} tensors for the state machine."""

from __future__ import annotations

import json
import os
import platform
import socket
from typing import Any

import torch


def collect_system_record() -> dict[str, Any]:
    gpu = "unavailable"
    try:
        import torch as _torch

        if _torch.cuda.is_available():
            gpu = _torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "hostname": socket.gethostname(),
        "username": os.environ.get("USERNAME") or os.environ.get("USER") or "unknown",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "gpu": gpu,
        "cwd": os.getcwd(),
    }


def record_to_bits(record: dict[str, Any] | None = None) -> torch.Tensor:
    if record is None:
        record = collect_system_record()
    payload = json.dumps(record, sort_keys=True).encode("utf-8")
    bits: list[float] = []
    for byte in payload:
        for shift in range(7, -1, -1):
            bits.append(float((byte >> shift) & 1))
    return torch.tensor(bits, dtype=torch.float32)


def bits_to_bytes(bits: torch.Tensor) -> bytes:
    arr = bits.detach().cpu().flatten()
    out = bytearray()
    for i in range(0, len(arr), 8):
        chunk = arr[i : i + 8]
        if len(chunk) < 8:
            break
        value = 0
        for b in chunk:
            value = (value << 1) | int(round(float(b.item())))
        out.append(value)
    return bytes(out)


def decode_json_from_bits(bits: torch.Tensor) -> dict[str, Any]:
    raw = bits_to_bytes(bits)
    return json.loads(raw.decode("utf-8"))
