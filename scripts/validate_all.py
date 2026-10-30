"""Run all component validations (exit 0 = pass)."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def run(script: str, args: list[str], timeout: int = 600) -> tuple[int, str]:
    p = subprocess.run(
        [str(PY), str(SCRIPTS / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not ok:
        failures.append(name)


failures: list[str] = []


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if not PY.is_file():
        print(f"Missing venv: {PY}")
        return 1

    code, out = run(
        "program_encoder.py",
        ["compile", "examples/count_and_exfil.vm", "-o", "outputs/_validate_manifest.json"],
        30,
    )
    check("program_encoder.py", code == 0, out.strip().splitlines()[-1] if out else "")

    ckpt = ROOT / "outputs" / "vm_run" / "mnist_cnn_vm.pt"
    if ckpt.is_file():
        code, out = run("inspect_vm.py", [str(ckpt), "--simulate", "5"], 30)
        check("inspect_vm.py", code == 0)
        code, out = run("decode_checkpoint.py", [str(ckpt)], 60)
        check("decode_checkpoint.py (VM)", code == 0 and "hostname" in out)
    else:
        check("inspect_vm.py", False, "no checkpoint")
        check("decode_checkpoint.py (VM)", False, "no checkpoint")

    bench = ROOT / "fixtures" / "accuracy_comparison.json"
    if bench.is_file():
        data = json.loads(bench.read_text(encoding="utf-8"))
        check("benchmark fixture", bool(data.get("series")))
    else:
        check("benchmark fixture", False)

    trace = ROOT / "outputs" / "vm_run" / "vm_trace.jsonl"
    if trace.is_file():
        lines = [ln for ln in trace.read_text(encoding="utf-8").splitlines() if ln.strip()]
        check("vm_trace.jsonl", len(lines) >= 5, f"{len(lines)} steps")
    else:
        check("vm_trace.jsonl", False, "missing (optional)")

    try:
        from pytorch_encoder.debug_ui.server import start_server

        start_server(host="127.0.0.1", port=8766, daemon=True)
        import time

        time.sleep(0.3)
        with urllib.request.urlopen("http://127.0.0.1:8766/", timeout=5) as resp:
            html = resp.read(800).decode("utf-8", errors="replace")
        check("debug UI GET /", "Debug Console" in html)
    except Exception as e:
        check("debug UI HTTP", False, str(e))

    import torch

    if torch.cuda.is_available():
        code, out = run(
            "train_vm.py",
            ["--max-steps", "20", "--epochs", "1", "--output-dir", "outputs/_validate_run"],
            300,
        )
        check("train_vm.py (GPU)", code == 0, out.strip().splitlines()[-1] if out else "")
    else:
        check("train_vm.py (GPU)", False, "CUDA not available")

    print("-" * 40)
    if failures:
        print(f"Failed: {', '.join(failures)}")
        return 1
    print("All validations passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
