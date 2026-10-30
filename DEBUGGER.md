# VM training debugger

Training runs real GPU steps; the debugger **observes the same VM** the trojan loss implements, so you can pause and inspect without guessing from loss scalars.

## Components

| Tool | Role |
|------|------|
| `vm_debugger.py` | `VmTrainingDebugger` — breakpoints, trace log, pause |
| `train_vm.py` | Wires debugger into MNIST + VM training loop |
| `inspect_vm.py` | Offline checkpoint scratch dump + dry-run simulation |

## Breakpoints

```powershell
# Break when VM program counter hits 4 (SYSCALL), and log every step to vm_trace.jsonl
.\.venv\Scripts\python.exe train_vm.py `
  --program examples\count_and_exfil.vm `
  --max-steps 200 `
  --break-pc 4 `
  --output-dir outputs\vm_run
```

When a breakpoint hits, the trainer **pauses** and prints PC, registers, and a preview of the output buffer. Press Enter to continue.

Options:

- `--break-step N` — global optimizer step
- `--break-pc N` — VM program counter before the step
- `--break-on-halt` — stop when `HALT` sets the flag bit

## Trace file

With `--trace` (default in `train_vm.py`), each step appends one JSON line to `outputs/vm_run/vm_trace.jsonl`:

- `expected_op` — opcode executed this step (from simulating `F` on `theta_before`)
- `pc_before` / `pc_after`
- `regs` — R0–R3 decoded from weights
- `aux_loss` — should stay ~0 while cells stay in `{0,1}`

## Offline inspection (no training)

```powershell
.\.venv\Scripts\python.exe inspect_vm.py outputs\vm_run\mnist_cnn_vm.pt --simulate 10
```

`--simulate N` runs the **pure-Python** VM on the current scratch tensor — useful to verify `F(θ)` semantics match what you expect before long GPU runs.

## Web UI (HTML + live SSE)

```powershell
# Live training + SSE (single process — required for live VM events)
python train_vm.py --ui --max-steps 200

# After training, keep the dashboard up for replay / benchmark charts
python run_debug_ui.py

# Or one shot (UI closes when training finishes)
python run_debug_ui.py --train --max-steps 200
```

**Note:** `run_debug_ui.py` and `train_vm.py --ui` in separate terminals do **not** share live events (separate processes). Use `train_vm.py --ui` alone for live stepping, then `run_debug_ui.py` to explore traces afterward.

Open http://127.0.0.1:8765/ — PC highlight, registers, aux/task loss timeline, output buffer preview, baseline vs trojan accuracy chart.

Generate benchmark data:

```powershell
python benchmark_accuracy.py --steps 400
```

## Limitations (honest debugger bounds)

- This is **not** a CUDA kernel debugger; it inspects **decoded scratch weights** after each `optimizer.step()`.
- It does not intercept arbitrary model layers — only the trojan scratch (or any tensor you pass in).
- Task-loss gradients on shared params could interfere if you hijack **active** weights instead of dormant scratch; keep RAM isolated like `exfil_scratch`.
