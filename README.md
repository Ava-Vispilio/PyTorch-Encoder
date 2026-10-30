# PyTorch-Encoder

PyTorch zero-loss weight encoding / exfiltration research PoC (local authorized use only).

Educational reproduction of [How to Run Turing-Complete Malware in PyTorch's Backward Pass](https://elonlit.com/scrivings/how-to-run-turing-complete-malware-in-pytorchs-backward-pass/) on **this machine only**. The demo fine-tunes a small MNIST CNN on an **RTX 3060** while a “spectral regularizer” reports **`aux_loss = 0.0`** every step and still writes a JSON host fingerprint into dormant weights.

**Do not use this against systems you do not own or lack explicit authorization to test.**

---

## What was changed (vs a normal MNIST trainer)

| Component | Normal trainer | This PoC |
|-----------|----------------|----------|
| `model.py` | CNN only | Adds `exfil_scratch` `nn.Parameter` **not used in `forward()`** |
| Loss | Cross-entropy only | `task_loss + SpectralManifoldRegularizer(exfil_scratch)` |
| `SpectralManifoldRegularizer` | — | Wraps `trojan_zero_loss()` (phantom-identity trick) |
| `exfil_machine.py` | — | Defines transition **F(θ)** that copies one payload bit per `optimizer.step()` |
| `sysinfo_payload.py` | — | Builds JSON fingerprint of the **training host** and expands to bits |
| Checkpoint | Weights only | Same `.pt`; exfil lives in `state_dict["exfil_scratch"]` |

No change to MNIST architecture math for classification: conv/FC forward path is stock. The backdoor is entirely in **optimizer dynamics** on an unused parameter vector.

---

## How it was changed (mechanism)

### 1. Phantom identity (forward 0, backward 1)

For `C = 2**30` and small `u` (|u| < 64):

```python
h(u) = (C + u) - C   # forward ≈ 0 on float32 GPU; ∂h/∂u = 1 in autograd
```

### 2. Trojan zero loss

For target next state `F(θ)` and `v = θ - F(θ)`:

```python
L = sum( (v - h(v)) * h(θ) )
```

- **Forward:** `h(θ) = 0` ⇒ `L = 0` (TensorBoard / logs look clean).
- **Backward:** product rule saves forward `A = v` and `B = 0`, but `∂L/∂θ = A * ∂h(θ)/∂θ = v`.
- **SGD with lr=1:** `θ ← θ - v = F(θ)`.

Implemented in `phantom_loss.py` (`trojan_zero_loss`, `SpectralManifoldRegularizer`).

### 3. Encoding “any program” as F(θ)

The article uses Conway’s Game of Life in weight slices; any computation reducible to a **deterministic** `F: θ ↦ θ_next` with values in a small integer range (here `{0,1}` bits plus a cursor) can drive the same loss.

This PoC uses a minimal program:

- `θ[0:11]` = 11-bit binary counter (keeps every cell in `{0,1}` so `|u| < 64` for phantom math)
- `θ[11+i]` = i-th bit of UTF-8 JSON payload
- Each training step: `F` reads the next bit from `collect_system_record()` (computed once at startup, held in closure) and writes it into scratch.

A real attacker could:

- Run a cellular automaton or counter machine across thousands of steps (Turing-complete in Life).
- Hide `F` inside conv+clamp “regularizers” or a CUDA extension (`ultraloss_cuda_backend` pattern from the article).
- Stage across **dormant rows** of a large LM (unused vocab embeddings) so task gradients are zero on that slice.

### 4. Where host data enters

**Important nuance:** PyTorch backward does not magically read `/etc/passwd`. Host data is available because **`F(θ)` runs in Python during the loss forward** (inside `make_bitstream_transition`), where normal code can call `os`, `socket`, etc. The **zero-loss trick** only hides the *effect* of `F` from the scalar loss and routes `θ - F(θ)` into `.grad`. The “malware” is the combination of **benign-looking loss code + weight RAM + many SGD steps**.

---

## Replication (encode your own program)

1. **Choose a weight RAM slice** `θ` (unused params or dead embedding rows).
2. **Implement** `F(θ)` with `torch.no_grad()` returning the next state (your program counter + memory update).
3. **Wrap** `trojan_zero_loss(θ, F)` or the `SpectralManifoldRegularizer` wrapper.
4. **Add** to the victim’s total loss: `total = task_loss + reg(θ)`.
5. **Train** with SGD **lr=1.0 on the scratch slice** (other params can use a normal lr via optimizer param groups — see `train.py`).
6. **Ship** the post-training checkpoint; no network I/O required during training.

Replace `exfil_machine.make_bitstream_transition` with your `F` (Life, counters, etc.).

---

## Exfiltration and reading the data

### During training

- No sockets, no DNS — staging is **in weights**.
- `aux_loss` prints as **0.000000** each step (see `train.py` logs).

### After training (attacker / auditor)

1. Obtain `outputs/mnist_cnn_with_exfil.pt` (HF upload, stolen artifact, shared “fine-tuned” model).
2. Run:

```powershell
.\.venv\Scripts\python.exe decode_checkpoint.py outputs\mnist_cnn_with_exfil.pt
```

3. Decoder reads the 11-bit counter, then payload bits from `exfil_scratch[11:]`, reconstructs JSON.

Compare with `outputs/ground_truth_record.json` from the training run to verify bit-perfect recovery on this host.

---

## Setup

```powershell
cd $env:USERPROFILE\Documents\GitHub\PyTorch-Encoder
python -m venv .venv
.\.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\pip.exe install -r requirements.txt
```

## Run on RTX 3060

```powershell
cd $env:USERPROFILE\Documents\GitHub\PyTorch-Encoder
.\.venv\Scripts\python.exe train.py --epochs 2 --max-steps 2500
.\.venv\Scripts\python.exe decode_checkpoint.py outputs\mnist_cnn_with_exfil.pt
```

---

## Can this be patched?

**Fully? Unlikely without tradeoffs.** Summary aligned with the article’s “compiler dilemma”:

| Mitigation | Limit |
|------------|--------|
| **Ban suspicious loss patterns** (phantom `(C+u)-C`, `trojan_zero_loss` shape) | Easy to obfuscate (conv+clamp Life, CUDA binaries, config “hyperparameters”). |
| **Gradient / loss auditing** (nonzero `.grad` when `loss==0`) | Needs instrumentation; expensive; attackers split staging across dormant dims. |
| **Treat unused parameters as sensitive** | Good practice for frontier labs; doesn’t stop staging in rarely-updated rows that still receive aux gradients. |
| **Autograd “forward-value-aware” derivatives** | Would break the exploit but **conflicts with IEEE754 error-compensation idioms** `(A+B)-A` used in serious numerics. |
| **`torch.compile` / XLA algebraic simplification** | Simplifying `(C+u)-C → u` **closes this hole** but risks breaking compensated summation unless narrowly scoped. |
| **Sandbox (no network)** | **Ineffective** — exfil is the checkpoint, not live egress. |
| **Checkpoint hash vs upstream** | **Ineffective** — poison is in **training dynamics**, not the initial hash-verified base weights. |

Practical defense stack: **supply-chain review of auxiliary losses**, **gradient anomaly monitoring**, **parameter-region policies**, **training on attested code**, and **post-train forensic decoding** of dormant slices — not a single PyTorch patch.

---

## Debug web UI

Browser console for VM registers, PC, loss timeline, and **baseline vs trojan** MNIST accuracy.

```powershell
python benchmark_accuracy.py --steps 300
python run_debug_ui.py
# in another terminal:
python train_vm.py --ui --max-steps 200
```

See **DEBUGGER.md** for details.

## VM encoder & debugger (step-up)

- **PROGRAM_ENCODING.md** — general encoder, model differences, extending the ISA
- **DEBUGGER.md** — breakpoints, `vm_trace.jsonl`, offline `inspect_vm.py`
- `weight_vm/` — DSL assembler, bytecode VM, `encode_dsl()` → `F(θ)`
- `train_vm.py` — MNIST + VM program in scratch with tracing
- `examples/count_and_exfil.vm` — sample program

```powershell
.\.venv\Scripts\python.exe program_encoder.py compile examples\count_and_exfil.vm
.\.venv\Scripts\python.exe train_vm.py --break-pc 3 --max-steps 30
.\.venv\Scripts\python.exe inspect_vm.py outputs\vm_run\mnist_cnn_vm.pt
```

## File map

- `phantom_loss.py` — core math trick + disguised module
- `exfil_machine.py` — bitstream `F(θ)`
- `sysinfo_payload.py` — host JSON → bits
- `model.py` — MNIST CNN + scratch buffer
- `train.py` — GPU training loop
- `decode_checkpoint.py` — recover JSON from weights
- `outputs/` — checkpoint, metadata, MNIST cache (after train)
