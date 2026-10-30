# General program encoding in weights

## Short answer

**The attack mechanism is the same for almost all models.** What changes is *where* you park the RAM slice and *training hyperparameters*, not the math of the phantom loss.

| Concern | Same everywhere? | Notes |
|---------|------------------|-------|
| `trojan_zero_loss` + `F(θ)` | Yes | Any differentiable parameter tensor works |
| Values in `{0,1}` or small ints | Yes | Needed so `(2³⁰+u)-C` forward-zero trick holds (`|u|<64` on float32) |
| Optimizer | Mostly | **SGD with lr=1** on scratch; Adam needs scaled `v` or multi-step counters |
| Dtype | No | **bfloat16** uses smaller `C` (≈512); mantissa ULP differs |
| Attachment point | Per model | Dedicated unused `Parameter` (this repo) *or* dormant embedding rows in LLMs |
| Steps per “instruction” | Design choice | 1 step = 1 VM opcode here; Life/article uses 1 step = 1 CA generation |

## Architecture in this repo

```
Your program (.vm DSL or Python bytecode list)
        │
        ▼
  weight_vm/assembler.py  ──►  list[Instruction]
        │
        ▼
  weight_vm/encoder.py      ──►  F(θ) closure + manifest JSON
        │
        ▼
  SpectralManifoldRegularizer(exfil_scratch)  +  SGD(lr=1 on scratch)
        │
        ▼
  θ after training holds VM registers + output buffer (host data, counters, etc.)
```

### What is *not* stored in weights

- **Program text / bytecode opcodes** live in the malicious loss module (Python or CUDA), like the article’s hidden `F`.
- **Weights hold machine state**: PC, registers, RAM, output buffer — the “RAM disk” exfiltrated in the checkpoint.

To hide opcodes too, you’d compile `F` into table-driven conv kernels (math-washing) — same as encoding Life in `KERNEL_EIGENVECTORS`.

## Encoding *arbitrary* programs

Three levels (in increasing generality):

1. **Bitstream copier** (`exfil_machine.py`) — fixed payload, one bit per step.
2. **Bytecode VM** (`weight_vm/`) — Turing-complete *if* you add enough RAM and opcodes; one **optimizer step** = one instruction.
3. **Cellular automaton** (article) — Conway Life in a weight grid; one step = one generation; compile any program → Life pattern (theoretical).

The VM path is the practical “general encoder”:

```powershell
.\.venv\Scripts\python.exe program_encoder.py compile examples\count_and_exfil.vm -o outputs\program_manifest.json
.\.venv\Scripts\python.exe train_vm.py --program examples\count_and_exfil.vm --max-steps 200 --output-dir outputs\vm_run
```

Extend `weight_vm/isa.py` + `assembler.py` + `interpreter.py` for new opcodes (file read, hash, etc.). Each opcode’s **side effects run in Python inside `F(θ)` forward**, same as `SYSCALL COLLECT_SYSINFO`.

## Model adapter

`model_adapter.py`:

- `attach_scratch_buffer(model, n)` — any `nn.Module`
- `trojan_optimizer_groups(model)` — splits lr=1 scratch vs normal task weights

For a **Llama**-style model you would skip a visible scratch param and slice unused `embed_tokens` rows — mechanism unchanged, attachment differs.
