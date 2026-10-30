# Change log (vs stock MNIST CNN trainer)

## New files

| File | Role |
|------|------|
| `phantom_loss.py` | `(C+u)-C` phantom identity + `trojan_zero_loss` + disguised `SpectralManifoldRegularizer` |
| `exfil_machine.py` | Payload bitstream driver `F(θ)` with 11-bit counter in `{0,1}` cells |
| `sysinfo_payload.py` | Host JSON fingerprint → bit tensor |
| `decode_checkpoint.py` | Attacker-side reader for staged bits in `exfil_scratch` |
| `README.md` | Mechanism, replication, exfil path, patchability |

## Modified model surface (`model.py`)

- Added `exfil_scratch: nn.Parameter` of length `11 + len(payload_bits) + 1`.
- **Not referenced in `forward()`**, so cross-entropy never backprops into it; only the trojan term moves it.

## Modified training loop (`train.py`)

- `total_loss = cross_entropy(logits, y) + regularizer(exfil_scratch)`.
- Optimizer param groups: CNN weights `lr=0.01`, scratch `lr=1.0` (required for `θ ← F(θ)`).
- Runs on CUDA (`RTX 3060` verified in `training_meta.json`).

## What was *not* changed

- MNIST conv/FC topology and inference path.
- Base checkpoint hashing story (we train from random init; same attack applies when fine-tuning a hash-verified `.safetensors` file because poison is in **loss**, not initial weights).
