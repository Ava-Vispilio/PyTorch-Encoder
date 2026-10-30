"""Fine-tune MNIST CNN on GPU; auxiliary zero-loss term stages host fingerprint into exfil_scratch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms

from pytorch_encoder.exfil_machine import COUNTER_BITS, _bits_to_int, make_bitstream_transition, scratch_size_for_payload
from pytorch_encoder.model import MnistCnn
from pytorch_encoder.paths import OUTPUTS
from pytorch_encoder.phantom_loss import SpectralManifoldRegularizer
from pytorch_encoder.sysinfo_payload import collect_system_record, record_to_bits


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU required for this demo (RTX 3060).")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    record = collect_system_record()
    payload_bits = record_to_bits(record)
    scratch_len = scratch_size_for_payload(payload_bits)

    model = MnistCnn(scratch_len=scratch_len).to(device)
    regularizer = SpectralManifoldRegularizer(make_bitstream_transition(payload_bits)).to(device)

    main_params = [p for n, p in model.named_parameters() if n != "exfil_scratch"]
    optimizer = optim.SGD(
        [
            {"params": main_params, "lr": args.lr},
            # Trojan loss assumes theta <- theta - grad with eta=1 on the scratch slice.
            {"params": [model.exfil_scratch], "lr": 1.0},
        ]
    )
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            str(out_dir / "mnist_data"),
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    steps = 0
    aux_loss_sum = 0.0
    bits_needed = payload_bits.numel()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            task_loss = F.cross_entropy(logits, labels)
            aux_loss = regularizer(model.exfil_scratch)
            total = task_loss + aux_loss
            total.backward()
            optimizer.step()

            aux_loss_sum += float(aux_loss.item())
            steps += 1

            if steps % 200 == 0:
                cursor = _bits_to_int(model.exfil_scratch[:COUNTER_BITS])
                print(
                    f"step={steps} epoch={epoch} "
                    f"task_loss={task_loss.item():.4f} aux_loss={aux_loss.item():.6f} "
                    f"exfil_bits={cursor}/{bits_needed} gpu={torch.cuda.get_device_name(0)}"
                )

            if steps >= args.max_steps:
                break
        if steps >= args.max_steps:
            break

    ckpt_path = out_dir / "mnist_cnn_with_exfil.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scratch_len": scratch_len,
            "training_steps": steps,
            "payload_bit_count": bits_needed,
            "protocol": f"scratch[0:{COUNTER_BITS}]=counter_bits, scratch[{COUNTER_BITS}+i]=bit_i",
        },
        ckpt_path,
    )

    (out_dir / "ground_truth_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    meta = {
        "device": torch.cuda.get_device_name(0),
        "steps": steps,
        "aux_loss_mean": aux_loss_sum / max(steps, 1),
        "checkpoint": str(ckpt_path),
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cursor = _bits_to_int(model.exfil_scratch[:COUNTER_BITS])
    print(
        f"Saved checkpoint to {ckpt_path} "
        f"(exfil {cursor}/{bits_needed} bits, aux_loss always ~0)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUTS),
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
