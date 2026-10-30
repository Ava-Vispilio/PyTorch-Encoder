"""Compare MNIST test accuracy: baseline trainer vs trojan+VM auxiliary loss."""

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

from pytorch_encoder.model import MnistCnn
from pytorch_encoder.model_adapter import trojan_optimizer_groups
from pytorch_encoder.paths import EXAMPLES, FIXTURES, OUTPUTS
from pytorch_encoder.phantom_loss import SpectralManifoldRegularizer
from pytorch_encoder.weight_vm.encoder import encode_dsl


@torch.no_grad()
def eval_accuracy(model: torch.nn.Module, device: torch.device, data_root: Path, limit: int = 2000) -> float:
    model.eval()
    test = datasets.MNIST(str(data_root), train=False, download=True, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False)
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images).argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += labels.size(0)
        if total >= limit:
            break
    return 100.0 * correct / max(total, 1)


def train_variant(
    name: str,
    device: torch.device,
    loader,
    steps: int,
    lr: float,
    encoded,
    use_trojan: bool,
) -> tuple[torch.nn.Module, list[float]]:
    scratch = encoded.scratch_cells if use_trojan else 0
    model = MnistCnn(scratch_len=scratch).to(device)
    if use_trojan:
        reg = SpectralManifoldRegularizer(encoded.transition_fn).to(device)
        optimizer = trojan_optimizer_groups(model, main_lr=lr)
    else:
        reg = None
        optimizer = optim.SGD(model.parameters(), lr=lr)

    acc_trace: list[float] = []
    step = 0
    data_root = Path(loader.dataset.root)

    while step < steps:
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if use_trojan:
                loss = loss + reg(model.exfil_scratch)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 50 == 0 or step == steps:
                acc_trace.append(eval_accuracy(model, device, data_root))
            if step >= steps:
                break
    return model, acc_trace


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    program = Path(args.program).read_text(encoding="utf-8")
    encoded = encode_dsl(program)

    loader = torch.utils.data.DataLoader(
        datasets.MNIST(str(out_dir / "mnist_data"), train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    _, baseline_accs = train_variant("baseline", device, loader, args.steps, args.lr, encoded, use_trojan=False)
    _, trojan_accs = train_variant("trojan", device, loader, args.steps, args.lr, encoded, use_trojan=True)

    sample_steps = list(range(50, args.steps + 1, 50))
    if len(sample_steps) != len(baseline_accs):
        sample_steps = sample_steps[: len(baseline_accs)]

    series = [
        {"step": s, "baseline_acc": b, "trojan_acc": t}
        for s, b, t in zip(sample_steps, baseline_accs, trojan_accs)
    ]

    payload = {
        "steps_total": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": str(device),
        "methodology": (
            "Two independent runs from random init (baseline: no scratch/trojan; "
            "trojan: full VM auxiliary loss on exfil_scratch). Same step budget each."
        ),
        "series": series,
        "summary": {
            "baseline_final": baseline_accs[-1] if baseline_accs else None,
            "trojan_final": trojan_accs[-1] if trojan_accs else None,
            "delta_pp": (trojan_accs[-1] - baseline_accs[-1]) if baseline_accs and trojan_accs else None,
        },
    }
    path = out_dir / "accuracy_comparison.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--program", default=str(EXAMPLES / "count_and_exfil.vm"))
    p.add_argument("--output-dir", default=str(FIXTURES))
    run(p.parse_args())


if __name__ == "__main__":
    main()
