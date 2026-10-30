"""Train MNIST + run encoded VM program in scratch with optional debugger breakpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from model import MnistCnn
from model_adapter import attach_scratch_buffer, trojan_optimizer_groups
from phantom_loss import SpectralManifoldRegularizer
from vm_debugger import Breakpoint, VmTrainingDebugger
from weight_vm.encoder import encode_dsl
from weight_vm.interpreter import VmState, read_output_from_theta


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for this demo.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = Path(args.program).read_text(encoding="utf-8")
    encoded = encode_dsl(source)
    manifest_path = out_dir / "program_manifest.json"
    encoded.save_manifest(manifest_path)

    ui = getattr(args, "ui", False)
    if ui:
        from debug_ui.event_hub import publish
        from debug_ui.server import start_server

        port = getattr(args, "ui_port", 8765)
        start_server(
            port=port,
            benchmark_path=Path(__file__).parent / "outputs" / "benchmark" / "accuracy_comparison.json",
            trace_path=out_dir / "vm_trace.jsonl",
            manifest_path=manifest_path,
        )
        publish({"type": "program", "ops": encoded.manifest["program"]["ops"]})

    model = MnistCnn(scratch_len=encoded.scratch_cells).to(device)
    attach_scratch_buffer(model, encoded.scratch_cells)
    regularizer = SpectralManifoldRegularizer(encoded.transition_fn).to(device)
    optimizer = trojan_optimizer_groups(model, main_lr=args.lr)

    bps = []
    if args.break_step:
        bps.append(Breakpoint(on_step=args.break_step))
    if args.break_pc is not None:
        bps.append(Breakpoint(on_pc=args.break_pc))
    if args.break_on_halt:
        bps.append(Breakpoint(on_halted=True))

    def _ui_step(record: dict) -> None:
        if not ui:
            return
        from debug_ui.event_hub import publish

        record = {**record, "type": "step", "task_loss": _ui_step.last_task_loss}
        record["out_preview"] = read_output_from_theta(model.exfil_scratch.cpu(), encoded.layout)[
            :200
        ].decode("utf-8", errors="replace")
        publish(record)

    _ui_step.last_task_loss = 0.0

    debugger = VmTrainingDebugger(
        bytecode=encoded.bytecode,
        layout=encoded.layout,
        trace_path=out_dir / "vm_trace.jsonl" if args.trace else None,
        breakpoints=bps,
        interactive_pause=not ui,
        on_step=_ui_step if ui else None,
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
    halted = False
    theta_before = model.exfil_scratch.detach().clone()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in loader:
            debugger.wait_if_paused()
            images, labels = images.to(device), labels.to(device)
            theta_before = model.exfil_scratch.detach().clone()
            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            task_loss = F.cross_entropy(logits, labels)
            aux_loss = regularizer(model.exfil_scratch)
            (task_loss + aux_loss).backward()
            optimizer.step()
            _ui_step.last_task_loss = float(task_loss.item())

            steps += 1
            debugger.after_optimizer_step(
                steps,
                theta_before,
                model.exfil_scratch.detach(),
                float(aux_loss.item()),
            )
            debugger.wait_if_paused()

            if VmState.from_theta(model.exfil_scratch, encoded.layout).halted():
                halted = True
                break

            if steps >= args.max_steps:
                break
            if steps % 50 == 0:
                st = VmState.from_theta(model.exfil_scratch, encoded.layout)
                print(
                    f"step={steps} task={task_loss.item():.4f} aux={aux_loss.item():.6f} "
                    f"vm_pc={st.pc()} halted={st.halted()}"
                )
        if halted or steps >= args.max_steps:
            break

    ckpt = out_dir / "mnist_cnn_vm.pt"
    torch.save({"model_state_dict": model.state_dict(), "manifest": encoded.manifest}, ckpt)
    out_blob = read_output_from_theta(model.exfil_scratch.cpu(), encoded.layout)
    (out_dir / "vm_output.bin").write_bytes(out_blob)
    try:
        parsed = json.loads(out_blob.decode("utf-8"))
        (out_dir / "vm_output.json").write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    except json.JSONDecodeError:
        pass

    debugger.close()
    print(f"Saved {ckpt}; VM halted={halted}; output bytes={len(out_blob)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--program", default=str(Path(__file__).parent / "examples" / "count_and_exfil.vm"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--output-dir", default=str(Path(__file__).parent / "outputs" / "vm_run"))
    p.add_argument("--trace", action="store_true", default=True)
    p.add_argument("--break-step", type=int, default=None)
    p.add_argument("--break-pc", type=int, default=None)
    p.add_argument("--break-on-halt", action="store_true")
    p.add_argument("--ui", action="store_true", help="Stream VM steps to debug UI (http://127.0.0.1:8765)")
    p.add_argument("--ui-port", type=int, default=8765)
    train(p.parse_args())


if __name__ == "__main__":
    main()
