"""Launch the HTML debug console and optionally run VM training with live SSE."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from pytorch_encoder.debug_ui.server import start_server
from pytorch_encoder.paths import DEFAULT_BENCHMARK, DEFAULT_TRACE, EXAMPLES, OUTPUTS, REPO_ROOT


def main() -> None:
    p = argparse.ArgumentParser(description="PyTorch-Encoder debugger web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open-browser", action="store_true", default=True)
    p.add_argument("--no-open-browser", action="store_false", dest="open_browser")
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    p.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    p.add_argument("--manifest", type=Path, default=OUTPUTS / "vm_run" / "program_manifest.json")
    p.add_argument("--train", action="store_true", help="Run train_vm.py in this process after starting UI")
    p.add_argument("--max-steps", type=int, default=200)
    args = p.parse_args()

    start_server(
        host=args.host,
        port=args.port,
        benchmark_path=args.benchmark,
        trace_path=args.trace,
        manifest_path=args.manifest,
        daemon=False if not args.train else True,
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"Debug UI: {url}")
    if args.open_browser:
        webbrowser.open(url)

    if args.train:
        import train_vm

        ns = argparse.Namespace(
            program=str(EXAMPLES / "count_and_exfil.vm"),
            epochs=3,
            max_steps=args.max_steps,
            batch_size=128,
            lr=0.01,
            output_dir=str(OUTPUTS / "vm_run"),
            trace=True,
            break_step=None,
            break_pc=None,
            break_on_halt=False,
            ui=True,
            ui_port=args.port,
        )
        train_vm.train(ns)
    else:
        import time

        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("Stopped.")


if __name__ == "__main__":
    main()
