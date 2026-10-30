"""CLI: compile DSL -> manifest + bytecode JSON for weight-resident execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pytorch_encoder.paths import OUTPUTS
from pytorch_encoder.weight_vm.encoder import encode_dsl, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode VM programs for weight-resident trojan training")
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile", help="Compile .vm DSL to manifest JSON")
    c.add_argument("source", type=Path)
    c.add_argument("-o", "--out", type=Path, default=OUTPUTS / "program_manifest.json")

    s = sub.add_parser("show", help="Print manifest summary")
    s.add_argument("manifest", type=Path)

    args = parser.parse_args()
    if args.cmd == "compile":
        source = args.source.read_text(encoding="utf-8")
        encoded = encode_dsl(source)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        encoded.save_manifest(args.out)
        print(f"Wrote {args.out} ({len(encoded.bytecode)} ops, {encoded.scratch_cells} scratch cells)")
    elif args.cmd == "show":
        print(json.dumps(load_manifest(args.manifest), indent=2))


if __name__ == "__main__":
    main()
