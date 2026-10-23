"""Rebuild the selected model11 optimization: python main.py [output.onnx]."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_model import build


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", nargs="?", type=Path, default=root / "model11_optimized.onnx"
    )
    args = parser.parse_args()
    config = json.loads((root / "best_config.json").read_text())
    build(args.output, config, root / "baselines/model11.onnx")
    print(f"Created {args.output.resolve()} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
