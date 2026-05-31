#!/usr/bin/env python3
"""Fail fast when a benchmark command is not running in a CUDA-visible process."""

from __future__ import annotations

import argparse
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", default="Prune-GR00T benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        print(
            f"[Prune-GR00T][ERROR] CUDA is not available for {args.context}. "
            "This run would fall back to CPU, so it is stopped.",
            file=sys.stderr,
        )
        print(
            "[Prune-GR00T][ERROR] Activate the GPU conda environment and check "
            "CUDA_VISIBLE_DEVICES / NVIDIA driver visibility.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    device = torch.cuda.current_device()
    print(
        "[Prune-GR00T] CUDA available: "
        f"device={device} name={torch.cuda.get_device_name(device)} "
        f"torch_cuda={torch.version.cuda}"
    )


if __name__ == "__main__":
    main()
