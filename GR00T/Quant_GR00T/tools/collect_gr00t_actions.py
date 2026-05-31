#!/usr/bin/env python3
"""Collect GR00T action chunks for the current baseline/quantized environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.quantvla_bench_common import (
    count_duquant_modules,
    load_libero_policy,
    runtime_metadata,
    sample_libero_observation,
    save_json,
    set_reproducibility,
    sync_cuda,
)


ACTION_KEYS = [
    "action.x",
    "action.y",
    "action.z",
    "action.roll",
    "action.pitch",
    "action.yaw",
    "action.gripper",
]


def flatten_action(action: dict[str, Any]) -> np.ndarray:
    parts = []
    for key in ACTION_KEYS:
        value = action[key]
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        parts.append(np.asarray(value).reshape(-1))
    return np.concatenate(parts, axis=0).astype(np.float32)


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_reproducibility(args.seed)
    policy = load_libero_policy(
        model_path=args.model_path,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )

    rows = []
    for idx in range(args.num_samples):
        obs = sample_libero_observation(args.image_height, args.image_width, args.seed + idx)
        set_reproducibility(args.seed + idx)
        action = policy.get_action(obs)
        sync_cuda()
        rows.append(flatten_action(action))

    actions = np.stack(rows, axis=0)
    np.savez_compressed(
        args.output_npz,
        actions=actions,
        action_keys=np.array(ACTION_KEYS),
        seed=np.array(args.seed),
        num_samples=np.array(args.num_samples),
    )

    summary = {
        "metadata": runtime_metadata(args.variant),
        "model": {
            "duquant_linear_modules": count_duquant_modules(policy.model),
            "denoising_steps": int(policy.denoising_steps),
        },
        "output_npz": args.output_npz,
        "num_samples": int(args.num_samples),
        "action_dim_flat": int(actions.shape[1]),
        "action_mean": float(actions.mean()),
        "action_std": float(actions.std()),
        "action_abs_max": float(np.max(np.abs(actions))),
    }
    if args.output_json:
        save_json(args.output_json, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="")
    parser.add_argument("--model-path", default="youliangtan/gr00t-n1.5-libero-long-posttrain")
    parser.add_argument(
        "--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig"
    )
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
