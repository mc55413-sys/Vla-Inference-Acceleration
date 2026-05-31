#!/usr/bin/env python3
"""Compare two collected GR00T action files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.quantvla_bench_common import save_json


def run(args: argparse.Namespace) -> dict:
    baseline = np.load(args.baseline_npz, allow_pickle=False)["actions"]
    candidate = np.load(args.candidate_npz, allow_pickle=False)["actions"]
    if baseline.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: baseline={baseline.shape}, candidate={candidate.shape}")

    diff = candidate - baseline
    l2_per_sample = np.linalg.norm(diff, axis=1)
    baseline_l2 = np.linalg.norm(baseline, axis=1)
    rel_l2 = l2_per_sample / np.maximum(baseline_l2, 1e-8)

    # Flat LIBERO action chunk layout is 16 timesteps * 7 action components.
    horizon = 16
    component_dim = baseline.shape[1] // horizon
    gripper_mismatch_rate = None
    if component_dim >= 7 and baseline.shape[1] % horizon == 0:
        baseline_gripper = baseline.reshape(baseline.shape[0], horizon, component_dim)[:, :, 6]
        candidate_gripper = candidate.reshape(candidate.shape[0], horizon, component_dim)[:, :, 6]
        gripper_mismatch_rate = float(
            np.mean((baseline_gripper > 0.5) != (candidate_gripper > 0.5))
        )

    result = {
        "baseline_npz": args.baseline_npz,
        "candidate_npz": args.candidate_npz,
        "num_samples": int(baseline.shape[0]),
        "flat_action_dim": int(baseline.shape[1]),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "l2_mean": float(l2_per_sample.mean()),
        "l2_median": float(np.median(l2_per_sample)),
        "l2_p95": float(np.percentile(l2_per_sample, 95)),
        "relative_l2_mean": float(rel_l2.mean()),
        "relative_l2_median": float(np.median(rel_l2)),
        "relative_l2_p95": float(np.percentile(rel_l2, 95)),
        "cosine_similarity_mean": float(
            np.mean(
                np.sum(baseline * candidate, axis=1)
                / np.maximum(np.linalg.norm(baseline, axis=1) * np.linalg.norm(candidate, axis=1), 1e-8)
            )
        ),
        "gripper_mismatch_rate": gripper_mismatch_rate,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-npz", required=True)
    parser.add_argument("--candidate-npz", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    save_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
