#!/usr/bin/env python3
"""Shared utilities for FastV pruning benchmark scripts."""

from __future__ import annotations

import json
import os
import random
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy


INTERESTING_ENV_KEYS = (
    "GR00T_FASTV_ENABLE",
    "GR00T_FASTV_K",
    "GR00T_FASTV_R",
    "GR00T_FASTV_VERBOSE",
    "GR00T_ATTN_IMPLEMENTATION",
    "CUDA_VISIBLE_DEVICES",
    "PYTHONPATH",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def sample_libero_observation(height: int = 256, width: int = 256, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "video.image": rng.integers(0, 256, size=(1, height, width, 3), dtype=np.uint8),
        "video.wrist_image": rng.integers(0, 256, size=(1, height, width, 3), dtype=np.uint8),
        # Match examples/Libero/eval/run_libero_eval.py: np.array(...) defaults to float64.
        # StateActionTransform caches the first dtype it sees, so synthetic probes must
        # use the same dtype as real LIBERO observations when sharing a long-lived server.
        "state.x": np.array([[0.0]], dtype=np.float64),
        "state.y": np.array([[0.0]], dtype=np.float64),
        "state.z": np.array([[0.0]], dtype=np.float64),
        "state.roll": np.array([[0.0]], dtype=np.float64),
        "state.pitch": np.array([[0.0]], dtype=np.float64),
        "state.yaw": np.array([[0.0]], dtype=np.float64),
        "state.gripper": np.array([[0.0, 0.0]], dtype=np.float64),
        "annotation.human.action.task_description": ["put the black bowl on the plate"],
    }


def summarize(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        for key, value in sample.items():
            grouped[key].append(float(value))

    summary: dict[str, dict[str, float]] = {}
    for key, values in grouped.items():
        arr = np.array(values, dtype=np.float64)
        summary[key] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "std": float(statistics.pstdev(values)),
        }
    return summary


def env_snapshot() -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in INTERESTING_ENV_KEYS:
            env[key] = value
    return dict(sorted(env.items()))


def flag_enabled(name: str) -> bool:
    return os.environ.get(name, "0") not in ("0", "false", "False", "")


def runtime_metadata(variant: str = "") -> dict[str, Any]:
    return {
        "variant": variant,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "fastv_env_enabled": flag_enabled("GR00T_FASTV_ENABLE"),
        "attn_implementation": os.environ.get("GR00T_ATTN_IMPLEMENTATION", "sdpa"),
        "env": env_snapshot(),
    }


def parameter_bytes(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        ptr = tensor.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += tensor.numel() * tensor.element_size()
    return int(total)


def nvidia_smi_memory_mb() -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        pid = str(os.getpid())
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in output.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 2 and parts[0] == pid:
                return int(parts[1])
    except Exception:
        return None
    return None


def load_libero_policy(
    model_path: str,
    data_config: str,
    embodiment_tag: str,
    denoising_steps: int,
) -> Gr00tPolicy:
    cfg = load_data_config(data_config)
    return Gr00tPolicy(
        model_path=model_path,
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        embodiment_tag=embodiment_tag,
        denoising_steps=denoising_steps,
    )


def save_json(path: str | os.PathLike[str], data: dict[str, Any]) -> None:
    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))
