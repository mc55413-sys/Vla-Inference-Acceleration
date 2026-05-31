# SPDX-FileCopyrightText: Copyright (c) 2025 GR00T-Cache Contributors
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for GR00T-Cache."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

import numpy as np
import torch


def tensor_hash(t: torch.Tensor) -> str:
    """Create a deterministic hash of a tensor's contents."""
    if t is None:
        return "none"
    # Use first few values + shape for fast hashing
    data = t.ravel()[:min(t.numel(), 1000)].cpu().float().numpy()
    shape = tuple(t.shape)
    h = hashlib.md5(data.tobytes() + str(shape).encode())
    return h.hexdigest()[:16]


def cosine_similarity_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute per-token cosine similarity between two batches.

    Args:
        a: [N, D] or [batch, N, D]
        b: [N, D] or [batch, N, D]

    Returns:
        Per-token cosine similarity.
    """
    a_norm = torch.nn.functional.normalize(a.float(), dim=-1)
    b_norm = torch.nn.functional.normalize(b.float(), dim=-1)
    return (a_norm * b_norm).sum(dim=-1)


def compute_entropy(attention_weights: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute attention entropy.

    H = -sum(p * log(p))

    Args:
        attention_weights: [..., seq_len] softmax-normalized attention scores.
        dim: dimension to compute entropy over.

    Returns:
        Entropy values with the specified dimension reduced.
    """
    # Clamp to avoid log(0)
    p = attention_weights.float().clamp(min=1e-12)
    entropy = -(p * p.log()).sum(dim=dim)
    return entropy


def compute_entropy_concentration(entropy: torch.Tensor) -> float:
    """Compute entropy concentration ratio.

    Low concentration → more uniform attention → less crucial tokens.
    High concentration → peaky attention → more task-crucial tokens.

    Args:
        entropy: Per-token attention entropy values.

    Returns:
        Concentration ratio in [0, 1].
    """
    if entropy.numel() == 0:
        return 0.0
    max_entropy = np.log(entropy.shape[-1]) if len(entropy.shape) > 0 else 1.0
    mean_entropy = entropy.mean().item()
    concentration = 1.0 - (mean_entropy / max_entropy) if max_entropy > 0 else 0.0
    return float(np.clip(concentration, 0.0, 1.0))


def compute_patch_similarity(
    current_images: torch.Tensor,
    previous_images: torch.Tensor,
    patch_size: int = 16,
) -> torch.Tensor:
    """Compute per-patch similarity between current and previous images.

    Args:
        current_images: [V, 3, H, W] float32 normalized images.
        previous_images: [V, 3, H, W] float32 normalized images.
        patch_size: Size of each patch for comparison.

    Returns:
        Per-view per-patch cosine similarity.
    """
    V, C, H, W = current_images.shape
    device = current_images.device

    # Extract patches
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    curr_patches = unfold(current_images)  # [V, C*patch^2, num_patches]
    prev_patches = unfold(previous_images)  # [V, C*patch^2, num_patches]

    n_patches = curr_patches.shape[-1]
    curr_patches = curr_patches.reshape(V, n_patches, -1)  # [V, n_patches, C*patch^2]
    prev_patches = prev_patches.reshape(V, n_patches, -1)

    # Per-patch cosine similarity
    sim = cosine_similarity_batch(curr_patches, prev_patches)  # [V, n_patches]
    return sim


def compute_proprio_delta(
    current_state: torch.Tensor,
    previous_state: Optional[torch.Tensor],
) -> float:
    """Compute L2 delta between current and previous proprioception.

    Args:
        current_state: Current state tensor [D] or [B, D].
        previous_state: Previous state tensor [D] or [B, D].

    Returns:
        L2 norm of the difference.
    """
    if previous_state is None:
        return float("inf")
    delta = (current_state.float() - previous_state.float()).norm(p=2)
    return delta.item()


def cuda_sync() -> None:
    """Synchronize CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def elapsed_ms(start: float) -> float:
    """Return elapsed time in milliseconds from a perf_counter start."""
    return (time.perf_counter() - start) * 1000.0


def cpu_timer():
    """Context-helper: returns (start) for use with elapsed_ms."""
    return time.perf_counter()


def gpu_timer_start() -> tuple[float, torch.cuda.Event, torch.cuda.Event]:
    """Start a GPU timer using CUDA events. Returns (cpu_start, start_event, end_event)."""
    cuda_sync()
    start = time.perf_counter()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    return start, start_ev, end_ev


def gpu_timer_end(
    start_ev: torch.cuda.Event, end_ev: torch.cuda.Event
) -> float:
    """End a GPU timer and return elapsed ms."""
    end_ev.record()
    cuda_sync()
    return start_ev.elapsed_time(end_ev)


def pad_to_match(x: torch.Tensor, y: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Pad x to match the size of y along dim."""
    diff = y.shape[dim] - x.shape[dim]
    if diff <= 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = diff
    padding = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, padding], dim=dim)


def dict_to_device(d: dict, device: torch.device) -> dict:
    """Move all tensors in a nested dict to device."""
    import tree
    def to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return x
    return tree.map_structure(to_device, d)


def summarize_statistics(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of values."""
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "p50": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": len(values),
    }
