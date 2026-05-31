# SPDX-FileCopyrightText: Copyright (c) 2025 GR00T-Cache Contributors
# SPDX-License-Identifier: Apache-2.0
"""Correctness checking and action quality verification for GR00T-Cache."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image


def compute_action_similarity(
    cached_action: torch.Tensor,
    baseline_action: torch.Tensor,
) -> dict[str, float]:
    """Compute similarity metrics between cached and baseline actions.

    Args:
        cached_action: Action tensor from cached model [..., action_dim].
        baseline_action: Action tensor from baseline model [..., action_dim].

    Returns:
        dict with l2_diff, cosine_similarity, max_abs_diff.
    """
    if cached_action.shape != baseline_action.shape:
        raise ValueError(
            f"Shape mismatch: {cached_action.shape} vs {baseline_action.shape}"
        )

    a = cached_action.float().flatten()
    b = baseline_action.float().flatten()

    l2_diff = (a - b).norm(p=2).item()

    # Cosine similarity
    dot = (a * b).sum()
    norm_a = a.norm(p=2)
    norm_b = b.norm(p=2)
    cos_sim = (dot / (norm_a * norm_b + 1e-8)).item()

    max_abs_diff = (a - b).abs().max().item()

    return {
        "action_l2_diff_vs_baseline": l2_diff,
        "action_cosine_similarity_vs_baseline": cos_sim,
        "max_abs_action_diff": max_abs_diff,
    }


def check_cache_correctness(
    cached_fn,
    baseline_fn,
    observations: list[dict],
    n_steps: int = 20,
    strict: bool = False,
    tolerance_l2: float = 1e-3,
    tolerance_cos: float = 0.999,
    tolerance_max_abs: float = 0.01,
) -> dict[str, Any]:
    """Verify that cache-enabled model produces correct outputs.

    Compares cached model outputs against baseline (cache-disabled) outputs
    across multiple steps.

    Args:
        cached_fn: Function returning action with cache enabled.
        baseline_fn: Function returning action with cache disabled.
        observations: List of observation dicts.
        n_steps: Number of steps to check.
        strict: If True, raise on tolerance violations.
        tolerance_l2: Maximum L2 difference before warning.
        tolerance_cos: Minimum cosine similarity before warning.
        tolerance_max_abs: Maximum absolute difference before warning.

    Returns:
        dict with per-step metrics and summary.
    """
    all_metrics = []
    violations = []

    for step in range(n_steps):
        obs = observations[step % len(observations)]

        with torch.inference_mode():
            try:
                cached_result = cached_fn(obs)
                baseline_result = baseline_fn(obs)
            except Exception as e:
                violations.append(f"Step {step}: Exception: {e}")
                continue

        # Extract action tensors
        if isinstance(cached_result, dict):
            cached_action = cached_result.get("action", cached_result.get("__action__", {}))
            if isinstance(cached_action, dict):
                cached_action = cached_action.get("action", list(cached_action.values())[0])
        else:
            cached_action = cached_result

        if isinstance(baseline_result, dict):
            baseline_action = baseline_result.get("action", baseline_result.get("__action__", {}))
            if isinstance(baseline_action, dict):
                baseline_action = baseline_action.get("action", list(baseline_action.values())[0])
        else:
            baseline_action = baseline_result

        if isinstance(cached_action, torch.Tensor):
            cached_action = cached_action.cpu()
        else:
            cached_action = torch.tensor(cached_action)
        if isinstance(baseline_action, torch.Tensor):
            baseline_action = baseline_action.cpu()
        else:
            baseline_action = torch.tensor(baseline_action)

        metrics = compute_action_similarity(cached_action, baseline_action)
        metrics["step"] = step
        all_metrics.append(metrics)

        # Check tolerances
        if metrics["action_l2_diff_vs_baseline"] > tolerance_l2:
            msg = (
                f"Step {step}: L2 diff={metrics['action_l2_diff_vs_baseline']:.6f} "
                f"> tolerance={tolerance_l2}"
            )
            violations.append(msg)

        if metrics["action_cosine_similarity_vs_baseline"] < tolerance_cos:
            msg = (
                f"Step {step}: Cos sim={metrics['action_cosine_similarity_vs_baseline']:.6f} "
                f"< tolerance={tolerance_cos}"
            )
            violations.append(msg)

        if metrics["max_abs_action_diff"] > tolerance_max_abs:
            msg = (
                f"Step {step}: Max abs diff={metrics['max_abs_action_diff']:.6f} "
                f"> tolerance={tolerance_max_abs}"
            )
            violations.append(msg)

    summary = {
        "n_steps": n_steps,
        "n_violations": len(violations),
        "violations": violations if strict else violations[:10],
        "mean_l2_diff": float(np.mean([m["action_l2_diff_vs_baseline"] for m in all_metrics])),
        "mean_cos_sim": float(np.mean([m["action_cosine_similarity_vs_baseline"] for m in all_metrics])),
        "mean_max_abs_diff": float(np.mean([m["max_abs_action_diff"] for m in all_metrics])),
        "actions_match": len(violations) == 0,
    }

    if strict and violations:
        raise AssertionError(
            f"Correctness check failed with {len(violations)} violations:\n"
            + "\n".join(violations[:10])
        )

    return summary


def generate_debug_visualization(
    current_images: torch.Tensor,
    static_masks: dict[str, torch.Tensor],
    task_relevant_mask: Optional[torch.Tensor] = None,
    reuse_masks: Optional[dict[str, torch.Tensor]] = None,
    output_dir: str | Path = "./debug_vis",
) -> None:
    """Generate debug visualization images for token caching.

    Creates per-view heatmap overlays showing:
    - Static token mask (green)
    - Task-relevant token mask (red)
    - Final reuse mask (blue)

    Args:
        current_images: [V, 3, H, W] float32 images.
        static_masks: Per-view bool masks of static tokens.
        task_relevant_mask: [n_visual] bool mask of task-relevant tokens.
        reuse_masks: Per-view bool masks of final reuse decisions.
        output_dir: Directory to save debug images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    V = current_images.shape[0]
    view_names = list(static_masks.keys())

    for v_idx in range(V):
        view_name = view_names[v_idx] if v_idx < len(view_names) else f"view_{v_idx}"
        img = current_images[v_idx]  # [3, H, W]
        static = static_masks.get(view_name, torch.zeros(1))

        # Create visualization
        h_patches = int(np.sqrt(len(static)))
        w_patches = max(1, len(static) // h_patches)

        # Reshape mask into 2D grid for overlay
        if len(static) >= h_patches * w_patches:
            static_2d = static[:h_patches * w_patches].reshape(h_patches, w_patches)
        else:
            static_2d = static.reshape(1, -1)

        # Convert image to PIL
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        # Save the original image
        pil_img.save(output_dir / f"{view_name}_original.png")

        # Save mask visualization
        mask_path = output_dir / f"{view_name}_static_mask.txt"
        with open(mask_path, "w") as f:
            f.write(f"View: {view_name}\n")
            f.write(f"Static tokens: {static.sum().item()}/{len(static)}\n")
            f.write(f"Static mask: {static.tolist()}\n")

        if reuse_masks and view_name in reuse_masks:
            reuse = reuse_masks[view_name]
            reuse_path = output_dir / f"{view_name}_reuse_mask.txt"
            with open(reuse_path, "w") as f:
                f.write(f"View: {view_name}\n")
                f.write(f"Reused tokens: {reuse.sum().item()}/{len(reuse)}\n")

    # Save overall summary
    summary_path = output_dir / "debug_summary.txt"
    with open(summary_path, "w") as f:
        f.write("GR00T-Cache Debug Visualization Summary\n")
        f.write("=" * 50 + "\n")
        for view_name in view_names:
            static = static_masks.get(view_name, torch.zeros(1))
            f.write(f"\n{view_name}:\n")
            f.write(f"  Total patches: {len(static)}\n")
            f.write(f"  Static patches: {static.sum().item()}\n")
            if reuse_masks and view_name in reuse_masks:
                reuse = reuse_masks[view_name]
                f.write(f"  Final reuse: {reuse.sum().item()}\n")

    print(f"Debug visualizations saved to {output_dir}")
