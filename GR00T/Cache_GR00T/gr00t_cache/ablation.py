# SPDX-FileCopyrightText: Copyright (c) 2025 GR00T-Cache Contributors
# SPDX-License-Identifier: Apache-2.0
"""Cache ablation study runner for GR00T-Cache.

Implements 7 preset configurations covering all combinations of:
- Static token selection
- Task-relevant eviction
- Layer-adaptive reuse
- Backbone vs action head caching
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from .config import CacheMode, GR00TCacheConfig
from .profiling import ProfileResults, profile_policy_pipeline, summarize_profile
from .flops_estimator import FLOPSEstimate
from .correctness import check_cache_correctness

# ---------------------------------------------------------------------------
# Ablation presets
# ---------------------------------------------------------------------------

ABLATION_PRESETS: dict[str, GR00TCacheConfig] = {
    "baseline": GR00TCacheConfig(enabled=False, cache_mode=CacheMode.NONE),

    "static_only": GR00TCacheConfig(
        enabled=True,
        cache_mode=CacheMode.BACKBONE_VISUAL_KV_CACHE,
        static_similarity_threshold=0.95,
        max_reuse_ratio=0.5,
        task_attention_threshold=None,
        task_topk=None,
        entropy_scale=0.0,  # Disable layer-adaptive
        collect_attention_maps=False,
    ),

    "static_plus_task_eviction": GR00TCacheConfig(
        enabled=True,
        cache_mode=CacheMode.BACKBONE_VISUAL_KV_CACHE,
        static_similarity_threshold=0.95,
        max_reuse_ratio=0.5,
        task_topk=10,  # Top-10 attention tokens evicted
        entropy_scale=0.0,  # Disable layer-adaptive
        collect_attention_maps=True,
    ),

    "static_plus_layer_adaptive": GR00TCacheConfig(
        enabled=True,
        cache_mode=CacheMode.BACKBONE_VISUAL_KV_CACHE,
        static_similarity_threshold=0.95,
        max_reuse_ratio=0.7,
        min_reuse_ratio=0.1,
        task_attention_threshold=None,
        task_topk=None,
        entropy_scale=1.0,  # Full layer-adaptive
        collect_attention_maps=True,
    ),

    "full_gr00t_cache": GR00TCacheConfig(
        enabled=True,
        cache_mode=CacheMode.FULL_CACHE,
        static_similarity_threshold=0.95,
        max_reuse_ratio=0.6,
        min_reuse_ratio=0.05,
        task_topk=10,
        entropy_scale=1.0,
        collect_attention_maps=True,
    ),

    "action_head_condition_cache_only": GR00TCacheConfig(
        enabled=True,
        cache_mode=CacheMode.ACTION_HEAD_CONDITION_KV_CACHE,
        static_similarity_threshold=0.95,
        max_reuse_ratio=0.6,
        entropy_scale=0.0,
        collect_attention_maps=False,
    ),

    "backbone_visual_cache_only": GR00TCacheConfig(
        enabled=True,
        cache_mode=CacheMode.BACKBONE_VISUAL_KV_CACHE,
        static_similarity_threshold=0.95,
        max_reuse_ratio=0.5,
        task_topk=10,
        entropy_scale=1.0,
        collect_attention_maps=True,
    ),
}


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    policy_factory: Callable[[GR00TCacheConfig], Any],
    observations: list[dict],
    presets: Optional[list[str]] = None,
    warmup_steps: int = 10,
    repeat_steps: int = 100,
    use_cuda_events: bool = True,
    profile_memory: bool = True,
    strict_correctness: bool = False,
    output_dir: Optional[str | Path] = None,
    flops_config: Optional[dict] = None,
) -> dict[str, Any]:
    """Run ablation study across specified cache presets.

    For each preset:
    1. Create a policy with the preset cache config
    2. Profile latency, memory
    3. Estimate TFLOPs
    4. Check correctness against baseline

    Args:
        policy_factory: Function that takes a GR00TCacheConfig and returns
            a policy instance (with get_action method).
        observations: List of observation dicts.
        presets: List of preset names to run. Defaults to all.
        warmup_steps: Warmup iterations.
        repeat_steps: Measurement iterations.
        use_cuda_events: Use CUDA events for GPU timing.
        profile_memory: Record CUDA memory usage.
        strict_correctness: Enable strict correctness checking.
        output_dir: Directory to save results.
        flops_config: Optional dict with FLOPs estimation parameters:
            backbone_num_layers, backbone_text_tokens, backbone_visual_tokens,
            backbone_d_model, backbone_ffn_dim, etc.

    Returns:
        dict mapping preset_name → results.
    """
    if presets is None:
        presets = list(ABLATION_PRESETS.keys())

    # Run baseline first
    baseline_config = ABLATION_PRESETS["baseline"]
    print("=" * 70)
    print("Running baseline (cache disabled)...")
    print("=" * 70)

    baseline_policy = policy_factory(baseline_config)
    baseline_results = profile_policy_pipeline(
        policy_fn=lambda obs: baseline_policy.get_action(obs),
        observations=observations,
        warmup_steps=warmup_steps,
        repeat_steps=repeat_steps,
        use_cuda_events=use_cuda_events,
        profile_memory=profile_memory,
    )
    baseline_results.model_name = "gr00t"
    baseline_results.cache_mode = "baseline"

    all_results = {"baseline": baseline_results}

    # Run each ablation preset
    for preset_name in presets:
        if preset_name == "baseline":
            continue
        if preset_name not in ABLATION_PRESETS:
            print(f"Unknown preset: {preset_name}, skipping")
            continue

        config = ABLATION_PRESETS[preset_name]
        print(f"\n{'=' * 70}")
        print(f"Running: {preset_name}")
        print(f"  cache_mode={config.cache_mode.value}")
        print(f"  static_sim_threshold={config.static_similarity_threshold}")
        print(f"  max_reuse_ratio={config.max_reuse_ratio}")
        print(f"  task_eviction={'on' if config.task_topk else 'off'}")
        print(f"  layer_adaptive={'on' if config.entropy_scale > 0 else 'off'}")
        print("=" * 70)

        policy = policy_factory(config)
        results = profile_policy_pipeline(
            policy_fn=lambda obs: policy.get_action(obs),
            observations=observations,
            warmup_steps=warmup_steps,
            repeat_steps=repeat_steps,
            use_cuda_events=use_cuda_events,
            profile_memory=profile_memory,
        )
        results.model_name = "gr00t"
        results.cache_mode = preset_name
        results.config_snapshot = config.to_dict()

        # Correctness check against baseline
        if strict_correctness:
            print(f"  Checking correctness vs baseline...")
            def cached_fn(obs):
                return policy.get_action(obs)
            def baseline_fn(obs):
                return baseline_policy.get_action(obs)

            try:
                correctness = check_cache_correctness(
                    cached_fn=cached_fn,
                    baseline_fn=baseline_fn,
                    observations=observations,
                    n_steps=min(repeat_steps, 20),
                    strict=strict_correctness,
                )
                results.action_l2_diff = [correctness["mean_l2_diff"]]
                results.action_cosine_sim = [correctness["mean_cos_sim"]]
                results.max_abs_action_diff = [correctness["mean_max_abs_diff"]]
                print(f"    L2 diff: {correctness['mean_l2_diff']:.6f}")
                print(f"    Cos sim: {correctness['mean_cos_sim']:.6f}")
            except Exception as e:
                print(f"    Correctness check failed: {e}")

        all_results[preset_name] = results

    # Generate comparison table
    comparison = _build_comparison_table(all_results, flops_config)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, results in all_results.items():
            results.save_csv(output_dir / f"{name}_raw.csv")
            results.save_summary(output_dir / f"{name}_summary.json")

        with open(output_dir / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

        comparison_text = format_ablation_table(comparison)
        (output_dir / "comparison.txt").write_text(comparison_text)
        print(f"\nResults saved to {output_dir}")

    return {"results": all_results, "comparison": comparison}


def _build_comparison_table(
    all_results: dict[str, ProfileResults],
    flops_config: Optional[dict] = None,
) -> dict[str, Any]:
    """Build a comparison table across all ablation presets."""
    baseline_summary = all_results.get("baseline")
    if baseline_summary is None:
        return {}

    baseline_stats = baseline_summary.summarize()
    baseline_e2e = baseline_stats.get("end_to_end_ms", {}).get("mean", 0)

    comparison = {}
    for name, results in all_results.items():
        stats = results.summarize()
        e2e_mean = stats.get("end_to_end_ms", {}).get("mean", 0)

        entry = {
            "end_to_end_ms": e2e_mean,
            "model_latency_ms": stats.get("model_latency_ms", {}).get("mean", 0),
            "vision_ms": stats.get("vision_ms", {}).get("mean", 0),
            "backbone_ms": stats.get("backbone_ms", {}).get("mean", 0),
            "action_head_ms": stats.get("action_head_ms", {}).get("mean", 0),
            "action_postprocess_ms": stats.get("action_postprocess_ms", {}).get("mean", 0),
            "control_frequency_hz": 1000.0 / e2e_mean if e2e_mean > 0 else 0,
            "speedup_vs_baseline": baseline_e2e / e2e_mean if e2e_mean > 0 else 1.0,
            "peak_cuda_memory_mb": stats.get("peak_cuda_memory_mb", {}).get("mean", 0),
            "reused_token_ratio": stats.get("reused_token_ratio", {}).get("mean", 0),
            "evicted_task_token_ratio": stats.get("evicted_task_token_ratio", {}).get("mean", 0),
            "cache_hit_rate": stats.get("cache_hit_rate", {}).get("mean", 0),
            "cache_reset_count": results.cache_reset_count[-1] if results.cache_reset_count else 0,
            "action_l2_diff": stats.get("action_l2_diff", {}).get("mean", float("nan")),
            "action_cosine_sim": stats.get("action_cosine_similarity_vs_baseline", {}).get("mean", float("nan")),
            "max_abs_action_diff": stats.get("max_abs_action_diff", {}).get("mean", float("nan")),
        }

        # FLOPs estimation
        if flops_config is not None and name != "baseline":
            try:
                from .flops_estimator import estimate_cache_transformer_flops
                reuse_ratio = entry["reused_token_ratio"]
                n_layers = flops_config.get("backbone_num_layers", 32)
                reuse_ratios = [reuse_ratio] * n_layers
                flops = estimate_cache_transformer_flops(
                    num_layers=n_layers,
                    text_tokens=flops_config["backbone_text_tokens"],
                    visual_tokens=flops_config["backbone_visual_tokens"],
                    d_model=flops_config["backbone_d_model"],
                    ffn_dim=flops_config["backbone_ffn_dim"],
                    reuse_ratios_by_layer=reuse_ratios,
                )
                entry.update(flops.to_dict())
            except Exception:
                pass

        comparison[name] = entry

    return comparison


def format_ablation_table(comparison: dict[str, Any]) -> str:
    """Format comparison results as a human-readable table."""
    if not comparison:
        return "No results to display."

    lines = []
    lines.append("=" * 120)
    lines.append("GR00T-Cache Ablation Study Results")
    lines.append("=" * 120)

    # Header
    headers = [
        "Preset", "E2E(ms)", "Model(ms)", "Backbone(ms)", "ActionHead(ms)",
        "Ctrl(Hz)", "Speedup", "Mem(MB)", "Reuse%", "FLOPs↓%", "L2diff"
    ]
    header_fmt = (
        "{:<30s} {:>9s}  {:>9s}  {:>12s}  {:>13s}  "
        "{:>7s}  {:>7s}  {:>8s}  {:>6s}  {:>7s}  {:>8s}"
    )
    lines.append(header_fmt.format(*headers))
    lines.append("-" * 120)

    row_fmt = (
        "{:<30s} {:>8.1f}  {:>8.1f}  {:>11.1f}  {:>12.1f}  "
        "{:>6.1f}  {:>6.2f}x  {:>7.0f}  {:>5.1f}%  {:>6.1f}%  {:>8.4f}"
    )

    for name, entry in comparison.items():
        lines.append(row_fmt.format(
            name,
            entry.get("end_to_end_ms", 0),
            entry.get("model_latency_ms", 0),
            entry.get("backbone_ms", 0),
            entry.get("action_head_ms", 0),
            entry.get("control_frequency_hz", 0),
            entry.get("speedup_vs_baseline", 1.0),
            entry.get("peak_cuda_memory_mb", 0),
            entry.get("reused_token_ratio", 0) * 100,
            entry.get("flops_reduction_percent", 0),
            entry.get("action_l2_diff", float("nan")),
        ))

    lines.append("=" * 120)
    return "\n".join(lines)
