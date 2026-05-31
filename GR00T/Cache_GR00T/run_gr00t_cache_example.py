#!/usr/bin/env python3
"""
Example runner for GR00T-Cache with dummy model.

This script demonstrates the complete GR00T-Cache pipeline:
1. Create dummy GR00T-like model
2. Run with different cache modes
3. Profile latency and estimate TFLOPs
4. Check correctness
5. Compare results

Usage:
    # Basic benchmark
    python run_gr00t_cache_example.py

    # Ablation study
    python run_gr00t_cache_example.py --ablation

    # With custom cache settings
    python run_gr00t_cache_example.py --max-reuse-ratio 0.7 --task-topk 5

    # Strict correctness check
    python run_gr00t_cache_example.py --strict
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure gr00t_cache is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from gr00t_cache.config import CacheMode, GR00TCacheConfig
from gr00t_cache.cache_manager import GR00TCacheManager
from gr00t_cache.token_index_map import TokenIndexMap
from gr00t_cache.attention_wrapper import CachedAttentionWrapper
from gr00t_cache.profiling import ProfileResults, ProfileTimer, summarize_profile
from gr00t_cache.flops_estimator import (
    estimate_cache_transformer_flops,
    compute_gr00t_model_flops,
)
from gr00t_cache.correctness import compute_action_similarity
from gr00t_cache.dummy_model import (
    DummyGR00TConfig,
    DummyGR00TModel,
    create_dummy_gr00t_model,
    create_dummy_observation,
)
from gr00t_cache.utils import cuda_sync, elapsed_ms


def run_single_benchmark(
    model: DummyGR00TModel,
    observations: list[dict],
    cache_config: GR00TCacheConfig,
    warmup_steps: int = 5,
    repeat_steps: int = 50,
    label: str = "benchmark",
) -> dict[str, Any]:
    """Run a single benchmark with the given cache configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {label}")
    print(f"  Cache mode: {cache_config.cache_mode.value}")
    print(f"  Max reuse: {cache_config.max_reuse_ratio}")
    print(f"  Task eviction: {'on' if cache_config.task_topk else 'off'}")
    print(f"  Layer adaptive: {'on' if cache_config.entropy_scale > 0 else 'off'}")
    print(f"{'='*60}")

    cache_manager = GR00TCacheManager(cache_config)

    latencies = []
    actions_list = []

    for step in range(warmup_steps + repeat_steps):
        obs = observations[step % len(observations)]

        cuda_sync()
        start = time.perf_counter()

        with torch.inference_mode():
            # Build token index map
            input_ids = torch.from_numpy(obs["input_ids"]).to(model.device)
            n_vis = (model.config.image_size // model.config.patch_size) ** 2
            token_map = TokenIndexMap.from_backbone_inputs(
                input_ids=input_ids,
                image_token_index=-1,
                view_info={
                    "external": (0, n_vis),
                    "wrist": (n_vis, n_vis * 2),
                },
            )

            # Slightly varied images for caching benefit
            if step > 0:
                noise = torch.randn_like(
                    torch.from_numpy(obs["pixel_values"])
                ) * 0.01
                obs["pixel_values"] = (obs["pixel_values"] + noise.numpy()).astype(np.float32)

            # Build current images tensor
            current_images = torch.from_numpy(obs["pixel_values"])

            # Compute reuse plan
            if cache_config.enabled:
                reuse_plan = cache_manager.get_reuse_plan(
                    current_images=current_images,
                    current_proprio=torch.from_numpy(obs["state"]).float(),
                    current_token_map=token_map,
                )
                cache_manager._current_reuse_plan = reuse_plan

            # Run model
            action = model.get_action(obs)

            # Update cache
            if cache_config.enabled:
                cache_manager.update_cache(
                    current_images=current_images,
                    current_visual_tokens=None,
                    current_token_map=token_map,
                    current_instruction=None,
                    current_proprio=torch.from_numpy(obs["state"]).float(),
                    attention_maps={},
                    layer_kv={},
                )

        cuda_sync()
        elapsed = elapsed_ms(start)

        if step >= warmup_steps:
            latencies.append(elapsed)
            actions_list.append(action["action"])

    # Compute statistics
    lat_arr = np.array(latencies)
    results = {
        "label": label,
        "cache_mode": cache_config.cache_mode.value,
        "mean_ms": float(lat_arr.mean()),
        "std_ms": float(lat_arr.std()),
        "p50_ms": float(np.median(lat_arr)),
        "p90_ms": float(np.percentile(lat_arr, 90)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "min_ms": float(lat_arr.min()),
        "max_ms": float(lat_arr.max()),
        "control_frequency_hz": 1000.0 / lat_arr.mean() if lat_arr.mean() > 0 else 0,
        "cache_stats": cache_manager.stats(),
    }

    # Theoretical FLOPs
    config = model.config
    n_vis_per_view = (config.image_size // config.patch_size) ** 2
    n_visual = n_vis_per_view * config.n_views
    n_text = config.text_tokens

    reuse_ratio = results["cache_stats"].get("reuse_ratio", 0.0)
    reuse_ratios = [reuse_ratio] * config.backbone_num_layers

    flops = estimate_cache_transformer_flops(
        num_layers=config.backbone_num_layers,
        text_tokens=n_text,
        visual_tokens=n_visual,
        d_model=config.backbone_hidden_size,
        ffn_dim=config.backbone_ffn_dim,
        reuse_ratios_by_layer=reuse_ratios,
    )
    results["flops"] = flops.to_dict()

    # Print results
    print(f"\n  Latency:  mean={results['mean_ms']:.2f}ms  "
          f"p50={results['p50_ms']:.2f}ms  p95={results['p95_ms']:.2f}ms  "
          f"ctrl={results['control_frequency_hz']:.1f}Hz")
    print(f"  FLOPs:    full={flops.full_tflops:.4f}T  "
          f"cached={flops.cached_tflops:.4f}T  "
          f"saved={flops.flops_reduction_percent:.1f}%")
    print(f"  Cache:    hits={results['cache_stats']['cache_hit_rate']:.2f}  "
          f"reuse={results['cache_stats']['reuse_ratio']:.2f}")

    return results


def run_comparison_benchmark(args) -> None:
    """Run baseline vs cached comparison."""
    config = DummyGR00TConfig(
        device="cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        backbone_num_layers=args.backbone_layers,
        dit_num_layers=args.dit_layers,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        num_inference_timesteps=args.denoising_steps,
    )

    # Create observations
    observations = [
        create_dummy_observation(batch_size=1, seed=i)
        for i in range(args.warmup + args.iters)
    ]

    results_list = []

    # 1. Baseline (no cache)
    model_baseline = create_dummy_gr00t_model(
        device=config.device, dtype=config.dtype,
        backbone_num_layers=config.backbone_num_layers,
        dit_num_layers=config.dit_num_layers,
    )
    baseline_config = GR00TCacheConfig(enabled=False, cache_mode=CacheMode.NONE)
    baseline_result = run_single_benchmark(
        model_baseline, observations, baseline_config,
        warmup_steps=args.warmup, repeat_steps=args.iters,
        label="baseline",
    )
    results_list.append(baseline_result)

    # Store baseline actions for correctness check
    baseline_actions = []
    model_ref = create_dummy_gr00t_model(
        device=config.device, dtype=config.dtype,
        backbone_num_layers=config.backbone_num_layers,
        dit_num_layers=config.dit_num_layers,
    )
    for i in range(min(args.iters, 20)):
        obs = observations[args.warmup + i]
        with torch.inference_mode():
            action = model_ref.get_action(obs)
        baseline_actions.append(torch.from_numpy(action["action"]))

    # 2. Full cache
    model_cached = create_dummy_gr00t_model(
        device=config.device, dtype=config.dtype,
        backbone_num_layers=config.backbone_num_layers,
        dit_num_layers=config.dit_num_layers,
    )
    cache_config = GR00TCacheConfig(
        enabled=True,
        cache_mode=CacheMode(args.cache_mode),
        static_similarity_threshold=args.static_sim,
        max_reuse_ratio=args.max_reuse,
        task_topk=args.task_topk,
        entropy_scale=args.entropy_scale,
        max_cache_age=args.max_cache_age,
        disable_wrist_cache=args.disable_wrist,
        per_view_budget=args.per_view_budget,
        debug=args.debug,
    )
    cached_result = run_single_benchmark(
        model_cached, observations, cache_config,
        warmup_steps=args.warmup, repeat_steps=args.iters,
        label="cached",
    )
    results_list.append(cached_result)

    # Correctness check
    if args.strict:
        print(f"\n{'='*60}")
        print("Correctness Check")
        print(f"{'='*60}")

        model_check = create_dummy_gr00t_model(
            device=config.device, dtype=config.dtype,
            backbone_num_layers=config.backbone_num_layers,
            dit_num_layers=config.dit_num_layers,
        )
        cache_manager = GR00TCacheManager(cache_config)

        all_sims = []
        all_l2 = []
        for i in range(min(args.iters, 20)):
            obs = observations[args.warmup + i]
            with torch.inference_mode():
                cached_action = model_check.get_action(obs)
            cached_tensor = torch.from_numpy(cached_action["action"])
            sim = compute_action_similarity(cached_tensor, baseline_actions[i])

            all_sims.append(sim["action_cosine_similarity_vs_baseline"])
            all_l2.append(sim["action_l2_diff_vs_baseline"])

        all_sims_np = np.array(all_sims)
        all_l2_np = np.array(all_l2)
        print(f"  Cosine similarity: {all_sims_np.mean():.6f} ± {all_sims_np.std():.6f}")
        print(f"  L2 diff: {all_l2_np.mean():.6f} ± {all_l2_np.std():.6f}")
        print(f"  Min cos sim: {all_sims_np.min():.6f}")
        print(f"  Max L2 diff: {all_l2_np.max():.6f}")

    # Comparison table
    print(f"\n{'='*80}")
    print("Comparison Summary")
    print(f"{'='*80}")
    print(f"{'Config':<20s} {'E2E(ms)':>8s} {'Ctrl(Hz)':>8s} {'FLOPs(T)':>10s} {'FLOPs↓':>7s}")

    for r in results_list:
        flops = r.get("flops", {})
        full_t = flops.get("full_tflops", 0)
        cached_t = flops.get("cached_tflops", 0)
        saved_pct = flops.get("flops_reduction_percent", 0)
        print(
            f"{r['label']:<20s} "
            f"{r['mean_ms']:>7.1f}  "
            f"{r['control_frequency_hz']:>7.1f}  "
            f"{cached_t:>9.4f}  "
            f"{saved_pct:>6.1f}%"
        )

    speedup = baseline_result["mean_ms"] / cached_result["mean_ms"] if cached_result["mean_ms"] > 0 else 1.0
    print(f"\n  Speedup: {speedup:.2f}x")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results_list, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="GR00T-Cache Example Runner")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--cache-mode", default="full_cache",
                        choices=["none", "backbone_visual_kv_cache",
                                  "action_head_condition_kv_cache", "full_cache"])
    parser.add_argument("--backbone-layers", type=int, default=6)
    parser.add_argument("--dit-layers", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--static-sim", type=float, default=0.95)
    parser.add_argument("--max-reuse", type=float, default=0.5)
    parser.add_argument("--task-topk", type=int, default=None)
    parser.add_argument("--entropy-scale", type=float, default=1.0)
    parser.add_argument("--max-cache-age", type=int, default=1)
    parser.add_argument("--disable-wrist", action="store_true")
    parser.add_argument("--per-view-budget", action="store_true", default=True)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", type=str, default="./gr00t_cache_results/comparison.json")
    args = parser.parse_args()

    if args.ablation:
        # Import and run the full CLI ablation
        from gr00t_cache.cli import run_dummy_ablation
        class ArgsNS:
            pass
        cli_args = ArgsNS()
        for k, v in vars(args).items():
            setattr(cli_args, k, v)
        cli_args.device = args.device
        cli_args.warmup_steps = args.warmup
        cli_args.repeat_steps = args.iters
        cli_args.use_cuda_events = torch.cuda.is_available()
        cli_args.profile_memory = True
        cli_args.output_dir = str(Path(args.output).parent)
        cli_args.strict_correctness_check = args.strict
        run_dummy_ablation(cli_args)
    else:
        run_comparison_benchmark(args)


if __name__ == "__main__":
    main()
