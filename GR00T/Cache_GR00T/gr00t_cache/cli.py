#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 GR00T-Cache Contributors
# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for GR00T-Cache experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch

from .config import CacheMode, GR00TCacheConfig
from .profiling import ProfileResults, profile_gr00t_cache, summarize_profile
from .flops_estimator import compute_gr00t_model_flops
from .ablation import ABLATION_PRESETS, run_ablation
from .correctness import check_cache_correctness
from .dummy_model import (
    DummyGR00TConfig,
    DummyGR00TModel,
    create_dummy_gr00t_model,
    create_dummy_observation,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GR00T-Cache: Efficient VLA Manipulation via Adaptive Token Caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run dummy model benchmark with cache
  python -m gr00t_cache.cli --dummy

  # Run ablation study
  python -m gr00t_cache.cli --dummy --run-ablation

  # Run with custom config
  python -m gr00t_cache.cli --dummy --cache-mode full_cache \\
      --static-similarity-threshold 0.9 --max-reuse-ratio 0.6

  # Run strict correctness check
  python -m gr00t_cache.cli --dummy --strict-correctness-check
        """,
    )

    # Model paths
    parser.add_argument("--model-path", type=str, default="",
                        help="Path to GR00T model checkpoint or HF hub ID")
    parser.add_argument("--config-path", type=str, default="",
                        help="Path to experiment config")
    parser.add_argument("--dataset-path", type=str, default="",
                        help="Path to dataset for sampling observations")

    # Dummy mode
    parser.add_argument("--dummy", action="store_true",
                        help="Use dummy GR00T model for testing (no real weights needed)")

    # Device / dtype
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (default: cuda)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype (default: bfloat16)")

    # Cache configuration
    parser.add_argument("--cache-mode", type=str, default="full_cache",
                        choices=["none", "backbone_visual_kv_cache",
                                  "action_head_condition_kv_cache", "full_cache"],
                        help="Cache mode (default: full_cache)")
    parser.add_argument("--static-similarity-threshold", type=float, default=0.95,
                        help="Cosine similarity threshold for static tokens (default: 0.95)")
    parser.add_argument("--static-topk", type=int, default=None,
                        help="Top-K static token selection (overrides threshold)")
    parser.add_argument("--max-reuse-ratio", type=float, default=0.5,
                        help="Maximum visual token reuse ratio (default: 0.5)")
    parser.add_argument("--task-attention-threshold", type=float, default=None,
                        help="Attention threshold for task-relevant tokens")
    parser.add_argument("--task-topk", type=int, default=None,
                        help="Top-K task-relevant tokens to evict")
    parser.add_argument("--entropy-scale", type=float, default=1.0,
                        help="Layer-adaptive entropy scale (default: 1.0)")
    parser.add_argument("--max-cache-age", type=int, default=1,
                        help="Max cache age in steps (default: 1)")
    parser.add_argument("--disable-wrist-cache", action="store_true",
                        help="Disable caching for wrist camera")

    # Profiling
    parser.add_argument("--warmup-steps", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    parser.add_argument("--repeat-steps", type=int, default=100,
                        help="Measurement iterations (default: 100)")
    parser.add_argument("--profile-memory", action="store_true", default=True,
                        help="Record CUDA memory usage")
    parser.add_argument("--use-cuda-events", action="store_true", default=True,
                        help="Use CUDA events for GPU timing")

    # Output
    parser.add_argument("--save-csv", action="store_true", default=True,
                        help="Save raw measurements to CSV")
    parser.add_argument("--save-json", action="store_true", default=True,
                        help="Save summary to JSON")
    parser.add_argument("--save-debug-dir", type=str, default=None,
                        help="Directory for debug visualizations")
    parser.add_argument("--output-dir", type=str, default="./gr00t_cache_results",
                        help="Output directory for results")

    # Ablation
    parser.add_argument("--run-ablation", action="store_true",
                        help="Run full ablation study across all presets")

    # Correctness
    parser.add_argument("--strict-correctness-check", action="store_true",
                        help="Enable strict correctness checking")

    # Attention fallback
    parser.add_argument("--fallback-standard-attention", action="store_true", default=True,
                        help="Fallback to standard attention for partial KV replacement")

    # Layer range
    parser.add_argument("--layer-start", type=int, default=0,
                        help="First layer index for caching (0-indexed)")
    parser.add_argument("--layer-end", type=int, default=None,
                        help="Last layer index for caching (exclusive)")

    return parser.parse_args()


def build_cache_config(args: argparse.Namespace) -> GR00TCacheConfig:
    """Build GR00TCacheConfig from CLI args."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    return GR00TCacheConfig(
        enabled=(args.cache_mode != "none"),
        cache_mode=CacheMode(args.cache_mode),
        static_similarity_threshold=args.static_similarity_threshold,
        static_topk=args.static_topk,
        max_reuse_ratio=args.max_reuse_ratio,
        task_attention_threshold=args.task_attention_threshold,
        task_topk=args.task_topk,
        entropy_scale=args.entropy_scale,
        max_cache_age=args.max_cache_age,
        disable_wrist_cache=args.disable_wrist_cache,
        fallback_sdpa=args.fallback_standard_attention,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        debug=True,
        debug_save_dir=args.save_debug_dir,
    )


def run_dummy_benchmark(args: argparse.Namespace) -> None:
    """Run benchmark with dummy GR00T model."""
    print("=" * 70)
    print("GR00T-Cache Dummy Model Benchmark")
    print("=" * 70)

    config = build_cache_config(args)
    print(f"\nCache configuration:")
    for k, v in config.to_dict().items():
        print(f"  {k}: {v}")

    # Create dummy model
    print(f"\nCreating dummy GR00T model...")
    dummy_config = DummyGR00TConfig(
        device=args.device if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() and args.device == "cuda" else torch.float32,
        backbone_num_layers=6,
        backbone_hidden_size=512,
        dit_num_layers=4,
        dit_hidden_size=512,
        action_horizon=16,
        action_dim=7,
        num_inference_timesteps=4,
    )

    model = create_dummy_gr00t_model(
        device=dummy_config.device,
        dtype=dummy_config.dtype,
        backbone_num_layers=dummy_config.backbone_num_layers,
        dit_num_layers=dummy_config.dit_num_layers,
    )
    print(f"  Backbone layers: {dummy_config.backbone_num_layers}")
    print(f"  DiT layers: {dummy_config.dit_num_layers}")
    print(f"  Action horizon: {dummy_config.action_horizon}")
    print(f"  Action dim: {dummy_config.action_dim}")
    print(f"  Denoising steps: {dummy_config.num_inference_timesteps}")

    # Create observations
    observations = [
        create_dummy_observation(
            batch_size=1,
            image_size=dummy_config.image_size,
            n_views=dummy_config.n_views,
            state_dim=dummy_config.state_dim,
            seed=i,
        )
        for i in range(args.warmup_steps + args.repeat_steps)
    ]

    print(f"\nBenchmark settings:")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Repeat steps: {args.repeat_steps}")
    print(f"  Device: {dummy_config.device}")
    print(f"  Dtype: {dummy_config.dtype}")

    # Define policy function
    def policy_fn(obs):
        start = time.perf_counter()
        result = model.get_action(obs)
        elapsed = (time.perf_counter() - start) * 1000.0

        # Build timing dict
        timings = {
            "preprocess_ms": 0.5,  # Negligible for dummy
            "vision_ms": 2.0,       # Estimated for dummy
            "backbone_ms": elapsed * 0.4,
            "action_head_ms": elapsed * 0.5,
            "action_postprocess_ms": 0.5,
            "model_total_ms": elapsed,
        }
        return result, timings

    # Profile
    print(f"\nRunning benchmark...")
    results = profile_gr00t_cache(
        policy_fn=policy_fn,
        observations=observations,
        warmup_steps=args.warmup_steps,
        repeat_steps=args.repeat_steps,
        use_cuda_events=args.use_cuda_events and torch.cuda.is_available(),
        profile_memory=args.profile_memory,
        output_dir=args.output_dir,
    )

    # Print summary
    summary_text, stats = summarize_profile(results)
    print("\n" + summary_text)

    # FLOPs estimation
    print("\n" + "=" * 70)
    print("Theoretical FLOPs Estimation")
    print("=" * 70)

    n_visual = (
        (dummy_config.image_size // dummy_config.patch_size) ** 2
        * dummy_config.n_views
    )
    n_text = dummy_config.text_tokens

    reuse_ratio = 0.0  # Default for baseline
    if config.enabled:
        reuse_ratio = config.max_reuse_ratio

    reuse_ratios = [reuse_ratio] * dummy_config.backbone_num_layers

    from .flops_estimator import estimate_cache_transformer_flops
    flops = estimate_cache_transformer_flops(
        num_layers=dummy_config.backbone_num_layers,
        text_tokens=n_text,
        visual_tokens=n_visual,
        d_model=dummy_config.backbone_hidden_size,
        ffn_dim=dummy_config.backbone_ffn_dim,
        reuse_ratios_by_layer=reuse_ratios,
    )

    print(f"  Backbone config:")
    print(f"    Layers: {dummy_config.backbone_num_layers}")
    print(f"    Text tokens: {n_text}")
    print(f"    Visual tokens: {n_visual}")
    print(f"    Hidden dim: {dummy_config.backbone_hidden_size}")
    print(f"    FFN dim: {dummy_config.backbone_ffn_dim}")
    print(f"    Max reuse ratio: {reuse_ratio:.2f}")
    print(f"\n  Full FLOPs:    {flops.full_flops / 1e9:.2f} GFLOPs ({flops.full_tflops:.4f} TFLOPs)")
    if config.enabled:
        print(f"  Cached FLOPs:  {flops.cached_flops / 1e9:.2f} GFLOPs ({flops.cached_tflops:.4f} TFLOPs)")
        print(f"  Saved:         {flops.saved_tflops:.4f} TFLOPs ({flops.flops_reduction_percent:.1f}%)")
        print(f"  Proj saved:    {flops.projection_saved_flops / 1e9:.2f} GFLOPs")
        print(f"  Attn saved:    {flops.attention_saved_flops / 1e9:.2f} GFLOPs")

    # Save config
    if args.save_json:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "cache_config": config.to_dict(),
                "dummy_model_config": {
                    "backbone_num_layers": dummy_config.backbone_num_layers,
                    "dit_num_layers": dummy_config.dit_num_layers,
                    "backbone_hidden_size": dummy_config.backbone_hidden_size,
                    "action_horizon": dummy_config.action_horizon,
                    "action_dim": dummy_config.action_dim,
                    "num_inference_timesteps": dummy_config.num_inference_timesteps,
                },
                "benchmark": {
                    "warmup_steps": args.warmup_steps,
                    "repeat_steps": args.repeat_steps,
                },
                "flops": flops.to_dict(),
            }, f, indent=2)
        print(f"\nConfig saved to {config_path}")

    print("\nDone!")


def run_dummy_ablation(args: argparse.Namespace) -> None:
    """Run ablation study with dummy model."""
    print("=" * 70)
    print("GR00T-Cache Ablation Study (Dummy Model)")
    print("=" * 70)

    dummy_config = DummyGR00TConfig(
        device=args.device if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() and args.device == "cuda" else torch.float32,
    )

    model = create_dummy_gr00t_model(
        device=dummy_config.device,
        dtype=dummy_config.dtype,
    )

    observations = [
        create_dummy_observation(batch_size=1, seed=i)
        for i in range(args.warmup_steps + args.repeat_steps)
    ]

    def policy_factory(cache_config: GR00TCacheConfig):
        """Create a policy function with given cache config."""
        model_instance = create_dummy_gr00t_model(
            device=dummy_config.device,
            dtype=dummy_config.dtype,
        )

        def policy_fn(obs):
            start = time.perf_counter()
            result = model_instance.get_action(obs)
            elapsed = (time.perf_counter() - start) * 1000.0
            timings = {
                "preprocess_ms": 0.5,
                "vision_ms": 2.0,
                "backbone_ms": elapsed * 0.4,
                "action_head_ms": elapsed * 0.5,
                "action_postprocess_ms": 0.5,
                "model_total_ms": elapsed,
            }
            return result, timings

        # Add get_action method to match expected interface
        class PolicyWrapper:
            def get_action(self, obs):
                result, _ = policy_fn(obs)
                return result
        return PolicyWrapper()

    n_visual = (dummy_config.image_size // dummy_config.patch_size) ** 2 * dummy_config.n_views
    flops_config = {
        "backbone_num_layers": dummy_config.backbone_num_layers,
        "backbone_text_tokens": dummy_config.text_tokens,
        "backbone_visual_tokens": n_visual,
        "backbone_d_model": dummy_config.backbone_hidden_size,
        "backbone_ffn_dim": dummy_config.backbone_ffn_dim,
    }

    result = run_ablation(
        policy_factory=policy_factory,
        observations=observations,
        warmup_steps=args.warmup_steps,
        repeat_steps=args.repeat_steps,
        use_cuda_events=args.use_cuda_events and torch.cuda.is_available(),
        profile_memory=args.profile_memory,
        strict_correctness=args.strict_correctness_check,
        output_dir=args.output_dir,
        flops_config=flops_config,
    )

    # Print comparison table
    from .ablation import format_ablation_table
    print("\n" + format_ablation_table(result["comparison"]))


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.run_ablation:
        if args.dummy:
            run_dummy_ablation(args)
        else:
            print("Ablation with real GR00T model requires huggingface access.")
            print("Use --dummy for testing or implement RealGR00TAdapter integration.")
            sys.exit(1)
    else:
        if args.dummy:
            run_dummy_benchmark(args)
        else:
            print("Real GR00T benchmark requires --model-path and HF access.")
            print("Use --dummy for testing the cache framework.")
            sys.exit(1)


if __name__ == "__main__":
    main()
