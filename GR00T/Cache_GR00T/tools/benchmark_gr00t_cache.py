#!/usr/bin/env python3
"""
GR00T-Cache Latency Benchmark — 5-Stage Breakdown

Measures latency across the full GR00T inference pipeline:
  1. Data latency      — observation ingestion
  2. Preprocess latency — raw obs → model-ready inputs (resize/normalize/tokenize/CPU→GPU)
  3. Vision latency    — vision encoder + projector
  4. LLM latency       — VLM backbone (language_model forward)
  5. Action latency    — action head (denoising loop) + action unnormalize

Definitions:
  Model Latency  = Vision + LLM + Action
  End-to-End     = Data + Preprocess + Model Latency

Usage:
  # Real GR00T model (requires HF access / local checkpoint)
  python tools/benchmark_gr00t_cache.py \
      --model-path nvidia/GR00T-N1.5-3B \
      --data-config examples.Libero.custom_data_config:LiberoDataConfig \
      --embodiment-tag new_embodiment \
      --denoising-steps 8 \
      --warmup 10 --iters 100

  # With cache enabled
  python tools/benchmark_gr00t_cache.py \
      --model-path nvidia/GR00T-N1.5-3B \
      --cache-mode full_cache --max-reuse-ratio 0.5 --task-topk 5

  # Dummy model (for testing the pipeline without real weights)
  python tools/benchmark_gr00t_cache.py --dummy
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# ── Add repo root ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ── Helpers ────────────────────────────────────────────────────────────

def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cpu_now_ms() -> float:
    return time.perf_counter() * 1000.0


def gpu_event_timer():
    """Context manager that returns CUDA-event-based elapsed_ms."""
    if not torch.cuda.is_available():
        return _CpuTimer()
    return _GpuEventTimer()


class _CpuTimer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0


class _GpuEventTimer:
    def __enter__(self):
        self.start_ev = torch.cuda.Event(enable_timing=True)
        self.end_ev = torch.cuda.Event(enable_timing=True)
        self.start_ev.record()
        return self

    def __exit__(self, *args):
        self.end_ev.record()
        torch.cuda.synchronize()
        self.elapsed_ms = self.start_ev.elapsed_time(self.end_ev)


def summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    a = np.array(values, dtype=np.float64)
    return {
        "mean_ms": float(a.mean()),
        "std_ms": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "p50_ms": float(np.median(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


# ── 5-stage timing wrapper around Gr00tPolicy ──────────────────────────

class Gr00tPolicyBenchmark:
    """Wraps Gr00tPolicy to extract the 5-stage latency breakdown."""

    def __init__(self, policy):
        self.policy = policy

    def step(self, observations: dict) -> dict[str, float]:
        """Run one policy step and return 5-stage + derived latencies (ms)."""
        policy = self.policy
        cuda_sync()
        t_total_start = time.perf_counter()

        # --- Stage 1: Data ---
        # In offline benchmark, data is already in memory → 0
        # In online setting, this is the time to receive observation from env
        data_ms = 0.0
        t_after_data = time.perf_counter()

        # --- Stage 2: Preprocess ---
        obs_copy = observations.copy()
        is_batch = policy._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = _unsqueeze_dict(obs_copy)
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        with gpu_event_timer() as preprocess_timer:
            normalized_input = policy.apply_transforms(obs_copy)

        preprocess_ms = preprocess_timer.elapsed_ms

        # --- Stage 3+4+5: Model (Vision + LLM + Action) ---
        with torch.inference_mode():
            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, model_timings = policy._get_action_from_normalized_input_profiled(
                        normalized_input
                    )
            else:
                _, model_timings = policy._get_action_from_normalized_input_profiled(
                    normalized_input
                )

        # Extract internal timings
        vision_ms = model_timings.get("server_system2_vision_ms", 0.0)
        llm_ms = model_timings.get("server_system2_reasoning_ms", 0.0)
        action_head_ms = model_timings.get("server_system1_action_head_ms", 0.0)
        postprocess_ms = model_timings.get("server_postprocess_untransform_ms", 0.0)

        # Action latency = action head (denoising loop) + unnormalize
        action_ms = action_head_ms + postprocess_ms

        # Derived
        model_ms = vision_ms + llm_ms + action_ms
        e2e_ms = data_ms + preprocess_ms + model_ms

        return {
            "data_ms": data_ms,
            "preprocess_ms": preprocess_ms,
            "vision_ms": vision_ms,
            "llm_ms": llm_ms,
            "action_ms": action_ms,
            "model_ms": model_ms,
            "e2e_ms": e2e_ms,
            # Raw internal timings for debugging
            "_vision_model_ms": model_timings.get("server_system2_vision_model_ms", 0.0),
            "_vision_projector_ms": model_timings.get("server_system2_vision_projector_ms", 0.0),
            "_action_head_ms": action_head_ms,
            "_postprocess_ms": postprocess_ms,
            "_prepare_input_ms": model_timings.get("server_model_prepare_input_to_device_ms", 0.0),
            "_backbone_ms": model_timings.get("server_system2_backbone_ms", 0.0),
        }


def _unsqueeze_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0)
        else:
            out[k] = v
    return out


# ── Main benchmark ─────────────────────────────────────────────────────

def run_benchmark(
    policy,
    observations: list[dict],
    warmup: int = 10,
    iters: int = 100,
    label: str = "gr00t",
    cache_config: Optional[dict] = None,
) -> dict[str, Any]:
    """Run latency benchmark with 5-stage breakdown."""
    bench = Gr00tPolicyBenchmark(policy)

    # Warmup
    print(f"  Warming up ({warmup} steps)...")
    for i in range(warmup):
        obs = observations[i % len(observations)]
        bench.step(obs)

    # Measurement
    print(f"  Measuring ({iters} steps)...")
    all_timings: dict[str, list[float]] = defaultdict(list)

    for i in range(iters):
        obs = observations[i % len(observations)]
        t = bench.step(obs)
        for k, v in t.items():
            all_timings[k].append(v)

    # Summarize
    result: dict[str, Any] = {
        "label": label,
        "cache_config": cache_config,
        "warmup": warmup,
        "iters": iters,
    }
    for key, values in all_timings.items():
        if key.startswith("_"):
            continue
        result[key] = summary_stats(values)

    # Print
    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  {'─'*60}")
    stage_labels = [
        ("data_ms", "1. Data"),
        ("preprocess_ms", "2. Preprocess"),
        ("vision_ms", "3. Vision"),
        ("llm_ms", "4. LLM"),
        ("action_ms", "5. Action"),
        ("model_ms", "   Model (3+4+5)"),
        ("e2e_ms", "   End-to-End (1-5)"),
    ]
    for key, name in stage_labels:
        s = result.get(key, {})
        if s:
            print(f"  {name:25s}  mean={s['mean_ms']:8.2f}  p50={s['p50_ms']:8.2f}  "
                  f"p95={s['p95_ms']:8.2f}  std={s['std_ms']:6.2f} ms")

    hz = 1000.0 / result["e2e_ms"]["mean_ms"] if result["e2e_ms"]["mean_ms"] > 0 else 0
    print(f"  {'Control Frequency':25s}  {hz:.1f} Hz")
    print()

    return result


# ── FLOPs estimation (from real model config) ──────────────────────────

def estimate_real_gr00t_flops(model, reuse_ratio: float = 0.0) -> dict:
    """Estimate theoretical TFLOPs from the real GR00T model's architecture."""
    from gr00t_cache.flops_estimator import estimate_cache_transformer_flops

    backbone = model.backbone
    eagle = backbone.eagle_model
    lm = eagle.language_model

    # Backbone LLM
    backbone_layers = len(lm.model.layers)
    backbone_d_model = lm.config.hidden_size
    # FFN dim: for Qwen2/Llama it's config.intermediate_size
    backbone_ffn = getattr(lm.config, "intermediate_size", backbone_d_model * 4)

    # Token counts (estimated from typical Libero scenario)
    # 2 views × dynamic tiles: roughly 512-1024 visual tokens
    # Text: ~50-100 tokens
    text_tokens = 80
    visual_tokens = 768  # conservative estimate for 2-camera Libero

    reuse_ratios = [reuse_ratio] * backbone_layers

    flops = estimate_cache_transformer_flops(
        num_layers=backbone_layers,
        text_tokens=text_tokens,
        visual_tokens=visual_tokens,
        d_model=backbone_d_model,
        ffn_dim=backbone_ffn,
        reuse_ratios_by_layer=reuse_ratios,
    )

    # DiT
    action_head = model.action_head
    dit = action_head.model
    dit_layers = len(dit.transformer_blocks)
    dit_d_model = dit.inner_dim
    dit_ffn = getattr(dit.config, "ff_inner_dim",
                      getattr(dit, "inner_dim", dit_d_model) * 4)
    dit_query = 1 + 32 + action_head.action_horizon  # state + future + actions
    dit_condition = text_tokens + visual_tokens

    from gr00t_cache.flops_estimator import FLOPSEstimate
    dit_full = 0.0
    for _ in range(dit_layers):
        from gr00t_cache.flops_estimator import transformer_layer_flops
        # Self-attn on query
        dit_full += transformer_layer_flops(dit_query, dit_d_model, dit_ffn)
        # Cross-attn on condition
        dit_full += transformer_layer_flops(dit_condition, dit_d_model, dit_ffn)

    # Per-step: action head runs num_inference_timesteps times
    num_steps = action_head.num_inference_timesteps
    dit_full *= num_steps

    dit_flops = {
        "full_gflops": dit_full / 1e9,
        "full_tflops": dit_full / 1e12,
        "num_layers": dit_layers,
        "num_steps": num_steps,
        "query_tokens": dit_query,
        "condition_tokens": dit_condition,
    }

    return {
        "backbone": {
            "num_layers": backbone_layers,
            "d_model": backbone_d_model,
            "ffn_dim": backbone_ffn,
            "text_tokens": text_tokens,
            "visual_tokens": visual_tokens,
            "reuse_ratio": reuse_ratio,
            **flops.to_dict(),
        },
        "dit": dit_flops,
        "total_gflops": flops.cached_flops / 1e9 + dit_full / 1e9,
        "total_tflops": flops.cached_tflops + dit_full / 1e12,
    }


# ── CSV / JSON output ──────────────────────────────────────────────────

def save_results(results: list[dict], output_dir: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # CSV
    csv_path = output_dir / "measurements.csv"
    rows = []
    for r in results:
        row = {"label": r["label"]}
        for stage in ["data_ms", "preprocess_ms", "vision_ms", "llm_ms", "action_ms",
                       "model_ms", "e2e_ms"]:
            s = r.get(stage, {})
            row[f"{stage}_mean"] = s.get("mean_ms", "")
            row[f"{stage}_p50"] = s.get("p50_ms", "")
            row[f"{stage}_p95"] = s.get("p95_ms", "")
            row[f"{stage}_std"] = s.get("std_ms", "")
        rows.append(row)

    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    print(f"Results saved to {output_dir}/")


# ── Observation sampling ───────────────────────────────────────────────

def sample_libero_observations(n: int, seed: int = 0) -> list[dict]:
    """Generate synthetic Libero-style observations for benchmarking.

    Uses 256x256 resolution matching the LiberoDataConfig expectation.
    """
    rng = np.random.default_rng(seed)
    observations = []
    for i in range(n):
        img = rng.integers(0, 256, size=(1, 256, 256, 3), dtype=np.uint8)
        wrist = rng.integers(0, 256, size=(1, 256, 256, 3), dtype=np.uint8)
        obs = {
            "video.image": img,
            "video.wrist_image": wrist,
            "state.x": np.array([[0.0]], dtype=np.float64),
            "state.y": np.array([[0.0]], dtype=np.float64),
            "state.z": np.array([[0.0]], dtype=np.float64),
            "state.roll": np.array([[0.0]], dtype=np.float64),
            "state.pitch": np.array([[0.0]], dtype=np.float64),
            "state.yaw": np.array([[0.0]], dtype=np.float64),
            "state.gripper": np.array([[0.0, 0.0]], dtype=np.float64),
            "annotation.human.action.task_description": ["put the black bowl on the plate"],
        }
        observations.append(obs)
    return observations


# ── Real GR00T path ────────────────────────────────────────────────────

def load_real_policy(args):
    """Load the real GR00T policy."""
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    print(f"Loading GR00T model from: {args.model_path}")
    data_config = load_data_config(args.data_config)
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )
    if args.denoising_steps is not None:
        policy.model.action_head.num_inference_timesteps = args.denoising_steps

    print(f"  Backbone layers: {len(policy.model.backbone.eagle_model.language_model.model.layers)}")
    print(f"  DiT layers: {len(policy.model.action_head.model.transformer_blocks)}")
    print(f"  Action horizon: {policy.model.action_head.action_horizon}")
    print(f"  Denoising steps: {policy.model.action_head.num_inference_timesteps}")
    return policy


def load_dummy_policy(args):
    """Load the dummy GR00T policy (for testing without real weights)."""
    from gr00t_cache.dummy_model import create_dummy_gr00t_model, create_dummy_observation

    print("Loading DUMMY GR00T model (no real weights)")
    model = create_dummy_gr00t_model(
        device="cuda" if torch.cuda.is_available() else "cpu",
        backbone_num_layers=getattr(args, "backbone_layers", 6),
        dit_num_layers=getattr(args, "dit_layers", 4),
        action_horizon=getattr(args, "action_horizon", 16),
        num_inference_timesteps=args.denoising_steps,
    )

    # Wrap in a minimal policy-like object
    class DummyPolicy:
        def __init__(self, model):
            self.model = model
            self._is_batch = False
            self._modality_transform = None

        def _check_state_is_batched(self, obs):
            return False

        def apply_transforms(self, obs):
            # No-op for dummy
            return obs

        def _get_action_from_normalized_input_profiled(self, normalized_input):
            cuda_sync()
            t0 = time.perf_counter()

            # Vision
            with gpu_event_timer() as vt:
                pixel_values = torch.from_numpy(normalized_input["pixel_values"]).to(
                    device=model.device, dtype=model.dtype
                )
                input_ids = torch.from_numpy(normalized_input["input_ids"]).to(
                    device=model.device, dtype=torch.long
                )
                state = torch.from_numpy(normalized_input["state"]).to(
                    device=model.device, dtype=model.dtype
                )
                backbone_out = model.backbone(pixel_values, input_ids)
            vision_ms = vt.elapsed_ms

            # LLM: estimated as backbone minus vision (simplified for dummy)
            # In dummy model, backbone includes vision+LLM in one go
            llm_ms = vision_ms * 0.1  # rough split
            vision_ms = vision_ms * 0.9

            # Action
            with gpu_event_timer() as at:
                cfg = model.config
                B = state.shape[0]
                cond = backbone_out["backbone_features"]
                actions = torch.randn(B, cfg.action_horizon, cfg.action_dim,
                                      device=model.device, dtype=model.dtype)
                dt = 1.0 / cfg.num_inference_timesteps
                for t in range(cfg.num_inference_timesteps):
                    t_disc = int(t / cfg.num_inference_timesteps * 999)
                    ts = torch.full((B,), t_disc, device=model.device, dtype=torch.long)
                    pred_vel = model.action_head(cond, actions, state, ts)
                    actions = actions + dt * pred_vel
            action_ms = at.elapsed_ms

            cuda_sync()
            total_ms = (time.perf_counter() - t0) * 1000.0

            timings = {
                "server_system2_vision_ms": vision_ms,
                "server_system2_vision_model_ms": vision_ms * 0.7,
                "server_system2_vision_projector_ms": vision_ms * 0.3,
                "server_system2_reasoning_ms": llm_ms,
                "server_system1_action_head_ms": action_ms,
                "server_postprocess_untransform_ms": 0.1,
                "server_system2_backbone_ms": vision_ms + llm_ms,
                "server_model_prepare_input_to_device_ms": 0.5,
                "server_model_total_ms": total_ms,
            }
            return actions, timings

        def _get_unnormalized_action(self, action):
            return {"action": action.float().cpu().numpy()}

    return DummyPolicy(model)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GR00T-Cache 5-Stage Latency Benchmark")

    # Model
    parser.add_argument("--model-path", type=str, default="nvidia/GR00T-N1.5-3B")
    parser.add_argument("--data-config", type=str,
                        default="examples.Libero.custom_data_config:LiberoDataConfig")
    parser.add_argument("--embodiment-tag", type=str, default="new_embodiment")
    parser.add_argument("--denoising-steps", type=int, default=8)

    # Dummy mode
    parser.add_argument("--dummy", action="store_true",
                        help="Use dummy model instead of real GR00T")

    # Cache
    parser.add_argument("--cache-mode", type=str, default="none",
                        choices=["none", "full_cache",
                                  "backbone_visual_kv_cache",
                                  "action_head_condition_kv_cache"])
    parser.add_argument("--max-reuse-ratio", type=float, default=0.5)
    parser.add_argument("--task-topk", type=int, default=None)
    parser.add_argument("--entropy-scale", type=float, default=1.0)

    # Benchmark
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")

    # Output
    parser.add_argument("--output-dir", type=str, default="./gr00t_cache_results")
    parser.add_argument("--save-csv", action="store_true", default=True)
    parser.add_argument("--save-json", action="store_true", default=True)

    args = parser.parse_args()

    print("=" * 70)
    print("  GR00T-Cache: 5-Stage Latency Benchmark")
    print(f"  Model: {'DUMMY' if args.dummy else args.model_path}")
    print(f"  Cache mode: {args.cache_mode}")
    print(f"  Warmup: {args.warmup}, Iters: {args.iters}")
    print("=" * 70)

    # Load model
    if args.dummy:
        policy = load_dummy_policy(args)
        observations = []
        for i in range(args.warmup + args.iters):
            from gr00t_cache.dummy_model import create_dummy_observation
            obs = create_dummy_observation(batch_size=1, seed=i)
            observations.append(obs)
    else:
        policy = load_real_policy(args)
        observations = sample_libero_observations(args.warmup + args.iters, seed=42)

    # Compute FLOPs estimate
    if not args.dummy:
        print("\n  FLOPs Estimation (theoretical):")
        flops = estimate_real_gr00t_flops(policy.model, reuse_ratio=(
            args.max_reuse_ratio if args.cache_mode != "none" else 0.0
        ))
        print(f"    Backbone: {flops['backbone']['num_layers']} layers, "
              f"d={flops['backbone']['d_model']}, ffn={flops['backbone']['ffn_dim']}")
        print(f"    Full: {flops['backbone']['full_tflops']:.4f} TFLOPs")
        if args.cache_mode != "none":
            print(f"    Cached: {flops['backbone']['cached_tflops']:.4f} TFLOPs "
                  f"({flops['backbone']['flops_reduction_percent']:.1f}% reduction)")
        print(f"    DiT: {flops['dit']['full_tflops']:.4f} TFLOPs "
              f"({flops['dit']['num_layers']} layers × {flops['dit']['num_steps']} steps)")
    else:
        flops = None

    # ── Baseline (cache disabled) ──
    print(f"\n{'='*70}")
    print("  BASELINE (cache disabled)")
    print(f"{'='*70}")
    result_baseline = run_benchmark(
        policy, observations,
        warmup=args.warmup, iters=args.iters,
        label="baseline",
    )

    # ── Cached (if requested) ──
    results = [result_baseline]

    if args.cache_mode != "none":
        print(f"{'='*70}")
        print(f"  CACHED (mode={args.cache_mode}, max_reuse={args.max_reuse_ratio})")
        print(f"{'='*70}")

        # Apply cache wrappers to the model
        from gr00t_cache.config import GR00TCacheConfig, CacheMode
        from gr00t_cache.cache_manager import GR00TCacheManager
        from gr00t_cache.token_index_map import TokenIndexMap
        from gr00t_cache.attention_wrapper import (
            apply_cache_to_backbone, apply_cache_to_action_head,
            remove_cache_from_model,
        )

        cache_cfg = GR00TCacheConfig(
            enabled=True,
            cache_mode=CacheMode(args.cache_mode),
            max_reuse_ratio=args.max_reuse_ratio,
            task_topk=args.task_topk,
            entropy_scale=args.entropy_scale,
            debug=False,
        )
        cache_mgr = GR00TCacheManager(cache_cfg)
        token_map = TokenIndexMap(n_visual=768, n_total=848)

        bb_wrappers = apply_cache_to_backbone(policy.model, cache_mgr, token_map, cache_cfg)
        ah_wrappers = apply_cache_to_action_head(policy.model, cache_mgr, cache_cfg)

        print(f"  Wrapped {len(bb_wrappers)} backbone layers, {len(ah_wrappers)} DiT blocks")

        result_cached = run_benchmark(
            policy, observations,
            warmup=args.warmup, iters=args.iters,
            label=f"cached_{args.cache_mode}",
            cache_config=cache_cfg.to_dict(),
        )

        # Cleanup
        remove_cache_from_model(policy.model, bb_wrappers)
        remove_cache_from_model(policy.model, ah_wrappers)

        results.append(result_cached)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  COMPARISON TABLE")
    print(f"{'='*80}")
    header = (f"{'':25s} {'Data':>8s} {'Preproc':>8s} {'Vision':>8s} "
              f"{'LLM':>8s} {'Action':>8s} {'Model':>8s} {'E2E':>8s} {'Hz':>7s}")
    print(header)
    print("-" * 80)

    for r in results:
        d = r.get("data_ms", {}).get("mean_ms", 0)
        p = r.get("preprocess_ms", {}).get("mean_ms", 0)
        v = r.get("vision_ms", {}).get("mean_ms", 0)
        l = r.get("llm_ms", {}).get("mean_ms", 0)
        a = r.get("action_ms", {}).get("mean_ms", 0)
        m = r.get("model_ms", {}).get("mean_ms", 0)
        e = r.get("e2e_ms", {}).get("mean_ms", 0)
        hz = 1000.0 / e if e > 0 else 0
        print(f"{r['label']:25s} {d:8.2f} {p:8.2f} {v:8.2f} {l:8.2f} {a:8.2f} {m:8.2f} {e:8.2f} {hz:6.1f}")

    print("-" * 80)

    if len(results) >= 2:
        b = results[0]
        c = results[1]
        e2e_b = b.get("e2e_ms", {}).get("mean_ms", 1)
        e2e_c = c.get("e2e_ms", {}).get("mean_ms", 1)
        speedup = e2e_b / e2e_c if e2e_c > 0 else 1.0
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  E2E reduction: {(1 - e2e_c/e2e_b)*100:.1f}%")

    # Save
    if args.save_csv or args.save_json:
        save_results(results, args.output_dir)
        if flops:
            with open(Path(args.output_dir) / "flops.json", "w") as f:
                json.dump(flops, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
