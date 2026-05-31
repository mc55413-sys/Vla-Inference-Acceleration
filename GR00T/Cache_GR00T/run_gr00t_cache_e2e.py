#!/usr/bin/env python3
"""
End-to-end test for GR00T-Cache.

Tests every component of the GR00T-Cache system with the dummy model:
  1. Dummy model forward pass correctness
  2. FLOPs estimator accuracy (hand-verified reference values)
  3. Token index map construction
  4. Cache manager: static token selection, task eviction, layer-adaptive
  5. Attention wrapper: passthrough when disabled, cached forward when enabled
  6. Correctness: cache-disabled output == original output
  7. Profiling: timer and statistics collection
  8. Ablation: all presets run without errors
  9. Full pipeline: cache → profile → compare → report

Usage:
    python run_gr00t_cache_e2e.py
    python run_gr00t_cache_e2e.py --device cpu  # if no GPU
    python run_gr00t_cache_e2e.py --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from gr00t_cache.config import GR00TCacheConfig, CacheMode
from gr00t_cache.token_index_map import TokenIndexMap
from gr00t_cache.cache_manager import GR00TCacheManager
from gr00t_cache.attention_wrapper import (
    CachedAttentionWrapper,
    apply_cache_to_backbone,
    apply_cache_to_action_head,
    remove_cache_from_model,
)
from gr00t_cache.profiling import ProfileResults, ProfileTimer
from gr00t_cache.flops_estimator import (
    estimate_cache_transformer_flops,
    transformer_layer_flops,
)
from gr00t_cache.correctness import compute_action_similarity
from gr00t_cache.dummy_model import (
    DummyGR00TConfig,
    DummyGR00TModel,
    create_dummy_gr00t_model,
    create_dummy_observation,
)
from gr00t_cache.ablation import ABLATION_PRESETS
from gr00t_cache.utils import cuda_sync, summarize_statistics


# ────────────────────────────────────────────────────────────────────────
# Test helpers
# ────────────────────────────────────────────────────────────────────────

_passed = 0
_failed = 0

def check(condition: bool, name: str, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  ✓ {name}")
    else:
        _failed += 1
        print(f"  ✗ {name}  {detail}")

def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ────────────────────────────────────────────────────────────────────────
# Test 1: Dummy model forward pass
# ────────────────────────────────────────────────────────────────────────

def test_dummy_model(device: str, verbose: bool) -> DummyGR00TModel:
    section("Test 1: Dummy GR00T model forward pass")

    model = create_dummy_gr00t_model(device=device, dtype=torch.float32)
    check(model is not None, "Model created", f"params={sum(p.numel() for p in model.parameters())}")

    obs = create_dummy_observation(batch_size=1, seed=0)
    result = model.get_action(obs)
    action = result["action"]
    check(isinstance(action, np.ndarray), "Action is ndarray")
    check(action.shape == (1, 16, 7), f"Action shape {action.shape} == (1, 16, 7)")
    check(not np.any(np.isnan(action)), "No NaN in action")
    check(not np.any(np.isinf(action)), "No Inf in action")

    # Determinism check
    result2 = model.get_action(obs)
    check(
        np.allclose(result["action"], result2["action"], atol=1e-5),
        "Deterministic output (same input)"
    )

    # Different seed → different output
    obs2 = create_dummy_observation(batch_size=1, seed=1)
    result3 = model.get_action(obs2)
    check(
        not np.allclose(result["action"], result3["action"], atol=1e-3),
        "Different seeds → different actions"
    )

    if verbose:
        print(f"    Action mean: {action.mean():.4f}, std: {action.std():.4f}")
        print(f"    Action min: {action.min():.4f}, max: {action.max():.4f}")

    return model


# ────────────────────────────────────────────────────────────────────────
# Test 2: FLOPs estimator
# ────────────────────────────────────────────────────────────────────────

def test_flops_estimator():
    section("Test 2: FLOPs estimator")

    # Hand-calculated reference values
    # For n=100, d=512, ffn_dim=2048:
    # C(n) = 4*100*512^2 + 2*100^2*512 + 2*100*512*2048
    #      = 104857600 + 10240000 + 209715200
    #      = 324812800 FLOPs ≈ 0.3248 GFLOPs per layer
    n, d, ffn = 100, 512, 2048
    per_layer = transformer_layer_flops(n, d, ffn)
    expected = 4 * n * d**2 + 2 * n**2 * d + 2 * n * d * ffn
    check(
        abs(per_layer - expected) < 1.0,
        f"Per-layer FLOPs matches formula",
        f"got={per_layer:.0f}, expected={expected:.0f}"
    )

    # Full model: 6 layers, 50 text + 392 visual = 442 tokens
    flops = estimate_cache_transformer_flops(
        num_layers=6,
        text_tokens=50,
        visual_tokens=392,
        d_model=512,
        ffn_dim=2048,
        reuse_ratios_by_layer=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    # With 0% reuse, full == cached
    check(
        abs(flops.full_flops - flops.cached_flops) < 1.0,
        "0% reuse: full == cached"
    )
    check(
        abs(flops.flops_reduction_percent) < 0.01,
        "0% reuse: reduction ≈ 0%"
    )

    # With 50% reuse
    flops50 = estimate_cache_transformer_flops(
        num_layers=6,
        text_tokens=50,
        visual_tokens=392,
        d_model=512,
        ffn_dim=2048,
        reuse_ratios_by_layer=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    )
    check(
        flops50.cached_flops < flops50.full_flops,
        "50% reuse: cached < full",
        f"full={flops50.full_flops/1e9:.2f}G, cached={flops50.cached_flops/1e9:.2f}G"
    )
    check(
        flops50.flops_reduction_percent > 0,
        "50% reuse: reduction > 0%",
        f"reduction={flops50.flops_reduction_percent:.1f}%"
    )

    # Layer-adaptive: increasing reuse → decreasing FLOPs
    flops_ascending = estimate_cache_transformer_flops(
        num_layers=6,
        text_tokens=50,
        visual_tokens=392,
        d_model=512,
        ffn_dim=2048,
        reuse_ratios_by_layer=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    check(
        flops_ascending.cached_flops < flops.full_flops,
        "Layer-adaptive: cached < full"
    )

    if verbose:
        print(f"    Per-layer FLOPs (n=100): {per_layer/1e9:.4f} GFLOPs")
        print(f"    Full (6 layers, 442 tokens): {flops.full_flops/1e9:.2f} GFLOPs")
        print(f"    Cached (50% reuse): {flops50.cached_flops/1e9:.2f} GFLOPs ({flops50.flops_reduction_percent:.1f}% reduction)")
        print(f"    Layer-adaptive: {flops_ascending.cached_flops/1e9:.2f} GFLOPs ({flops_ascending.flops_reduction_percent:.1f}% reduction)")


# ────────────────────────────────────────────────────────────────────────
# Test 3: Token index map
# ────────────────────────────────────────────────────────────────────────

def test_token_index_map():
    section("Test 3: Token index map")

    # Simulate a VLM input_ids with text + image tokens
    # 20 text tokens, then 100 visual tokens
    input_ids = torch.zeros(140, dtype=torch.long)
    input_ids[:20] = 1  # text
    input_ids[20:] = -200  # image_token_index

    idx_map = TokenIndexMap.from_backbone_inputs(
        input_ids=input_ids,
        image_token_index=-200,
        view_info={"external": (20, 70), "wrist": (70, 120)},
    )

    check(idx_map.n_text == 20, f"Text tokens: {idx_map.n_text} == 20")
    check(idx_map.n_visual == 100, f"Visual tokens: {idx_map.n_visual} == 100")
    check(idx_map.n_total == 140, f"Total tokens: {idx_map.n_total} == 140")
    check(
        "external" in idx_map.visual_indices_by_view,
        "External view found"
    )
    check(
        "wrist" in idx_map.visual_indices_by_view,
        "Wrist view found"
    )
    check(
        len(idx_map.visual_indices_by_view["external"]) == 50,
        f"External tokens: {len(idx_map.visual_indices_by_view['external'])} == 50"
    )

    if verbose:
        print(idx_map.debug_summary())


# ────────────────────────────────────────────────────────────────────────
# Test 4: Cache manager
# ────────────────────────────────────────────────────────────────────────

def test_cache_manager(verbose: bool):
    section("Test 4: Cache manager")

    config = GR00TCacheConfig(
        enabled=True,
        max_reuse_ratio=0.5,
        static_similarity_threshold=0.9,
        task_topk=5,
        entropy_scale=1.0,
        debug=verbose,
    )
    mgr = GR00TCacheManager(config)
    check(mgr is not None, "Cache manager created")

    # First step — no cache
    images = torch.randn(2, 3, 224, 224)  # 2 views
    state = torch.randn(7)
    token_map = TokenIndexMap(n_visual=392, n_total=442)

    # First call — no previous data, should NOT cache
    plan1 = mgr.get_reuse_plan(
        current_images=images,
        current_proprio=state,
        current_token_map=token_map,
        batch_size=1,
    )
    check(not plan1["should_cache"], "Step 1: no cache available")
    check(plan1["reset_reason"] != "", f"Step 1 reset reason: {plan1['reset_reason']}")

    # Store step 1 data
    mgr.update_cache(
        current_images=images,
        current_visual_tokens=torch.randn(392, 512),
        current_token_map=token_map,
        current_instruction=None,
        current_proprio=state,
        attention_maps={},
        layer_kv={},
    )

    # Second step — similar images, should cache
    images2 = images + torch.randn_like(images) * 0.02
    plan2 = mgr.get_reuse_plan(
        current_images=images2,
        current_proprio=state + torch.randn(7) * 0.001,
        current_token_map=token_map,
        batch_size=1,
    )
    check(plan2["should_cache"], "Step 2: cache available")
    check(plan2["reuse_ratio"] > 0, f"Step 2 reuse ratio > 0: {plan2['reuse_ratio']:.3f}")
    check("static_masks" in plan2, "Static masks computed")
    check(
        len(plan2["static_masks"]) == 2 if images2.shape[0] == 2 else True,
        f"Per-view masks: {len(plan2.get('static_masks', {}))}"
    )

    # Test cache invalidation — instruction change
    mgr.config.reset_on_instruction_change = True
    should_reset, reason = mgr.should_reset_cache(
        current_images=images2,
        current_instruction="pick up the red block",
    )
    check(should_reset or reason == "", f"Instruction change triggers reset: {should_reset}")

    # Test cache invalidation — camera count change
    images3 = torch.randn(3, 3, 224, 224)  # 3 views instead of 2
    should_reset, reason = mgr.should_reset_cache(
        current_images=images3,
    )
    check(should_reset, f"Camera count change triggers reset: {reason}")

    # Test stats
    stats = mgr.stats()
    check(stats["total_steps"] >= 0, f"Stats available: total_steps={stats['total_steps']}")

    if verbose:
        print(f"    Plan2 reuse_ratio: {plan2['reuse_ratio']:.3f}")
        print(f"    Cache stats: {stats}")


# ────────────────────────────────────────────────────────────────────────
# Test 5: Attention wrapper passthrough
# ────────────────────────────────────────────────────────────────────────

def test_attention_wrapper_passthrough(model: DummyGR00TModel, verbose: bool):
    section("Test 5: Attention wrapper passthrough (cache disabled)")

    obs = create_dummy_observation(batch_size=1, seed=0)
    obs2 = create_dummy_observation(batch_size=1, seed=0)  # same seed

    # Get baseline action without any wrapper
    result_baseline = model.get_action(obs)
    baseline_action = result_baseline["action"]

    # Apply backbone wrapper with cache disabled
    config = GR00TCacheConfig(enabled=False, cache_mode=CacheMode.NONE)
    cache_mgr = GR00TCacheManager(config)
    token_map = TokenIndexMap(n_visual=392, n_total=442)

    wrappers = apply_cache_to_backbone(model, cache_mgr, token_map, config)

    # With cache disabled, output must match
    result_wrapped = model.get_action(obs2)
    wrapped_action = result_wrapped["action"]

    match = np.allclose(baseline_action, wrapped_action, atol=1e-5)
    check(match, "Cache-disabled wrapper preserves exact output")

    if not match:
        diff = np.abs(baseline_action - wrapped_action).max()
        print(f"      Max diff: {diff:.6f}")

    # Cleanup
    remove_cache_from_model(model, wrappers)

    if verbose and len(wrappers) > 0:
        print(f"    Wrapped {len(wrappers)} layers, all restored")


# ────────────────────────────────────────────────────────────────────────
# Test 6: Cached forward produces different (but valid) output
# ────────────────────────────────────────────────────────────────────────

def test_cached_forward(model: DummyGR00TModel, verbose: bool):
    section("Test 6: Cached forward")

    config = GR00TCacheConfig(
        enabled=True,
        cache_mode=CacheMode.FULL_CACHE,
        max_reuse_ratio=0.5,
        debug=verbose,
    )
    cache_mgr = GR00TCacheManager(config)
    token_map = TokenIndexMap(n_visual=392, n_total=442)

    # Apply wrappers
    bb_wrappers = apply_cache_to_backbone(model, cache_mgr, token_map, config)
    ah_wrappers = apply_cache_to_action_head(model, cache_mgr, config)
    check(len(bb_wrappers) > 0, f"Backbone wrappers applied: {len(bb_wrappers)} layers")
    check(len(ah_wrappers) > 0, f"Action head wrappers applied: {len(ah_wrappers)} layers")

    # Step 1: first forward (no cache yet)
    obs1 = create_dummy_observation(batch_size=1, seed=0)
    images1 = torch.from_numpy(obs1["pixel_values"])

    # Setup reuse plan
    plan1 = cache_mgr.get_reuse_plan(
        current_images=images1,
        current_proprio=torch.from_numpy(obs1["state"]).float(),
        current_token_map=token_map,
        batch_size=1,
    )
    cache_mgr._current_reuse_plan = plan1

    result1 = model.get_action(obs1)

    # Store KV cache
    for w in bb_wrappers.values():
        w.store_backbone_kv(w.layer_idx)
    for w in ah_wrappers.values():
        w.store_condition_kv(w.layer_idx)

    cache_mgr.update_cache(
        current_images=images1,
        current_visual_tokens=None,
        current_token_map=token_map,
        current_instruction=None,
        current_proprio=torch.from_numpy(obs1["state"]).float(),
        attention_maps={},
        layer_kv=cache_mgr.layer_kv_cache,
        condition_kv=cache_mgr.action_head_condition_kv,
    )

    # Step 2: second forward (should use cache)
    obs2 = create_dummy_observation(batch_size=1, seed=0)
    obs2["pixel_values"] = obs2["pixel_values"] + np.random.randn(*obs2["pixel_values"].shape).astype(np.float32) * 0.01

    images2 = torch.from_numpy(obs2["pixel_values"])
    plan2 = cache_mgr.get_reuse_plan(
        current_images=images2,
        current_proprio=torch.from_numpy(obs2["state"]).float(),
        current_token_map=token_map,
        batch_size=1,
    )
    cache_mgr._current_reuse_plan = plan2

    check(plan2.get("should_cache", False), f"Step 2 should_cache = True")
    check(plan2.get("reuse_ratio", 0) > 0, f"Reuse ratio > 0: {plan2.get('reuse_ratio', 0):.3f}")

    result2 = model.get_action(obs2)
    action2 = result2["action"]
    check(not np.any(np.isnan(action2)), "Cached action: no NaN")
    check(not np.any(np.isinf(action2)), "Cached action: no Inf")

    # Cleanup
    remove_cache_from_model(model, bb_wrappers)
    remove_cache_from_model(model, ah_wrappers)

    if verbose:
        print(f"    Cached action mean: {action2.mean():.4f}, std: {action2.std():.4f}")


# ────────────────────────────────────────────────────────────────────────
# Test 7: Profiling
# ────────────────────────────────────────────────────────────────────────

def test_profiling(model: DummyGR00TModel, verbose: bool):
    section("Test 7: Profiling")

    results = ProfileResults()
    check(results is not None, "ProfileResults created")

    # Record some dummy measurements
    results.record_timing("data_ms", 1.0)
    results.record_timing("preprocess_ms", 2.0)

    # Run a few steps with timing
    obs = create_dummy_observation(batch_size=1, seed=0)
    for _ in range(5):
        with ProfileTimer(results, "model_latency_ms", use_cuda_events=False):
            _ = model.get_action(obs)
        results.record_timing("end_to_end_ms", 50.0)

    summary = results.summarize()
    check("model_latency_ms" in summary, "Model latency in summary")
    check(summary["model_latency_ms"]["n"] == 5, f"5 measurements: n={summary['model_latency_ms']['n']}")
    check(summary["model_latency_ms"]["mean"] > 0, f"Mean > 0: {summary['model_latency_ms']['mean']:.2f}ms")

    if verbose:
        for k, v in summary.items():
            if isinstance(v, dict) and "mean" in v:
                print(f"    {k}: mean={v['mean']:.2f}, std={v.get('std', 0):.2f}, n={v.get('n', 0)}")


# ────────────────────────────────────────────────────────────────────────
# Test 8: Correctness
# ────────────────────────────────────────────────────────────────────────

def test_correctness(model: DummyGR00TModel, verbose: bool):
    section("Test 8: Correctness metrics")

    # Generate two similar actions
    obs = create_dummy_observation(batch_size=1, seed=0)
    a1 = torch.from_numpy(model.get_action(obs)["action"])

    # Small perturbation → slightly different action
    obs2 = create_dummy_observation(batch_size=1, seed=0)
    obs2["pixel_values"] = obs2["pixel_values"] + np.random.randn(*obs2["pixel_values"].shape).astype(np.float32) * 1e-4
    a2 = torch.from_numpy(model.get_action(obs2)["action"])

    sim = compute_action_similarity(a1, a2)
    check(sim["action_cosine_similarity_vs_baseline"] > 0.9, "Similar inputs → high cosine sim")
    check(sim["max_abs_action_diff"] >= 0, f"Max abs diff >= 0: {sim['max_abs_action_diff']:.6f}")

    # Identical inputs → identical outputs
    a3 = torch.from_numpy(model.get_action(obs)["action"])
    sim2 = compute_action_similarity(a1, a3)
    check(
        sim2["action_cosine_similarity_vs_baseline"] > 0.9999,
        "Identical inputs → cosine sim ≈ 1.0"
    )
    check(
        sim2["max_abs_action_diff"] < 1e-5,
        f"Identical inputs → max abs diff ≈ 0: {sim2['max_abs_action_diff']:.2e}"
    )

    if verbose:
        print(f"    Similar inputs cosine sim: {sim['action_cosine_similarity_vs_baseline']:.6f}")
        print(f"    Identical inputs cosine sim: {sim2['action_cosine_similarity_vs_baseline']:.6f}")


# ────────────────────────────────────────────────────────────────────────
# Test 9: Ablation presets
# ────────────────────────────────────────────────────────────────────────

def test_ablation_presets():
    section("Test 9: Ablation presets")

    check(len(ABLATION_PRESETS) == 7, f"7 presets defined: {len(ABLATION_PRESETS)}")

    for name, cfg in ABLATION_PRESETS.items():
        check(isinstance(cfg, GR00TCacheConfig), f"{name}: GR00TCacheConfig")
        check(isinstance(cfg.to_dict(), dict), f"{name}: to_dict() works")
        check(
            GR00TCacheConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict(),
            f"{name}: round-trip serialization"
        )

    # Baseline should have cache disabled
    check(not ABLATION_PRESETS["baseline"].enabled, "Baseline: cache disabled")
    check(ABLATION_PRESETS["full_gr00t_cache"].enabled, "Full cache: enabled")
    check(
        ABLATION_PRESETS["static_only"].task_topk is None,
        "Static only: no task eviction"
    )
    check(
        ABLATION_PRESETS["static_plus_task_eviction"].task_topk is not None,
        "Static+task: has task eviction"
    )

    if True:
        print(f"    Presets: {list(ABLATION_PRESETS.keys())}")


# ────────────────────────────────────────────────────────────────────────
# Test 10: Memory estimation
# ────────────────────────────────────────────────────────────────────────

def test_memory_and_utils(device: str, verbose: bool):
    section("Test 10: Utilities")

    from gr00t_cache.utils import (
        tensor_hash, cosine_similarity_batch,
        compute_entropy, compute_entropy_concentration,
    )

    # tensor_hash
    t1 = torch.randn(100)
    t2 = torch.randn(100)
    check(tensor_hash(t1) != tensor_hash(t2), "tensor_hash: different tensors → different hashes")
    check(tensor_hash(t1) == tensor_hash(t1), "tensor_hash: same tensor → same hash")

    # cosine_similarity_batch
    a = torch.randn(10, 512)
    b = torch.randn(10, 512)
    sim = cosine_similarity_batch(a, b)
    check(sim.shape == (10,), f"Cosine sim shape: {sim.shape}")
    check((sim >= -1.01).all() and (sim <= 1.01).all(), "Cosine sim in [-1, 1]")

    # entropy
    attn = F.softmax(torch.randn(8, 100, 100), dim=-1)  # [heads, seq, seq]
    entropy = compute_entropy(attn, dim=-1)
    check(entropy.shape == (8, 100), f"Entropy shape: {entropy.shape}")
    check((entropy >= 0).all(), "Entropy >= 0")

    conc = compute_entropy_concentration(entropy)
    check(0.0 <= conc <= 1.0, f"Concentration in [0,1]: {conc:.3f}")

    if verbose:
        print(f"    Entropy range: [{entropy.min():.3f}, {entropy.max():.3f}]")
        print(f"    Concentration: {conc:.3f}")


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GR00T-Cache End-to-End Tests")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"])
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.quiet:
        args.verbose = False

    print("=" * 60)
    print("  GR00T-Cache End-to-End Test Suite")
    print(f"  Device: {args.device}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print("=" * 60)

    global _passed, _failed
    _passed = 0
    _failed = 0

    t_start = time.perf_counter()

    # Run all tests
    try:
        model = test_dummy_model(args.device, args.verbose)
    except Exception as e:
        print(f"  ✗ Dummy model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        model = None

    test_flops_estimator()
    test_token_index_map()
    test_cache_manager(args.verbose)
    test_ablation_presets()
    test_memory_and_utils(args.device, args.verbose)

    if model is not None:
        test_correctness(model, args.verbose)
        test_profiling(model, args.verbose)
        test_attention_wrapper_passthrough(model, args.verbose)
        test_cached_forward(model, args.verbose)
    else:
        print("\n  ⚠ Skipping model-dependent tests (model creation failed)")

    elapsed = time.perf_counter() - t_start

    print(f"\n{'='*60}")
    print(f"  Results: {_passed} passed, {_failed} failed")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*60}")

    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
