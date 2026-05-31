#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 GR00T-Cache Contributors
# SPDX-License-Identifier: Apache-2.0
"""
GR00T Inference Server with GR00T-Cache support.

Usage:
  # Baseline (no cache)
  python scripts/inference_service_cache.py --server \
      --model-path /path/to/model --embodiment-tag new_embodiment \
      --data-config examples.Libero.custom_data_config:LiberoDataConfig

  # With cache
  python scripts/inference_service_cache.py --server \
      --model-path /path/to/model --embodiment-tag new_embodiment \
      --data-config examples.Libero.custom_data_config:LiberoDataConfig \
      --cache-mode full_cache --max-reuse-ratio 0.5 --task-topk 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gr00t.eval.robot import RobotInferenceServer
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy

from gr00t_cache.config import GR00TCacheConfig, CacheMode
from gr00t_cache.cache_manager import GR00TCacheManager
from gr00t_cache.token_index_map import TokenIndexMap
from gr00t_cache.attention_wrapper import (
    apply_cache_to_backbone, apply_cache_to_action_head,
)


class CachedPolicyWrapper:
    """Wraps a Gr00tPolicy to add cross-step KV cache management.

    Intercepts get_action() and get_action_profiled() to:
    1. Compute reuse plan before inference
    2. Store K/V after inference for next step
    3. Expose cache statistics
    """

    def __init__(self, policy: Gr00tPolicy, cache_config: GR00TCacheConfig):
        self.policy = policy
        self.model = policy.model
        self.cache_config = cache_config
        self.cache_manager = GR00TCacheManager(cache_config)
        self._bb_wrappers = {}
        self._ah_wrappers = {}
        self._step_count = 0

        # Apply attention wrappers
        self._setup_cache_wrappers()

    def _setup_cache_wrappers(self):
        token_map = TokenIndexMap(n_visual=768, n_total=848)
        self._bb_wrappers = apply_cache_to_backbone(
            self.model, self.cache_manager, token_map, self.cache_config,
        )
        self._ah_wrappers = apply_cache_to_action_head(
            self.model, self.cache_manager, self.cache_config,
        )
        print(f"[GR00T-Cache] Wrapped {len(self._bb_wrappers)} backbone + "
              f"{len(self._ah_wrappers)} action-head attention layers")
        print(f"[GR00T-Cache] mode={self.cache_config.cache_mode.value}, "
              f"max_reuse={self.cache_config.max_reuse_ratio}, "
              f"task_topk={self.cache_config.task_topk}")

    # ── Public API (must match what RobotInferenceServer expects) ──

    def get_modality_config(self):
        return self.policy.get_modality_config()

    def get_action(self, observations: dict) -> dict:
        self._before_step(observations)
        result = self.policy.get_action(observations)
        self._after_step(observations)
        return result

    def get_action_profiled(self, observations: dict) -> dict:
        self._before_step(observations)
        result = self.policy.get_action_profiled(observations)
        self._after_step(observations)

        # Inject cache stats into timing
        plan = self.cache_manager._current_reuse_plan or {}
        if "__timing__" in result:
            result["__timing__"]["cache_reuse_ratio"] = plan.get("reuse_ratio", 0.0)
            result["__timing__"]["cache_hit"] = 1.0 if plan.get("should_cache") else 0.0
            result["__timing__"]["cache_age"] = float(plan.get("cache_age", 0))
        return result

    # ── Cache lifecycle ──

    def _before_step(self, observations: dict):
        current_images = self._extract_images(observations)
        current_proprio = self._extract_proprio(observations)
        token_map = TokenIndexMap(n_visual=768, n_total=848)

        reuse_plan = self.cache_manager.get_reuse_plan(
            current_images=current_images,
            current_proprio=current_proprio,
            current_token_map=token_map,
            batch_size=1,
        )
        self.cache_manager._current_reuse_plan = reuse_plan

    def _after_step(self, observations: dict):
        for w in self._bb_wrappers.values():
            w.store_backbone_kv(w.layer_idx)
        for w in self._ah_wrappers.values():
            w.store_condition_kv(w.layer_idx)

        current_images = self._extract_images(observations)
        current_proprio = self._extract_proprio(observations)
        token_map = TokenIndexMap(n_visual=768, n_total=848)

        self.cache_manager.update_cache(
            current_images=current_images,
            current_visual_tokens=None,
            current_token_map=token_map,
            current_instruction=None,
            current_proprio=current_proprio,
            attention_maps={},
            layer_kv=self.cache_manager.layer_kv_cache,
            condition_kv=self.cache_manager.action_head_condition_kv,
        )
        self._step_count += 1

    def reset_cache(self):
        self.cache_manager.reset()
        self._step_count = 0
        print("[GR00T-Cache] Cache reset")

    @property
    def cache_stats(self):
        return self.cache_manager.stats()

    # ── Helpers ──

    @staticmethod
    def _extract_images(observations: dict) -> torch.Tensor:
        imgs = []
        for key in sorted(observations.keys()):
            if "video" in key and "image" in key:
                v = observations[key]
                if isinstance(v, np.ndarray):
                    if v.ndim >= 3:
                        frame = v[-1] if v.ndim == 4 and v.shape[0] > 0 else v
                        if frame.shape[-1] == 3:
                            frame = frame.transpose(2, 0, 1)
                        t = torch.from_numpy(frame).float() / 255.0
                        imgs.append(t)
        if imgs:
            return torch.stack(imgs)
        return torch.zeros(2, 3, 256, 256)

    @staticmethod
    def _extract_proprio(observations: dict) -> Optional[torch.Tensor]:
        parts = []
        for key in sorted(observations.keys()):
            if "state" in key:
                v = observations[key]
                if isinstance(v, np.ndarray):
                    parts.append(torch.from_numpy(v).float().flatten())
                elif isinstance(v, torch.Tensor):
                    parts.append(v.float().flatten())
        if parts:
            return torch.cat(parts)
        return None


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GR00T Inference Server with GR00T-Cache support"
    )
    parser.add_argument("--server", action="store_true", default=True)
    parser.add_argument("--model-path", type=str,
                        default="/home/dell/.cache/huggingface/hub/models--youliangtan--gr00t-n1.5-libero-long-posttrain/snapshots/aa49078d5cc9ce72917bc4312f1ef12771f277de")
    parser.add_argument("--embodiment-tag", type=str, default="new_embodiment")
    parser.add_argument("--data-config", type=str,
                        default="examples.Libero.custom_data_config:LiberoDataConfig")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--denoising-steps", type=int, default=8)

    # Cache
    parser.add_argument("--cache-mode", type=str, default="none",
                        choices=["none", "full_cache", "backbone_visual_kv_cache",
                                  "action_head_condition_kv_cache"])
    parser.add_argument("--max-reuse-ratio", type=float, default=0.5)
    parser.add_argument("--task-topk", type=int, default=None)
    parser.add_argument("--entropy-scale", type=float, default=1.0)

    args = parser.parse_args()

    # Load policy
    print(f"Loading GR00T model: {args.model_path}")
    data_config = load_data_config(args.data_config)
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )

    # ── Fix dynamic tiling AFTER policy is loaded (BOTH baseline & cache) ──
    # Force Eagle processor to use exactly 1 tile per image (no dynamic tiling).
    # This stabilizes sequence length, enabling fair baseline-vs-cache comparison.
    transform = policy._modality_transform
    for t in getattr(transform, "transforms", []):
        if hasattr(t, "eagle_processor") and hasattr(t.eagle_processor, "image_processor"):
            ip = t.eagle_processor.image_processor
            ip.max_dynamic_tiles = 1
            ip.min_dynamic_tiles = 1
            ip.use_thumbnail = False
            print(f"[GR00T-Cache] Eagle processor fixed: "
                  f"max_dynamic_tiles=1, use_thumbnail=False "
                  f"(applied to both baseline & cache)")
            break

    # ── FLOPs estimation ──
    from gr00t_cache.model_flops import compute_gr00t_static_flops, format_flops_table
    flops_result = compute_gr00t_static_flops(
        policy.model,
        visual_tokens=512,   # 2 views × 256 tokens (fixed tiling, no thumbnail)
        text_tokens=80,
        num_denoising_steps=args.denoising_steps,
    )
    print(format_flops_table(flops_result))

    # ── Apply DiT block cache if requested ──
    if args.cache_mode != "none":
        from gr00t_cache.dit_cache import apply_dit_cache_to_action_head

        dit_cache_mgr = apply_dit_cache_to_action_head(
            policy.model.action_head,
            num_steps=args.denoising_steps,
            sim_threshold=0.99,
            warmup_steps=2,
            cache_self_attn=True,
            cache_cross_attn=True,
            early_blocks=4,  # first 4 blocks always computed
        )
        print(f"[GR00T-Cache] DiT block cache ENABLED: "
              f"warmup=2 steps, sim_threshold=0.99, early_blocks=4")
        print(f"[GR00T-Cache] Blocks 4-15 will cache outputs & reuse in later steps")

        # Also apply backbone attention wrappers for cross-timestep visual KV cache
        cache_config = GR00TCacheConfig(
            enabled=True,
            cache_mode=CacheMode(args.cache_mode),
            max_reuse_ratio=args.max_reuse_ratio,
            task_topk=args.task_topk,
            entropy_scale=args.entropy_scale,
        )
        server_policy = CachedPolicyWrapper(policy, cache_config)
        print(f"[GR00T-Cache] Backbone visual KV cache also ENABLED")
    else:
        server_policy = policy
        print("[GR00T-Cache] Server starting WITHOUT cache (baseline)")

    # Start server
    server = RobotInferenceServer(server_policy, port=args.port)
    server.run()


if __name__ == "__main__":
    main()
