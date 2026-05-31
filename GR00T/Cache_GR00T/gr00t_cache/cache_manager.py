# SPDX-FileCopyrightText: Copyright (c) 2025 GR00T-Cache Contributors
# SPDX-License-Identifier: Apache-2.0
"""
GR00TCacheManager: Core cache management for visual token KV reuse.

This module implements the VLA-Cache method adapted for GR00T:
1. Static token selection via patch similarity
2. Task-relevant token eviction via attention scores
3. Layer-adaptive token reuse via attention entropy
4. Cache invalidation for scene changes
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from .config import CacheMode, GR00TCacheConfig
from .token_index_map import TokenIndexMap
from .utils import (
    compute_entropy,
    compute_entropy_concentration,
    compute_patch_similarity,
    compute_proprio_delta,
    tensor_hash,
)


class GR00TCacheManager:
    """Manages cross-timestep KV cache reuse for GR00T models.

    Implements the VLA-Cache algorithm:
      1. Identify static visual tokens via patch similarity
      2. Evict task-relevant tokens using attention scores
      3. Apply layer-adaptive reuse ratios
      4. Invalidate cache on scene/instruction changes
    """

    def __init__(self, config: GR00TCacheConfig):
        self.config = config
        self._reset_internal()

    def _reset_internal(self) -> None:
        """Clear all internal cache state."""
        self.cache_step_id: int = -1
        self.cache_age: int = 0
        self.reset_count: int = 0
        self.total_steps: int = 0
        self.cache_hits: int = 0

        # Previous step data
        self.previous_images: Optional[torch.Tensor] = None  # [V, 3, H, W]
        self.previous_visual_tokens: Optional[torch.Tensor] = None  # [n_vis, D]
        self.previous_token_index_map: Optional[TokenIndexMap] = None
        self.previous_instruction_hash: Optional[str] = None
        self.previous_proprio_state: Optional[torch.Tensor] = None
        self.previous_condition_features: Optional[torch.Tensor] = None

        # Per-layer KV cache for backbone
        self.layer_kv_cache: dict[int, dict[str, torch.Tensor]] = {}
        # Per-layer attention maps (for task relevance)
        self.layer_attention_maps: dict[int, torch.Tensor] = {}

        # Action head condition KV cache (cross-attention)
        self.action_head_condition_kv: dict[int, dict[str, torch.Tensor]] = {}

        # Per-step reuse plan
        self._current_reuse_plan: Optional[dict] = None

    def reset(self) -> None:
        """Explicitly reset the cache."""
        self._reset_internal()

    def should_reset_cache(
        self,
        current_images: torch.Tensor,
        current_instruction: Optional[str] = None,
        current_proprio: Optional[torch.Tensor] = None,
        current_token_map: Optional[TokenIndexMap] = None,
        batch_size: int = 1,
    ) -> tuple[bool, str]:
        """Determine if the cache should be invalidated.

        Checks all invalidation conditions specified in the config.

        Returns:
            (should_reset, reason) tuple.
        """
        cfg = self.config
        reasons = []

        # Episode start / first step
        if self.cache_step_id < 0:
            reasons.append("first_step")

        # Cache age exceeded
        if self.cache_age > cfg.max_cache_age:
            reasons.append(f"cache_age_exceeded({self.cache_age}>{cfg.max_cache_age})")

        # Instruction changed
        if cfg.reset_on_instruction_change and current_instruction is not None:
            instr_hash = tensor_hash(
                torch.tensor(
                    [ord(c) for c in current_instruction[:1000]],
                    dtype=torch.long,
                )
            )
            if self.previous_instruction_hash is not None and instr_hash != self.previous_instruction_hash:
                reasons.append("instruction_changed")

        # Camera count changed
        if cfg.reset_on_camera_change and self.previous_images is not None:
            if current_images.shape[0] != self.previous_images.shape[0]:
                reasons.append(
                    f"camera_count_changed({current_images.shape[0]}!={self.previous_images.shape[0]})"
                )

        # Image shape changed
        if cfg.reset_on_image_shape_change and self.previous_images is not None:
            if current_images.shape != self.previous_images.shape:
                reasons.append(
                    f"image_shape_changed({list(current_images.shape)}!={list(self.previous_images.shape)})"
                )

        # Visual token grid changed
        if current_token_map is not None and self.previous_token_index_map is not None:
            if current_token_map.n_visual != self.previous_token_index_map.n_visual:
                reasons.append(
                    f"visual_token_count_changed({current_token_map.n_visual}!={self.previous_token_index_map.n_visual})"
                )

        # Proprioception delta exceeded
        if cfg.reset_on_proprio_change and current_proprio is not None:
            delta = compute_proprio_delta(current_proprio, self.previous_proprio_state)
            if delta > cfg.proprio_delta_threshold:
                reasons.append(f"proprio_delta_exceeded({delta:.4f}>{cfg.proprio_delta_threshold})")

        # Batch size changed
        if self.previous_visual_tokens is not None and batch_size != self.previous_visual_tokens.shape[0]:
            reasons.append(f"batch_size_changed({batch_size})")

        should_reset = len(reasons) > 0
        return should_reset, "; ".join(reasons) if reasons else ""

    def compute_static_tokens(
        self,
        current_images: torch.Tensor,
        previous_images: torch.Tensor,
        patch_grid_info: Optional[dict[str, tuple[int, int]]] = None,
    ) -> dict[str, torch.Tensor]:
        """Identify static visual tokens via patch similarity.

        Compares current and previous frames patch-wise to find regions
        that haven't changed significantly.

        Args:
            current_images: [V, 3, H, W] float32 normalized current images.
            previous_images: [V, 3, H, W] float32 previous images.
            patch_grid_info: Optional per-view (h_patches, w_patches).

        Returns:
            dict mapping view_name to boolean mask of static tokens.
        """
        cfg = self.config
        V = current_images.shape[0]
        patch_sim = compute_patch_similarity(current_images, previous_images, patch_size=16)
        # patch_sim: [V, n_patches]

        view_names = list(patch_grid_info.keys()) if patch_grid_info else [f"view_{i}" for i in range(V)]

        static_masks = {}
        for v_idx, view_name in enumerate(view_names):
            sim = patch_sim[v_idx]  # [n_patches]

            if cfg.static_topk is not None and cfg.static_topk < len(sim):
                # Top-k most similar patches are static
                _, topk_idx = torch.topk(sim, cfg.static_topk)
                static_mask = torch.zeros(len(sim), dtype=torch.bool, device=sim.device)
                static_mask[topk_idx] = True
            else:
                # Threshold-based
                static_mask = sim >= cfg.static_similarity_threshold

            # Apply per-view budget
            if cfg.per_view_budget:
                if "wrist" in view_name.lower() and cfg.disable_wrist_cache:
                    static_mask[:] = False
                elif "wrist" in view_name.lower():
                    max_static = max(1, int(len(static_mask) * cfg.wrist_max_reuse_ratio))
                    n_static = static_mask.sum().item()
                    if n_static > max_static:
                        # Keep top-k by similarity
                        _, keep_idx = torch.topk(sim, max_static)
                        static_mask[:] = False
                        static_mask[keep_idx] = True
                else:
                    max_static = max(1, int(len(static_mask) * cfg.external_max_reuse_ratio))
                    n_static = static_mask.sum().item()
                    if n_static > max_static:
                        _, keep_idx = torch.topk(sim, max_static)
                        static_mask[:] = False
                        static_mask[keep_idx] = True

            static_masks[view_name] = static_mask

        return static_masks

    def compute_task_relevance(
        self,
        attention_maps: dict[int, torch.Tensor],
        token_index_map: TokenIndexMap,
    ) -> torch.Tensor:
        """Extract task-relevance scores for visual tokens from attention maps.

        Priority: Use action-head cross-attention if available,
        otherwise use language-to-vision attention from backbone.

        Args:
            attention_maps: Layer index → attention tensor mapping.
                For backbone: [B, heads, seq_len, seq_len] per layer.
                For DiT cross-attn: [B, heads, query_len, condition_len].
            token_index_map: Token index map identifying visual token positions.

        Returns:
            Relevance scores for each visual token [n_visual].
        """
        cfg = self.config
        visual_idx = token_index_map.visual_indices
        if visual_idx is None:
            return torch.ones(token_index_map.n_visual)

        device = next(iter(attention_maps.values())).device

        # Aggregate attention scores across layers
        all_scores = []
        for layer_idx, attn in attention_maps.items():
            if attn is None:
                continue

            # Average over heads and batch
            if attn.dim() == 4:
                # [B, heads, seq, seq] → average over heads and batch
                attn_avg = attn.mean(dim=(0, 1))  # [seq, seq]
            else:
                attn_avg = attn.mean(dim=0)

            # For backbone self-attention: extract rows that are text tokens
            # attending to visual tokens
            if token_index_map.text_indices is not None and len(token_index_map.text_indices) > 0:
                text_to_vis = attn_avg[token_index_map.text_indices][:, visual_idx]
                vis_relevance = text_to_vis.mean(dim=0)  # Average over querying text tokens
            else:
                # For cross-attention maps: just average over query positions
                if attn_avg.shape[0] >= len(visual_idx):
                    vis_relevance = attn_avg.mean(dim=0)[:len(visual_idx)]
                else:
                    vis_relevance = attn_avg.mean(dim=0)

            if len(vis_relevance) == len(visual_idx):
                all_scores.append(vis_relevance.to(device))

        if not all_scores:
            # No attention maps available — return uniform scores
            return torch.ones(token_index_map.n_visual, device=device) / token_index_map.n_visual

        # Average across layers
        relevance = torch.stack(all_scores).mean(dim=0)

        # Normalize to [0, 1]
        relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-8)

        return relevance

    def evict_task_relevant_tokens(
        self,
        static_indices: dict[str, torch.Tensor],
        relevance_scores: torch.Tensor,
        token_index_map: TokenIndexMap,
    ) -> dict[str, torch.Tensor]:
        """Remove task-relevant tokens from the static set.

        Tokens with high attention scores must be recomputed even if they
        appear visually static.

        Args:
            static_indices: Per-view boolean masks of static tokens.
            relevance_scores: Per-visual-token relevance [0,1].
            token_index_map: Token index map.

        Returns:
            dict with:
                "final_reuse": bool mask of tokens to reuse (per view)
                "task_evicted": bool mask of tokens evicted from static set
                "task_relevant": bool mask of task-relevant tokens
        """
        cfg = self.config

        task_relevant = torch.zeros(token_index_map.n_visual, dtype=torch.bool, device=relevance_scores.device)

        if cfg.task_attention_threshold is not None:
            task_relevant = relevance_scores >= cfg.task_attention_threshold
        elif cfg.task_topk is not None:
            _, topk_idx = torch.topk(relevance_scores, min(cfg.task_topk, len(relevance_scores)))
            task_relevant[topk_idx] = True

        # Build per-view reuse masks
        final_reuse = {}
        task_evicted = {}
        vis_offset = 0

        for view_name, static_mask in static_indices.items():
            n_vis = len(static_mask)
            view_slice = slice(vis_offset, vis_offset + n_vis)

            # Task-relevant tokens for this view
            view_task_rel = task_relevant[view_slice]

            # Static but not task-relevant → can reuse
            view_reuse = static_mask.clone() & (~view_task_rel)
            # Task-relevant tokens that were initially static → evicted
            view_evicted = static_mask.clone() & view_task_rel

            final_reuse[view_name] = view_reuse
            task_evicted[view_name] = view_evicted

            vis_offset += n_vis

        return {
            "final_reuse": final_reuse,
            "task_evicted": task_evicted,
            "task_relevant": task_relevant,
            "relevance_scores": relevance_scores,
        }

    def compute_layer_adaptive_reuse(
        self,
        attention_maps: dict[int, torch.Tensor],
        base_reuse_plan: dict[str, torch.Tensor],
        n_visual: int,
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Compute per-layer adaptive token reuse ratios.

        Early layers have higher entropy (less concentrated attention) and
        should be more conservative. Deeper layers with concentrated attention
        can reuse more tokens.

        Args:
            attention_maps: Per-layer attention tensors.
            base_reuse_plan: Base reuse masks from static + task analysis.
            n_visual: Total number of visual tokens.

        Returns:
            dict mapping layer_idx → {view_name → reuse_mask}.
        """
        cfg = self.config

        # Compute per-layer entropy concentration
        layer_concentrations = {}
        for layer_idx, attn in attention_maps.items():
            if attn is None:
                continue
            # Average over heads and batch
            attn_avg = attn.float().mean(dim=(0, 1))
            # Entropy per token position
            entropy = compute_entropy(attn_avg, dim=-1)
            concentration = compute_entropy_concentration(entropy)
            layer_concentrations[layer_idx] = concentration

        if not layer_concentrations:
            # No attention info — use base plan uniformly
            return {0: base_reuse_plan}

        # Normalize concentrations to [0,1] across layers
        concs = list(layer_concentrations.values())
        conc_min, conc_max = min(concs), max(concs)
        if conc_max > conc_min:
            norm_concs = {
                k: (v - conc_min) / (conc_max - conc_min)
                for k, v in layer_concentrations.items()
            }
        else:
            norm_concs = {k: 0.5 for k in layer_concentrations.keys()}

        # Map concentration → reuse ratio
        layer_reuse = {}
        for layer_idx, norm_conc in norm_concs.items():
            # Higher concentration → more reuse allowed
            reuse_ratio = (
                cfg.min_reuse_ratio
                + (cfg.max_reuse_ratio - cfg.min_reuse_ratio)
                * norm_conc
                * cfg.entropy_scale
            )
            reuse_ratio = min(reuse_ratio, cfg.max_reuse_ratio)
            reuse_ratio = max(reuse_ratio, cfg.min_reuse_ratio)

            # Apply ratio to each view's base reuse mask
            layer_reuse[layer_idx] = {}
            for view_name, base_mask in base_reuse_plan.items():
                n_view = len(base_mask)
                target_reuse = max(1, int(n_view * reuse_ratio))
                current_reuse = base_mask.sum().item()

                if current_reuse <= target_reuse:
                    layer_reuse[layer_idx][view_name] = base_mask
                else:
                    # Subsample base reuse mask to meet target ratio
                    # Keep tokens with higher static similarity (earlier in sorted order)
                    indices = torch.where(base_mask)[0]
                    keep = indices[:target_reuse]
                    new_mask = torch.zeros_like(base_mask)
                    new_mask[keep] = True
                    layer_reuse[layer_idx][view_name] = new_mask

        return layer_reuse

    def update_cache(
        self,
        current_images: torch.Tensor,
        current_visual_tokens: torch.Tensor,
        current_token_map: TokenIndexMap,
        current_instruction: Optional[str],
        current_proprio: Optional[torch.Tensor],
        attention_maps: dict[int, torch.Tensor],
        layer_kv: dict[int, dict[str, torch.Tensor]],
        condition_kv: Optional[dict[int, dict[str, torch.Tensor]]] = None,
        condition_features: Optional[torch.Tensor] = None,
    ) -> None:
        """Update the cache with current step data.

        Args:
            current_images: Current visual observations.
            current_visual_tokens: Current visual tokens.
            current_token_map: Token index map.
            current_instruction: Current instruction text.
            current_proprio: Current proprioception state.
            attention_maps: Collected attention maps.
            layer_kv: Per-layer KV cache from backbone.
            condition_kv: Per-layer KV cache from action head condition.
            condition_features: Condition features (for similarity check).
        """
        should_reset, reason = self.should_reset_cache(
            current_images=current_images,
            current_instruction=current_instruction,
            current_proprio=current_proprio,
            current_token_map=current_token_map,
        )

        if should_reset:
            if self.config.debug:
                print(f"[GR00T-Cache] Cache reset: {reason}")
            self.reset_count += 1
            self.cache_age = 0
        else:
            self.cache_hits += 1
            self.cache_age += 1

        # Store current state
        self.cache_step_id += 1
        self.total_steps += 1
        self.previous_images = current_images.detach().clone()
        if current_visual_tokens is not None:
            self.previous_visual_tokens = current_visual_tokens.detach().clone()
        self.previous_token_index_map = current_token_map

        if current_instruction is not None:
            self.previous_instruction_hash = tensor_hash(
                torch.tensor([ord(c) for c in current_instruction[:1000]], dtype=torch.long)
            )

        if current_proprio is not None:
            self.previous_proprio_state = current_proprio.detach().clone()

        # Store KV caches
        if layer_kv:
            self.layer_kv_cache = {
                layer: {k: v.detach().clone() for k, v in kv.items()}
                for layer, kv in layer_kv.items()
            }

        if attention_maps:
            self.layer_attention_maps = {
                layer: attn.detach().clone() for layer, attn in attention_maps.items()
            }

        if condition_kv:
            self.action_head_condition_kv = {
                layer: {k: v.detach().clone() for k, v in kv.items()}
                for layer, kv in condition_kv.items()
            }

        if condition_features is not None:
            self.previous_condition_features = condition_features.detach().clone()

    def get_reuse_plan(
        self,
        current_images: torch.Tensor,
        current_instruction: Optional[str] = None,
        current_proprio: Optional[torch.Tensor] = None,
        current_token_map: Optional[TokenIndexMap] = None,
        attention_maps: Optional[dict[int, torch.Tensor]] = None,
        batch_size: int = 1,
    ) -> dict:
        """Compute the full reuse plan for the current step.

        This is the main entry point for cache decision-making.

        Returns:
            dict with keys:
                "should_cache": bool
                "reuse_per_layer": dict[int, dict[str, torch.Tensor]]
                "reuse_ratio": float
                "static_masks": dict[str, torch.Tensor]
                "task_evicted": dict[str, torch.Tensor]
                "cache_age": int
                "reset_reason": str (empty if not reset)
        """
        cfg = self.config

        if not cfg.enabled:
            return {"should_cache": False, "reuse_ratio": 0.0}

        # Check if cache should be invalidated
        should_reset, reset_reason = self.should_reset_cache(
            current_images=current_images,
            current_instruction=current_instruction,
            current_proprio=current_proprio,
            current_token_map=current_token_map,
            batch_size=batch_size,
        )

        should_cache = (
            self.previous_images is not None
            and not should_reset
            and self.cache_step_id >= 0
        )

        if not should_cache:
            return {
                "should_cache": False,
                "reuse_ratio": 0.0,
                "reset_reason": reset_reason,
                "cache_age": self.cache_age,
            }

        # Step 1: Find static tokens
        static_masks = self.compute_static_tokens(
            current_images,
            self.previous_images,
            current_token_map.view_patch_grids if current_token_map else None,
        )

        # Step 2: Compute task relevance
        relevance_scores = None
        task_evicted = None
        final_reuse = static_masks  # Default: use static masks directly

        if attention_maps:
            relevance_scores = self.compute_task_relevance(
                attention_maps, current_token_map or TokenIndexMap()
            )
            eviction_result = self.evict_task_relevant_tokens(
                static_masks,
                relevance_scores,
                current_token_map or TokenIndexMap(),
            )
            final_reuse = eviction_result["final_reuse"]
            task_evicted = eviction_result["task_evicted"]

        # Step 3: Layer-adaptive reuse
        if attention_maps and len(attention_maps) > 0:
            reuse_per_layer = self.compute_layer_adaptive_reuse(
                attention_maps,
                final_reuse,
                current_token_map.n_visual if current_token_map else 0,
            )
        else:
            reuse_per_layer = {0: final_reuse}

        # Compute overall reuse ratio
        total_vis = sum(len(m) for m in final_reuse.values())
        if total_vis > 0:
            reused_vis = sum(m.sum().item() for m in final_reuse.values())
            reuse_ratio = reused_vis / total_vis
        else:
            reuse_ratio = 0.0

        plan = {
            "should_cache": True,
            "reuse_per_layer": reuse_per_layer,
            "reuse_ratio": reuse_ratio,
            "static_masks": static_masks,
            "task_evicted": task_evicted,
            "relevance_scores": relevance_scores,
            "cache_age": self.cache_age,
            "reset_reason": "",
        }

        self._current_reuse_plan = plan
        return plan

    def get_condition_cache(
        self,
        layer_idx: int,
    ) -> Optional[dict[str, torch.Tensor]]:
        """Retrieve cached condition K/V for action head cross-attention.

        Returns:
            {'k': tensor, 'v': tensor} or None if not available.
        """
        if layer_idx in self.action_head_condition_kv:
            return self.action_head_condition_kv[layer_idx]
        return None

    def get_backbone_kv(
        self, layer_idx: int
    ) -> Optional[dict[str, torch.Tensor]]:
        """Retrieve cached backbone KV for a specific layer."""
        if layer_idx in self.layer_kv_cache:
            return self.layer_kv_cache[layer_idx]
        return None

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate over the session."""
        if self.total_steps == 0:
            return 0.0
        return self.cache_hits / self.total_steps

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        plan = self._current_reuse_plan or {}
        return {
            "cache_step_id": self.cache_step_id,
            "cache_age": self.cache_age,
            "cache_hit_rate": self.cache_hit_rate,
            "reset_count": self.reset_count,
            "total_steps": self.total_steps,
            "reuse_ratio": plan.get("reuse_ratio", 0.0),
            "has_static_masks": plan.get("static_masks") is not None,
            "has_task_eviction": plan.get("task_evicted") is not None,
            "has_layer_adaptive": len(plan.get("reuse_per_layer", {})) > 1,
            "n_backbone_kv_layers": len(self.layer_kv_cache),
            "n_condition_kv_layers": len(self.action_head_condition_kv),
        }
