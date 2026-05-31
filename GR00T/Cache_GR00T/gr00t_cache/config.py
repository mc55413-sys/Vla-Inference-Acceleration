# SPDX-FileCopyrightText: Copyright (c) 2025 GR00T-Cache Contributors
# SPDX-License-Identifier: Apache-2.0
"""Configuration dataclass for GR00T-Cache."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CacheMode(str, Enum):
    """Cache mode for GR00T-Cache."""
    NONE = "none"
    BACKBONE_VISUAL_KV_CACHE = "backbone_visual_kv_cache"
    ACTION_HEAD_CONDITION_KV_CACHE = "action_head_condition_kv_cache"
    FULL_CACHE = "full_cache"


@dataclass
class GR00TCacheConfig:
    """Configuration for GR00T-Cache.

    Controls all aspects of the visual token caching behavior
    for the GR00T VLA model.

    Attributes:
        enabled: Master switch for caching.
        cache_mode: Which parts of the model to cache.
        static_similarity_threshold: Cosine similarity threshold for
            declaring a visual token "static" (range 0-1).
        static_topk: If set, use top-k instead of threshold for static
            token selection.
        max_reuse_ratio: Maximum fraction of visual tokens that can be
            reused in any layer.
        min_reuse_ratio: Minimum fraction of visual tokens to always
            recompute.
        task_attention_threshold: Attention score threshold for task-
            relevant token detection. Tokens above this are forced to
            recompute.
        task_topk: If set, use top-k attention tokens as task-relevant.
        entropy_scale: Multiplier for layer-adaptive reuse ratio scaling.
            Higher values = more aggressive caching in early layers.
        max_cache_age: Number of policy steps before forced cache reset.
        per_view_budget: Apply reuse budget per camera view separately.
        disable_wrist_cache: Never cache wrist camera visual tokens.
        wrist_max_reuse_ratio: More conservative reuse for wrist cam.
        external_max_reuse_ratio: Max reuse for external (3rd-person) cameras.
        reset_on_instruction_change: Invalidate cache when instruction changes.
        reset_on_proprio_change: Invalidate cache on large proprioception delta.
        proprio_delta_threshold: L2 threshold for proprioception change.
        reset_on_camera_change: Invalidate when camera count changes.
        reset_on_image_shape_change: Invalidate when image resolution changes.
        collect_attention_maps: Whether to collect attention scores during forward.
        layer_start: First transformer layer to apply caching (0-indexed).
        layer_end: Last transformer layer to apply caching (exclusive).
        fallback_sdpa: Fall back to SDPA when flash attention doesn't
            support partial KV replacement.
        debug: Print debug information.
        debug_save_dir: Directory to save debug visualizations.
    """
    # ---- Master switches ----
    enabled: bool = True
    cache_mode: CacheMode = CacheMode.FULL_CACHE

    # ---- Static token selection ----
    static_similarity_threshold: float = 0.95
    static_topk: Optional[int] = None

    # ---- Reuse budget ----
    max_reuse_ratio: float = 0.5
    min_reuse_ratio: float = 0.0

    # ---- Task-relevant eviction ----
    task_attention_threshold: Optional[float] = None
    task_topk: Optional[int] = None

    # ---- Layer-adaptive ----
    entropy_scale: float = 1.0

    # ---- Cache lifecycle ----
    max_cache_age: int = 1

    # ---- Per-view policy ----
    per_view_budget: bool = True
    disable_wrist_cache: bool = False
    wrist_max_reuse_ratio: float = 0.2
    external_max_reuse_ratio: float = 0.6

    # ---- Cache invalidation ----
    reset_on_instruction_change: bool = True
    reset_on_proprio_change: bool = True
    proprio_delta_threshold: float = 0.05
    reset_on_camera_change: bool = True
    reset_on_image_shape_change: bool = True

    # ---- Attention collection ----
    collect_attention_maps: bool = True

    # ---- Layer range ----
    layer_start: int = 0
    layer_end: Optional[int] = None

    # ---- Fallback ----
    fallback_sdpa: bool = True

    # ---- Debug ----
    debug: bool = False
    debug_save_dir: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize config to a JSON-safe dict."""
        return {
            "enabled": self.enabled,
            "cache_mode": self.cache_mode.value,
            "static_similarity_threshold": self.static_similarity_threshold,
            "static_topk": self.static_topk,
            "max_reuse_ratio": self.max_reuse_ratio,
            "min_reuse_ratio": self.min_reuse_ratio,
            "task_attention_threshold": self.task_attention_threshold,
            "task_topk": self.task_topk,
            "entropy_scale": self.entropy_scale,
            "max_cache_age": self.max_cache_age,
            "per_view_budget": self.per_view_budget,
            "disable_wrist_cache": self.disable_wrist_cache,
            "wrist_max_reuse_ratio": self.wrist_max_reuse_ratio,
            "external_max_reuse_ratio": self.external_max_reuse_ratio,
            "reset_on_instruction_change": self.reset_on_instruction_change,
            "reset_on_proprio_change": self.reset_on_proprio_change,
            "proprio_delta_threshold": self.proprio_delta_threshold,
            "reset_on_camera_change": self.reset_on_camera_change,
            "reset_on_image_shape_change": self.reset_on_image_shape_change,
            "collect_attention_maps": self.collect_attention_maps,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "fallback_sdpa": self.fallback_sdpa,
            "debug": self.debug,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GR00TCacheConfig":
        """Load config from a dict."""
        if "cache_mode" in d and isinstance(d["cache_mode"], str):
            d = dict(d)
            d["cache_mode"] = CacheMode(d["cache_mode"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def clone(self, **overrides) -> "GR00TCacheConfig":
        """Create a copy with optional overrides."""
        d = self.to_dict()
        d.update(overrides)
        return self.from_dict(d)
