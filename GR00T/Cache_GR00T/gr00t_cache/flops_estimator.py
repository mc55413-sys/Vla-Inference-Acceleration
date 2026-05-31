# SPDX-FileCopyrightText: Copyright (c) 2025 GR00T-Cache Contributors
# SPDX-License-Identifier: Apache-2.0
"""Theoretical TFLOPs estimation for Transformer models with token caching.

Provides both naive (per-token FLOPs count) and conservative (KV-projection
only) estimates of computational savings from visual token reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FLOPSEstimate:
    """Container for FLOPs estimation results."""
    full_flops: float
    cached_flops: float
    projection_saved_flops: float
    attention_saved_flops: float
    total_saved_flops: float
    flops_reduction_percent: float

    @property
    def full_tflops(self) -> float:
        return self.full_flops / 1e12

    @property
    def cached_tflops(self) -> float:
        return self.cached_flops / 1e12

    @property
    def saved_tflops(self) -> float:
        return self.total_saved_flops / 1e12

    def to_dict(self) -> dict:
        return {
            "full_flops": self.full_flops,
            "cached_flops": self.cached_flops,
            "projection_saved_flops": self.projection_saved_flops,
            "attention_saved_flops": self.attention_saved_flops,
            "total_saved_flops": self.total_saved_flops,
            "flops_reduction_percent": self.flops_reduction_percent,
            "full_tflops": self.full_tflops,
            "cached_tflops": self.cached_tflops,
            "saved_tflops": self.saved_tflops,
        }


def transformer_layer_flops(
    n_tokens: int,
    d_model: int,
    ffn_dim: int,
    n_kv_heads: int = -1,
    n_q_heads: int = -1,
    head_dim: int = -1,
) -> float:
    """Compute FLOPs for one transformer layer.

    Uses the standard formula:
    C(n) = 4 * n * d^2 + 2 * n^2 * d + 2 * n * d * ffn_dim

    Where:
    - 4 * n * d^2: Q, K, V, O projections (4x matmul)
    - 2 * n^2 * d: Attention score computation + weighted sum
    - 2 * n * d * ffn_dim: Two FFN matmuls (up + down)

    Args:
        n_tokens: Number of tokens in the sequence.
        d_model: Hidden dimension.
        ffn_dim: FFN intermediate dimension.
        n_kv_heads: Number of KV heads (for GQA estimate; -1 = same as Q).
        n_q_heads: Number of query heads.
        head_dim: Dimension per head.

    Returns:
        Estimated FLOPs per layer.
    """
    # QKV projections: 3 * n * d^2
    proj_flops = 3 * n_tokens * d_model * d_model
    # Output projection: n * d^2
    out_proj_flops = n_tokens * d_model * d_model
    # Total projection FLOPs
    total_proj = proj_flops + out_proj_flops  # 4 * n * d^2

    # Attention: score matmul n*n*d + weighted sum n*n*d
    attn_flops = 2 * n_tokens * n_tokens * d_model

    # FFN: Two matmuls n * d * ffn_dim each
    ffn_flops = 2 * n_tokens * d_model * ffn_dim

    return total_proj + attn_flops + ffn_flops


def estimate_transformer_flops(
    num_layers: int,
    n_tokens: int,
    d_model: int,
    ffn_dim: int,
) -> float:
    """Estimate total FLOPs for a transformer (naive count).

    Args:
        num_layers: Number of transformer layers.
        n_tokens: Total sequence length.
        d_model: Hidden dimension.
        ffn_dim: FFN intermediate dimension.

    Returns:
        Total estimated FLOPs.
    """
    per_layer = transformer_layer_flops(n_tokens, d_model, ffn_dim)
    return per_layer * num_layers


def estimate_cache_transformer_flops(
    num_layers: int,
    text_tokens: int,
    visual_tokens: int,
    d_model: int,
    ffn_dim: int,
    reuse_ratios_by_layer: list[float],
) -> FLOPSEstimate:
    """Estimate FLOPs with per-layer token reuse.

    When visual tokens are cached, the effective sequence length per layer is:
        n_eff_l = text_tokens + visual_tokens * (1 - reuse_ratio_l)

    Args:
        num_layers: Number of transformer layers.
        text_tokens: Number of text tokens (always computed).
        visual_tokens: Number of visual tokens (partially reusable).
        d_model: Hidden dimension.
        ffn_dim: FFN intermediate dimension.
        reuse_ratios_by_layer: List of reuse ratios per layer [0,1].
            Length should match num_layers.

    Returns:
        FLOPSEstimate with full and cached flops.
    """
    n_full = text_tokens + visual_tokens
    full_flops = sum(
        transformer_layer_flops(n_full, d_model, ffn_dim)
        for _ in range(num_layers)
    )

    if len(reuse_ratios_by_layer) != num_layers:
        # Pad or truncate
        if len(reuse_ratios_by_layer) < num_layers:
            # Replicate last value
            reuse_ratios_by_layer = reuse_ratios_by_layer + [
                reuse_ratios_by_layer[-1]
            ] * (num_layers - len(reuse_ratios_by_layer))
        else:
            reuse_ratios_by_layer = reuse_ratios_by_layer[:num_layers]

    cached_flops = 0.0
    projection_saved = 0.0
    attention_saved = 0.0

    for r in reuse_ratios_by_layer:
        n_eff = text_tokens + visual_tokens * (1.0 - r)

        # Full compute for this layer
        layer_full = transformer_layer_flops(n_full, d_model, ffn_dim)
        # Cached compute
        layer_cached = transformer_layer_flops(int(n_eff), d_model, ffn_dim)

        cached_flops += layer_cached

        # Conservative estimate: only K/V projection saved
        # K projection: n_visual * r * d^2
        # V projection: n_visual * r * d^2
        proj_saved = 2 * visual_tokens * r * d_model * d_model
        projection_saved += proj_saved

        # Attention saved: fewer tokens in Q*K^T and attn*V
        # Full attn: 2 * (n_full)^2 * d
        # Reduced attn: 2 * (n_eff)^2 * d
        attn_saved_layer = 2 * (n_full ** 2 - int(n_eff) ** 2) * d_model
        attention_saved += max(0, attn_saved_layer)

    total_saved = projection_saved + attention_saved

    reduction = (1.0 - cached_flops / full_flops) * 100 if full_flops > 0 else 0.0

    return FLOPSEstimate(
        full_flops=full_flops,
        cached_flops=cached_flops,
        projection_saved_flops=projection_saved,
        attention_saved_flops=attention_saved,
        total_saved_flops=total_saved,
        flops_reduction_percent=reduction,
    )


def compute_gr00t_model_flops(
    backbone_num_layers: int,
    backbone_text_tokens: int,
    backbone_visual_tokens: int,
    backbone_d_model: int,
    backbone_ffn_dim: int,
    backbone_reuse_ratios: list[float],
    dit_num_layers: int,
    dit_query_tokens: int,
    dit_condition_tokens: int,
    dit_d_model: int,
    dit_ffn_dim: int,
    dit_condition_reuse_ratios: Optional[list[float]] = None,
    vl_self_attn_layers: int = 1,
) -> dict[str, FLOPSEstimate]:
    """Compute combined GR00T model FLOPs estimates.

    Estimates FLOPs for:
    1. Backbone LLM (vision-language processing)
    2. Action head VL self-attention
    3. Action head DiT (cross-attention + self-attention)

    Returns:
        dict with per-component and total FLOPs estimates.
    """
    estimates = {}

    # Backbone LLM
    estimates["backbone"] = estimate_cache_transformer_flops(
        num_layers=backbone_num_layers,
        text_tokens=backbone_text_tokens,
        visual_tokens=backbone_visual_tokens,
        d_model=backbone_d_model,
        ffn_dim=backbone_ffn_dim,
        reuse_ratios_by_layer=backbone_reuse_ratios,
    )

    # VL self-attention in action head
    vl_n_tokens = backbone_text_tokens + backbone_visual_tokens
    estimates["vl_self_attn"] = FLOPSEstimate(
        full_flops=vl_self_attn_layers * transformer_layer_flops(vl_n_tokens, dit_d_model, dit_ffn_dim),
        cached_flops=vl_self_attn_layers * transformer_layer_flops(vl_n_tokens, dit_d_model, dit_ffn_dim),
        projection_saved_flops=0,
        attention_saved_flops=0,
        total_saved_flops=0,
        flops_reduction_percent=0.0,
    )

    # DiT (cross-attention uses condition tokens, self-attention uses query tokens)
    if dit_condition_reuse_ratios is None:
        dit_condition_reuse_ratios = [0.0] * dit_num_layers

    dit_full = 0.0
    dit_cached = 0.0
    for r in dit_condition_reuse_ratios:
        eff_cond = dit_condition_tokens * (1.0 - r)
        # Self-attention on query tokens (no reuse)
        dit_full += transformer_layer_flops(dit_query_tokens, dit_d_model, dit_ffn_dim)
        dit_cached += transformer_layer_flops(dit_query_tokens, dit_d_model, dit_ffn_dim)
        # Cross-attention: Q from query, K/V from condition
        # We save on K/V projection of condition tokens
        dit_full += transformer_layer_flops(dit_condition_tokens, dit_d_model, dit_ffn_dim)
        dit_cached += transformer_layer_flops(int(eff_cond), dit_d_model, dit_ffn_dim)

    dit_reduction = (1.0 - dit_cached / dit_full) * 100 if dit_full > 0 else 0.0
    estimates["dit"] = FLOPSEstimate(
        full_flops=dit_full,
        cached_flops=dit_cached,
        projection_saved_flops=dit_full - dit_cached,
        attention_saved_flops=0,  # DiT cross-attn is dominated by projection
        total_saved_flops=dit_full - dit_cached,
        flops_reduction_percent=dit_reduction,
    )

    # Total
    total_full = sum(e.full_flops for e in estimates.values())
    total_cached = sum(e.cached_flops for e in estimates.values())
    total_saved = sum(e.total_saved_flops for e in estimates.values())
    total_reduction = (1.0 - total_cached / total_full) * 100 if total_full > 0 else 0.0

    estimates["total"] = FLOPSEstimate(
        full_flops=total_full,
        cached_flops=total_cached,
        projection_saved_flops=sum(e.projection_saved_flops for e in estimates.values()),
        attention_saved_flops=sum(e.attention_saved_flops for e in estimates.values()),
        total_saved_flops=total_saved,
        flops_reduction_percent=total_reduction,
    )

    return estimates
