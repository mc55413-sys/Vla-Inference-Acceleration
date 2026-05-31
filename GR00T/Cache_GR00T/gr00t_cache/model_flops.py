# SPDX-FileCopyrightText: Copyright (c) 2025 GR00T-Cache Contributors
# SPDX-License-Identifier: Apache-2.0
"""
GR00T model FLOPs counter — measures every nn.Linear and nn.Conv2d.

Formula:
  Linear:  FLOPs = 2 × tokens × in_features × out_features
  Conv2d:  FLOPs = 2 × batch × out_c × out_h × out_w × (k_h × k_w × in_c / groups)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


def count_linear_flops(module: nn.Linear, input_tensor: torch.Tensor) -> int:
    """Count FLOPs for nn.Linear (or DuQuantLinear)."""
    in_features = module.in_features
    out_features = module.out_features
    tokens = input_tensor.numel() // in_features  # batch * seq_len or batch * tokens
    return 2 * tokens * in_features * out_features


def count_conv2d_flops(module: nn.Conv2d, input_tensor: torch.Tensor) -> int:
    """Count FLOPs for nn.Conv2d."""
    if input_tensor.dim() != 4:
        return 0
    batch, in_c, in_h, in_w = input_tensor.shape
    out_c = module.out_channels
    k_h, k_w = module.kernel_size
    stride = module.stride
    padding = module.padding
    dilation = module.dilation
    groups = module.groups

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1

    return 2 * batch * out_c * out_h * out_w * (k_h * k_w * in_c // groups)


# ── FLOPs counting by hook ──────────────────────────────────────────────

class FLOPsCounter:
    """Collects per-module FLOPs via forward hooks."""

    def __init__(self):
        self.flops: dict[str, int] = {}
        self.handles: list = []

    def _hook_fn(self, name: str):
        def hook(module, inputs, output):
            if not inputs or inputs[0] is None:
                return
            inp = inputs[0]
            if isinstance(module, nn.Linear):
                self.flops[name] = self.flops.get(name, 0) + count_linear_flops(module, inp)
            elif isinstance(module, nn.Conv2d):
                self.flops[name] = self.flops.get(name, 0) + count_conv2d_flops(module, inp)

        return hook

    def register(self, model: nn.Module, prefix: str = ""):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                full_name = f"{prefix}.{name}" if prefix else name
                handle = module.register_forward_hook(self._hook_fn(full_name))
                self.handles.append(handle)

    def reset(self):
        self.flops.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def total(self) -> int:
        return sum(self.flops.values())

    def summary(self) -> dict[str, Any]:
        total = self.total()
        by_component: dict[str, int] = {}
        for name, flops in self.flops.items():
            # Classify
            if "vision" in name or "patch_embed" in name:
                comp = "vision_encoder"
            elif "mlp1" in name:
                comp = "vision_projector"
            elif "language_model" in name or "lm_head" in name:
                comp = "backbone_llm"
            elif "eagle_linear" in name:
                comp = "backbone_bridge"
            elif "vlln" in name:
                comp = "s1_vision_norm"
            elif "vl_self_attention" in name:
                comp = "s1_vision_self_attn"
            elif "state_encoder" in name:
                comp = "s1_state_encoder"
            elif "action_encoder" in name:
                comp = "s1_action_encoder"
            elif "action_decoder" in name:
                comp = "s1_action_decoder"
            elif "future_tokens" in name:
                comp = "s1_future_tokens"
            elif "transformer_blocks" in name or ".blocks." in name:
                comp = "s1_dit"
            elif "position_embedding" in name:
                comp = "s1_pos_embed"
            elif "time" in name and "dit" in prefix:
                comp = "s1_dit"
            elif "to_q" in name or "to_k" in name or "to_v" in name or "to_out" in name:
                comp = "s1_dit"
            elif "ff" in name:
                comp = "s1_dit"
            elif "norm" in name or "ada" in name:
                comp = "s1_dit"
            else:
                comp = "other"
            by_component[comp] = by_component.get(comp, 0) + flops

        return {
            "total_flops": total,
            "total_gflops": total / 1e9,
            "total_tflops": total / 1e12,
            "by_component": {
                k: {"flops": v, "gflops": v / 1e9, "pct": v / total * 100 if total else 0}
                for k, v in sorted(by_component.items(), key=lambda x: -x[1])
            },
            "per_module_flops": dict(sorted(self.flops.items(), key=lambda x: -x[1])[:20]),
        }


# ── Static FLOPs calculation (no hooks, uses model config) ────────────────

def compute_gr00t_static_flops(
    model,
    visual_tokens: int = 768,
    text_tokens: int = 80,
    state_tokens: int = 1,
    future_tokens: int = 32,
    action_tokens: int = 16,
    num_denoising_steps: int = 8,
) -> dict[str, Any]:
    """Compute theoretical FLOPs from model architecture (no hooks)."""
    backbone = model.backbone
    eagle = backbone.eagle_model
    lm = eagle.language_model

    backbone_layers = len(lm.model.layers)
    backbone_d = lm.config.hidden_size
    backbone_ffn = getattr(lm.config, "intermediate_size", backbone_d * 4)
    vision_d = eagle.config.vision_config.hidden_size
    projector_d = backbone_d  # projected to LLM dim

    total_emb_tokens = text_tokens + visual_tokens  # total tokens in LLM

    flops = {}

    # ── Vision encoder ──
    # SigLIP: Conv2d patch embed + transformer layers
    # Approximate: Conv2d(3→vision_d, kernel=patch_size, stride=patch_size)
    # For image 256x256, patch=14 → 18*18=324 patches, but Eagle uses 448x448 tiles
    # For Libero 256x256 with Eagle dynamic tiling: ~1-2 tiles of 448x448
    vision_patches = visual_tokens  # after pixel shuffle
    # Vision transformer is already counted by hooks; static is approximate
    # We'll skip detailed vision FLOPs and use hook-based measurement for that

    # ── Vision projector (mlp1: LayerNorm + Linear + GELU + Linear) ──
    mlp1_flops = (
        2 * visual_tokens * vision_d * projector_d  # first linear
        + 2 * visual_tokens * projector_d * projector_d  # second linear
    )
    flops["vision_projector"] = mlp1_flops

    # ── Backbone LLM per layer ──
    # Q/K/V projections: 3 * 2 * total_emb_tokens * d^2
    # O projection: 2 * total_emb_tokens * d^2
    # Attention matmul: 2 * total_emb_tokens^2 * d
    # FFN: 2 * 2 * total_emb_tokens * d * ffn_dim
    per_layer = (
        8 * total_emb_tokens * backbone_d * backbone_d  # Q/K/V/O
        + 2 * total_emb_tokens * total_emb_tokens * backbone_d  # attention
        + 4 * total_emb_tokens * backbone_d * backbone_ffn  # FFN
    )
    backbone_total = per_layer * backbone_layers
    flops["backbone_llm"] = backbone_total

    # ── Eagle linear (backbone → condition) ──
    eagle_linear_flops = 2 * total_emb_tokens * backbone_d * 1536
    flops["backbone_bridge"] = eagle_linear_flops

    # ── VL self-attention in action head ──
    ah = model.action_head
    vl_sa_layers = len(ah.vl_self_attention.transformer_blocks) if hasattr(ah.vl_self_attention, "transformer_blocks") else 1
    vl_sa_d = getattr(ah.vl_self_attention, "inner_dim", 512)
    vl_sa_ffn = vl_sa_d * 4  # approximate
    vl_sa_per_layer = (
        8 * total_emb_tokens * vl_sa_d * vl_sa_d
        + 2 * total_emb_tokens * total_emb_tokens * vl_sa_d
        + 4 * total_emb_tokens * vl_sa_d * vl_sa_ffn
    )
    flops["s1_vl_self_attn"] = vl_sa_per_layer * vl_sa_layers

    # ── DiT action head ──
    dit = ah.model
    dit_layers = len(dit.transformer_blocks)
    dit_d = dit.inner_dim
    dit_ffn = getattr(dit.config, "ff_inner_dim", dit_d * 4)

    query_tokens = state_tokens + future_tokens + action_tokens  # ~49
    condition_tokens = total_emb_tokens  # text + visual

    # Per DiT layer (per denoising step):
    # Self-attn on query: Q/K/V/O (4×2×query×d²) + attn matmul (2×query²×d)
    # Cross-attn: Q from query (2×query×d²), K/V from condition (2×2×cond×d²), O (2×query×d²)
    #   + attn matmul (2×query×cond×d)
    # FFN: 4×query×d×ffn
    per_dit_layer = (
        4 * 2 * query_tokens * dit_d * dit_d  # self-attn Q/K/V/O
        + 2 * query_tokens * query_tokens * dit_d  # self-attn matmul
        + 2 * query_tokens * dit_d * dit_d  # cross attn Q
        + 2 * 2 * condition_tokens * dit_d * dit_d  # cross attn K/V
        + 2 * query_tokens * dit_d * dit_d  # cross attn O
        + 2 * query_tokens * condition_tokens * dit_d  # cross attn matmul
        + 4 * query_tokens * dit_d * dit_ffn  # FFN
    )
    dit_per_step = per_dit_layer * dit_layers
    dit_total = dit_per_step * num_denoising_steps

    # With cache within a step: save (num_steps - 1) * cross_attn K/V + O projections
    # cross_attn saved per step: 2 * cond * d² (K) + 2 * cond * d² (V) + 2 * query * d² (O)
    cross_attn_saved_per_step = (
        2 * 2 * condition_tokens * dit_d * dit_d  # K/V
        + 2 * query_tokens * dit_d * dit_d  # O
    )
    dit_cached_total = dit_total - cross_attn_saved_per_step * dit_layers * (num_denoising_steps - 1)

    flops["s1_dit"] = dit_total
    flops["s1_dit_cached"] = max(dit_total, 0)

    # ── Action decoder ──
    decoder_flops = (
        2 * query_tokens * dit_d * ah.hidden_size * 2  # two CategorySpecificMLP layers
        + 2 * query_tokens * ah.hidden_size * ah.action_dim
    )
    flops["s1_action_decoder"] = decoder_flops

    # ── Totals ──
    total_full = sum(flops.values())
    total_cached = total_full - (dit_total - dit_cached_total)

    return {
        "components": {k: {"gflops": v / 1e9} for k, v in flops.items()},
        "full": {
            "total_gflops": total_full / 1e9,
            "total_tflops": total_full / 1e12,
        },
        "cached": {
            "total_gflops": total_cached / 1e9,
            "total_tflops": total_cached / 1e12,
        },
        "saved": {
            "total_gflops": (total_full - total_cached) / 1e9,
            "total_tflops": (total_full - total_cached) / 1e12,
            "pct": (total_full - total_cached) / total_full * 100 if total_full else 0,
        },
        "model_params": {
            "backbone_layers": backbone_layers,
            "backbone_d": backbone_d,
            "backbone_ffn": backbone_ffn,
            "total_tokens": total_emb_tokens,
            "visual_tokens": visual_tokens,
            "text_tokens": text_tokens,
            "dit_layers": dit_layers,
            "dit_d": dit_d,
            "dit_ffn": dit_ffn,
            "num_denoising_steps": num_denoising_steps,
            "query_tokens": query_tokens,
            "condition_tokens": condition_tokens,
        },
    }


# ── Quick print ─────────────────────────────────────────────────────────

def format_flops_table(flops_result: dict, model=None) -> str:
    """Format FLOPs results as a readable table."""
    lines = []
    lines.append("=" * 70)
    lines.append("  GR00T Model FLOPs Breakdown")
    lines.append("=" * 70)

    params = flops_result.get("model_params", {})
    if params:
        lines.append(f"  Architecture:")
        lines.append(f"    Backbone: {params.get('backbone_layers')} layers, "
                     f"d={params.get('backbone_d')}, ffn={params.get('backbone_ffn')}")
        lines.append(f"    Tokens: {params.get('total_tokens')} total "
                     f"({params.get('text_tokens')} text + {params.get('visual_tokens')} visual)")
        lines.append(f"    DiT: {params.get('dit_layers')} layers, "
                     f"d={params.get('dit_d')}, {params.get('num_denoising_steps')} denoising steps")
        lines.append(f"    Query: {params.get('query_tokens')}, Condition: {params.get('condition_tokens')}")

    lines.append(f"\n  Component FLOPs:")
    components = flops_result.get("components", {})
    for name, info in sorted(components.items(), key=lambda x: -x[1]["gflops"]):
        lines.append(f"    {name:30s} {info['gflops']:10.2f} GFLOPs")

    lines.append(f"\n  Summary:")
    full = flops_result.get("full", {})
    cached = flops_result.get("cached", {})
    saved = flops_result.get("saved", {})
    lines.append(f"    Full model:       {full.get('total_gflops', 0):10.2f} GFLOPs  "
                 f"({full.get('total_tflops', 0):.4f} TFLOPs)")
    if saved.get("total_gflops", 0) > 0:
        lines.append(f"    Cached model:     {cached.get('total_gflops', 0):10.2f} GFLOPs  "
                     f"({cached.get('total_tflops', 0):.4f} TFLOPs)")
        lines.append(f"    FLOPs saved:      {saved.get('total_gflops', 0):10.2f} GFLOPs  "
                     f"({saved.get('total_tflops', 0):.4f} TFLOPs, {saved.get('pct', 0):.1f}%)")

    lines.append("=" * 70)
    return "\n".join(lines)
