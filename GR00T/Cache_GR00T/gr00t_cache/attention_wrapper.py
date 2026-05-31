# SPDX-FileCopyrightText: Copyright (c) 2025 GR00T-Cache Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Safe attention wrapper for GR00T-Cache KV reuse.

Design principles:
1. PASSTHROUGH FIRST — when cache is disabled or shapes don't match, call the
   original attention module with ZERO overhead (no projection, no SDPA).
2. Only intercept when ALL conditions are met:
   - Cache is enabled
   - Cached K/V exists and shapes match
   - Reuse plan has tokens to reuse
3. For the DiT action head: cache condition K/V WITHIN a single get_action call
   (across denoising steps), NOT across timesteps (safer, no shape issues).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache_manager import GR00TCacheManager
from .config import GR00TCacheConfig


class CachedAttentionWrapper(nn.Module):
    """Wraps attention module. Only intercepts when cache can actually help."""

    def __init__(
        self,
        original_attention: nn.Module,
        cache_manager: GR00TCacheManager,
        layer_idx: int,
        attention_type: str = "self_attn",
        visual_token_indices: Optional[torch.Tensor] = None,
        is_condition_side: bool = False,
        collect_attention_maps: bool = True,
        fallback_sdpa: bool = True,
    ):
        super().__init__()
        self.original_attention = original_attention
        self.cache_manager = cache_manager
        self.layer_idx = layer_idx
        self.attention_type = attention_type
        self.visual_token_indices = visual_token_indices
        self.is_condition_side = is_condition_side
        self.collect_attention_maps = collect_attention_maps
        self.fallback_sdpa = fallback_sdpa

        # Detect type
        self._is_hf = hasattr(original_attention, "q_proj") and hasattr(original_attention, "k_proj")
        self._is_diffusers = hasattr(original_attention, "to_q") and hasattr(original_attention, "to_k")

        # Stored state
        self._stored_k: Optional[torch.Tensor] = None
        self._stored_v: Optional[torch.Tensor] = None
        self._last_attn_weights: Optional[torch.Tensor] = None

        # Per-step condition cache for DiT (within a single get_action call)
        self._step_condition_k: Optional[torch.Tensor] = None
        self._step_condition_v: Optional[torch.Tensor] = None
        self._step_condition_key: Optional[int] = None  # step id for invalidation

    def _can_cache(self) -> bool:
        """Check if we should intercept this forward call."""
        cfg = self.cache_manager.config
        if not cfg.enabled:
            return False

        plan = self.cache_manager._current_reuse_plan
        if plan is None or not plan.get("should_cache", False):
            return False

        return True

    def _kvs_match(self, cached_kv: dict) -> bool:
        """Check that stored K/V shapes match cached K/V."""
        if self._stored_k is None or self._stored_v is None:
            return False
        cached_k = cached_kv.get("k")
        cached_v = cached_kv.get("v")
        if cached_k is None or cached_v is None:
            return False
        return (
            cached_k.shape == self._stored_k.shape
            and cached_v.shape == self._stored_v.shape
        )

    # ── Main forward ─────────────────────────────────────────────────

    def forward(self, *args, **kwargs) -> Any:
        """Passthrough-first: only intercept when cache is useful.

        When cache conditions aren't met, the original attention runs unchanged.
        """
        if not self._can_cache():
            return self.original_attention(*args, **kwargs)

        # Check for cached KV
        if self._is_hf:
            return self._forward_hf_maybe_cached(*args, **kwargs)
        elif self._is_diffusers:
            return self._forward_diffusers_maybe_cached(*args, **kwargs)

        return self.original_attention(*args, **kwargs)

    # ── HF-style (backbone LLM) ─────────────────────────────────────

    def _forward_hf_maybe_cached(self, *args, **kwargs) -> Any:
        """Try to use cached KV; fall back to original if unsafe."""
        attn = self.original_attention

        if args:
            hidden_states = args[0]
        else:
            hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            return attn(*args, **kwargs)

        # Check shape consistency
        cached_kv = self.cache_manager.get_backbone_kv(self.layer_idx)
        if cached_kv is None or not self._kvs_match(cached_kv):
            # Shapes don't match — use original attention (flash/SDPA)
            result = attn(*args, **kwargs)
            # Still store current K/V for potential next step
            self._store_current_hf_kv(hidden_states)
            return result

        # Shapes match — we can try cache
        B, T, C = hidden_states.shape
        n_heads = self._get_num_heads(attn)
        head_dim = C // n_heads if n_heads else 64
        n_kv_heads = getattr(attn, "num_key_value_heads", n_heads)

        # Compute Q, K, V
        q = attn.q_proj(hidden_states).view(B, T, n_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(hidden_states).view(B, T, n_kv_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(B, T, n_kv_heads, head_dim).transpose(1, 2)

        if n_kv_heads < n_heads:
            k = k.repeat_interleave(n_heads // n_kv_heads, dim=1)
            v = v.repeat_interleave(n_heads // n_kv_heads, dim=1)

        # Apply cached K/V for static visual tokens
        reuse_plan = self.cache_manager._current_reuse_plan
        reuse_masks = reuse_plan.get("reuse_per_layer", {}).get(
            self.layer_idx, reuse_plan.get("reuse_per_layer", {}).get(0, {})
        )
        if reuse_masks and self.visual_token_indices is not None:
            k, v = self._apply_reuse_masks(k, v, cached_kv, reuse_masks)

        # Store for next step
        self._stored_k = k.detach()
        self._stored_v = v.detach()

        # Attention (SDPA — flash can't do partial KV)
        attention_mask = kwargs.get("attention_mask", None)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask,
            dropout_p=0.0, is_causal=(attention_mask is None),
        )

        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, T, C)
        attn_out = attn.o_proj(attn_out)

        output_attentions = kwargs.get("output_attentions", False)
        if output_attentions:
            return attn_out, self._last_attn_weights, kwargs.get("past_key_value", None)
        return attn_out, None, kwargs.get("past_key_value", None)

    def _store_current_hf_kv(self, hidden_states: torch.Tensor) -> None:
        """Store current K/V for next step."""
        try:
            attn = self.original_attention
            C = hidden_states.shape[-1]
            n_heads = self._get_num_heads(attn)
            head_dim = C // n_heads if n_heads else 64
            n_kv_heads = getattr(attn, "num_key_value_heads", n_heads)
            B, T = hidden_states.shape[0], hidden_states.shape[1]

            k = attn.k_proj(hidden_states).view(B, T, n_kv_heads, head_dim).transpose(1, 2)
            v = attn.v_proj(hidden_states).view(B, T, n_kv_heads, head_dim).transpose(1, 2)
            if n_kv_heads < n_heads:
                k = k.repeat_interleave(n_heads // n_kv_heads, dim=1)
                v = v.repeat_interleave(n_heads // n_kv_heads, dim=1)

            self._stored_k = k.detach()
            self._stored_v = v.detach()
        except Exception:
            pass  # Silently ignore — best-effort

    # ── Diffusers-style (DiT) ───────────────────────────────────────

    def _forward_diffusers_maybe_cached(self, *args, **kwargs) -> Any:
        """DiT attention: cache condition K/V within a single get_action call.

        The DiT runs N denoising steps per get_action call. The condition
        (encoder_hidden_states) is IDENTICAL across all steps. We cache
        the K/V projection of the condition side after the first step.
        """
        attn = self.original_attention

        if args:
            hidden_states = args[0]
        else:
            hidden_states = kwargs.get("hidden_states")
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
        if encoder_hidden_states is None and len(args) > 1:
            encoder_hidden_states = args[1]

        is_cross_attn = encoder_hidden_states is not None

        if not is_cross_attn:
            # Self-attention on action latents — never cache, always passthrough
            return attn(*args, **kwargs)

        # Cross-attention: try to use per-step cached condition K/V
        current_step = getattr(self.cache_manager, "cache_step_id", -1)

        if (
            self._step_condition_k is not None
            and self._step_condition_v is not None
            and self._step_condition_key == current_step
        ):
            # Reuse cached condition K/V from earlier denoising step
            k = self._step_condition_k
            v = self._step_condition_v
            B, T_q, C = hidden_states.shape
            q = attn.to_q(hidden_states)
            q_head = attn.head_to_batch_dim(q) if hasattr(attn, "head_to_batch_dim") else q

            scale = getattr(attn, "scale", 1.0)
            attn_out = F.scaled_dot_product_attention(
                q_head, k, v, dropout_p=0.0, scale=scale,
            )

            if hasattr(attn, "batch_to_head_dim"):
                attn_out = attn.batch_to_head_dim(attn_out)
            else:
                H = self._get_num_heads(attn)
                attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, T_q, C)

            to_out = getattr(attn, "to_out", None)
            if to_out is not None:
                if isinstance(to_out, nn.ModuleList):
                    for layer in to_out:
                        attn_out = layer(attn_out)
                else:
                    attn_out = to_out(attn_out)
            return attn_out

        # First denoising step — compute normally, then cache
        result = attn(*args, **kwargs)

        # Store condition K/V for remaining denoising steps
        try:
            k_raw = attn.to_k(encoder_hidden_states)
            v_raw = attn.to_v(encoder_hidden_states)
            if hasattr(attn, "head_to_batch_dim"):
                self._step_condition_k = attn.head_to_batch_dim(k_raw).detach()
                self._step_condition_v = attn.head_to_batch_dim(v_raw).detach()
            else:
                self._step_condition_k = k_raw.detach()
                self._step_condition_v = v_raw.detach()
            self._step_condition_key = current_step
        except Exception:
            pass

        return result

    # ── Helpers ─────────────────────────────────────────────────────

    def _get_num_heads(self, attn) -> int:
        for attr in ("heads", "num_heads", "num_attention_heads"):
            val = getattr(attn, attr, None)
            if val is not None:
                return val
        # Infer from q_proj weight
        proj = getattr(attn, "q_proj", None) or getattr(attn, "to_q", None)
        if proj is not None and hasattr(proj, "weight"):
            out_features = proj.weight.shape[0]
            for d in (128, 64, 96, 80, 32):
                if out_features % d == 0 and out_features // d <= 64:
                    return out_features // d
            # Guess: common values
            if out_features == 2048:
                return 16
            if out_features == 4096:
                return 32
        return 8

    def _apply_reuse_masks(
        self, k, v, cached_kv, reuse_masks,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached_k = cached_kv["k"]
        cached_v = cached_kv["v"]
        if cached_k.shape != k.shape:
            return k, v

        vis_idx = self.visual_token_indices
        if vis_idx is None:
            return k, v

        n_vis = len(vis_idx)
        reuse_mask = torch.zeros(min(n_vis, k.shape[2]), dtype=torch.bool, device=k.device)
        offset = 0
        for view_mask in reuse_masks.values():
            n = len(view_mask)
            end = min(offset + n, len(reuse_mask))
            reuse_mask[offset:end] = view_mask[:end - offset].to(k.device)
            offset += n

        seq_mask = torch.zeros(k.shape[2], dtype=torch.bool, device=k.device)
        vis_idx_dev = vis_idx.to(k.device)
        valid_len = min(len(reuse_mask), len(vis_idx_dev))
        seq_mask[vis_idx_dev[:valid_len]] = reuse_mask[:valid_len]

        mask_k = seq_mask[None, None, :, None].expand_as(k)
        k = torch.where(mask_k, cached_k.to(k.device), k)
        v = torch.where(mask_k, cached_v.to(v.device), v)
        return k, v

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        return self._last_attn_weights

    def store_backbone_kv(self, layer_idx: int) -> None:
        if self._stored_k is not None and self._stored_v is not None:
            self.cache_manager.layer_kv_cache[layer_idx] = {
                "k": self._stored_k.detach().clone(),
                "v": self._stored_v.detach().clone(),
            }

    def store_condition_kv(self, layer_idx: int) -> None:
        if self._step_condition_k is not None and self._step_condition_v is not None:
            self.cache_manager.action_head_condition_kv[layer_idx] = {
                "k": self._step_condition_k.detach().clone(),
                "v": self._step_condition_v.detach().clone(),
            }


# ── Apply / Remove ──────────────────────────────────────────────────

def apply_cache_to_backbone(model, cache_manager, token_index_map, config):
    wrappers = {}
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return wrappers

    eagle_model = getattr(backbone, "eagle_model", backbone)
    language_model = getattr(eagle_model, "language_model", eagle_model)
    model_obj = getattr(language_model, "model", language_model)
    layers = getattr(model_obj, "layers", None)
    if layers is None:
        layers = getattr(backbone, "layers", None)

    if layers is None:
        return wrappers

    vis_idx = token_index_map.visual_indices if token_index_map else None
    start, end = config.layer_start, config.layer_end or len(layers)

    for idx in range(start, min(end, len(layers))):
        layer = layers[idx]
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue

        wrapper = CachedAttentionWrapper(
            original_attention=attn,
            cache_manager=cache_manager,
            layer_idx=idx,
            attention_type="self_attn",
            visual_token_indices=vis_idx,
            collect_attention_maps=config.collect_attention_maps,
            fallback_sdpa=config.fallback_sdpa,
        )
        layer.self_attn = wrapper
        wrappers[f"backbone.layer_{idx}.self_attn"] = wrapper

    return wrappers


def apply_cache_to_action_head(model, cache_manager, config):
    wrappers = {}
    action_head = getattr(model, "action_head", None)
    if action_head is None:
        return wrappers

    dit = getattr(action_head, "model", action_head)
    blocks = getattr(dit, "transformer_blocks", None)
    if blocks is None:
        blocks = getattr(dit, "blocks", None)
    if blocks is None:
        return wrappers

    for idx, block in enumerate(blocks):
        attn = getattr(block, "attn1", None)
        if attn is None:
            continue
        wrapper = CachedAttentionWrapper(
            original_attention=attn,
            cache_manager=cache_manager,
            layer_idx=idx,
            attention_type="cross_attn",
            collect_attention_maps=config.collect_attention_maps,
            fallback_sdpa=config.fallback_sdpa,
        )
        block.attn1 = wrapper
        wrappers[f"action_head.block_{idx}.attn1"] = wrapper

    return wrappers


def remove_cache_from_model(model, wrappers):
    for path, wrapper in wrappers.items():
        original = wrapper.original_attention
        parts = path.split(".")
        if path.startswith("backbone.layer_"):
            layer_idx = int(parts[1].split("_")[-1])
            backbone = model.backbone
            eagle = getattr(backbone, "eagle_model", backbone)
            lm = getattr(eagle, "language_model", eagle)
            m = getattr(lm, "model", lm)
            layers = getattr(m, "layers", getattr(backbone, "layers", None))
            if layers is not None and layer_idx < len(layers):
                layers[layer_idx].self_attn = original
        elif path.startswith("action_head.block_"):
            block_idx = int(parts[1].split("_")[-1])
            dit = (
                model.action_head.model
                if hasattr(model.action_head, "model")
                else model.action_head
            )
            blocks = getattr(dit, "transformer_blocks", getattr(dit, "blocks", None))
            if blocks is not None and block_idx < len(blocks):
                blocks[block_idx].attn1 = original
