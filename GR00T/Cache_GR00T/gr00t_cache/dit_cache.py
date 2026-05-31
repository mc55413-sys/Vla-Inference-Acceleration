# SPDX-FileCopyrightText: Copyright (c) 2025 GR00T-Cache Contributors
# SPDX-License-Identifier: Apache-2.0
"""
DiT Block Cache — reuses block outputs across denoising steps.

Based on ideas from:
  - DeepCache (CVPR 2024): caches layer features in diffusion models
  - ToCa (arXiv 2024): token caching based on maturity scores
  - Δ-DiT (arXiv 2024): delta computation between timesteps

Strategy for GR00T DiT (16 blocks, 8 denoising steps):
  - Steps 0-1: compute all blocks (warmup, learn cache)
  - Steps 2-7: for each block, check if input is similar to cached input;
    if yes (cosine sim > threshold), reuse cached output → skip block entirely
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cosine_sim_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fast cosine similarity between flattened tensors — auto-handles device."""
    if a.device != b.device:
        b = b.to(a.device)
    a_f = a.float().ravel()
    b_f = b.float().ravel()
    dot = (a_f * b_f).sum()
    return (dot / (a_f.norm() * b_f.norm() + 1e-8)).item()


class DiTBlockCache:
    """Per-block cache: stores (input, output) and condition K/V.

    On each denoising step:
      1. Compute cosine similarity between current input and cached input
      2. If similar enough → return cached output (skip block)
      3. Otherwise → compute block, update cache
    """

    def __init__(
        self,
        block_idx: int,
        is_self_attn: bool,
        sim_threshold: float = 0.995,
        min_skip_steps: int = 0,
    ):
        self.block_idx = block_idx
        self.is_self_attn = is_self_attn
        self.sim_threshold = sim_threshold
        self.min_skip_steps = min_skip_steps

        self.cached_input: Optional[torch.Tensor] = None
        self.cached_output: Optional[torch.Tensor] = None
        self.cached_encoder_input: Optional[torch.Tensor] = None

        # Stats
        self.hits = 0
        self.misses = 0
        self.total_calls = 0

    def should_reuse(
        self, hidden_states: torch.Tensor, step: int, encoder_hs: Optional[torch.Tensor] = None
    ) -> bool:
        """Check if cached output can be reused."""
        if step < self.min_skip_steps:
            return False
        if self.cached_input is None or self.cached_output is None:
            return False
        # Ensure cached tensors are on the same device
        if self.cached_input.device != hidden_states.device:
            self.cached_input = self.cached_input.to(hidden_states.device)
            self.cached_output = self.cached_output.to(hidden_states.device)
            if self.cached_encoder_input is not None:
                self.cached_encoder_input = self.cached_encoder_input.to(hidden_states.device)

        # Check input similarity
        input_sim = _cosine_sim_flat(hidden_states, self.cached_input)
        if input_sim < self.sim_threshold:
            return False

        # For cross-attn blocks, also check encoder input
        if not self.is_self_attn and encoder_hs is not None and self.cached_encoder_input is not None:
            enc_sim = _cosine_sim_flat(encoder_hs, self.cached_encoder_input)
            if enc_sim < 0.999:
                return False

        return True

    def forward(
        self,
        block: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> torch.Tensor:
        """Forward through block, with optional cache reuse."""
        self.total_calls += 1

        if self.should_reuse(hidden_states, step, encoder_hidden_states):
            self.hits += 1
            return self.cached_output

        self.misses += 1

        # Compute block
        output = block(hidden_states, encoder_hidden_states=encoder_hidden_states)

        # Update cache (keep on same device)
        self.cached_input = hidden_states.detach()
        self.cached_output = output.detach()
        if encoder_hidden_states is not None:
            self.cached_encoder_input = encoder_hidden_states.detach()

        return output

    def hit_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.hits / self.total_calls

    def stats(self) -> dict[str, Any]:
        return {
            "block_idx": self.block_idx,
            "is_self_attn": self.is_self_attn,
            "hits": self.hits,
            "misses": self.misses,
            "total": self.total_calls,
            "hit_rate": self.hit_rate(),
        }


class DiTStepCacheManager:
    """Manages block-level caching for the entire DiT across denoising steps.

    Usage:
      cache_mgr = DiTStepCacheManager(dit, num_steps=8)
      for step in range(num_steps):
          action_features = ...
          query = cat(state, future, action_features)
          output = cache_mgr.forward(query, encoder_hs=condition, step=step)
    """

    def __init__(
        self,
        dit,  # DiT module (model.action_head.model)
        num_steps: int = 8,
        sim_threshold: float = 0.995,
        warmup_steps: int = 2,
        cache_self_attn: bool = True,
        cache_cross_attn: bool = True,
        early_blocks: int = 4,  # first N blocks always computed (early layers matter more)
    ):
        self.dit = dit
        self.num_steps = num_steps
        self.sim_threshold = sim_threshold
        self.warmup_steps = warmup_steps
        self.early_blocks = early_blocks

        blocks = dit.transformer_blocks
        interleave = getattr(dit.config, "interleave_self_attention", True)
        self.num_blocks = len(blocks)

        self.block_caches = []
        for i in range(self.num_blocks):
            is_self_attn = interleave and (i % 2 == 1)
            enabled = (is_self_attn and cache_self_attn) or (not is_self_attn and cache_cross_attn)
            # Early blocks never cached (they capture important changes)
            enabled = enabled and (i >= early_blocks)

            self.block_caches.append(
                DiTBlockCache(
                    block_idx=i,
                    is_self_attn=is_self_attn,
                    sim_threshold=sim_threshold,
                    min_skip_steps=warmup_steps if enabled else 999,
                )
            )

        # Stats
        self.total_saved_ms = 0.0
        self.total_compute_ms = 0.0
        self.step_count = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        step: int,
        return_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        """Forward through DiT with block-level caching.

        Args:
            hidden_states: [B, query_len, D] — query tokens
            encoder_hidden_states: [B, cond_len, D] — condition tokens
            timestep: [B] — timestep indices
            step: current denoising step [0, num_steps)
            return_all_hidden_states: passed to DiT.forward

        Returns:
            DiT output [B, query_len, output_dim]
        """
        self.step_count += 1

        # Encode timestep
        temb = self.dit.timestep_encoder(timestep)
        device = hidden_states.device
        temb = temb.to(device=device, dtype=hidden_states.dtype)

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]

        for i, block in enumerate(self.dit.transformer_blocks):
            is_self_attn = (
                i % 2 == 1 and getattr(self.dit.config, "interleave_self_attention", True)
            )

            if is_self_attn:
                # Self-attention block
                hidden_states = self._cached_block_forward(
                    i, block, hidden_states,
                    encoder_hidden_states=None,
                    temb=temb,
                    step=step,
                )
            else:
                # Cross-attention block
                hidden_states = self._cached_block_forward(
                    i, block, hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    step=step,
                )
            all_hidden_states.append(hidden_states)

        # Output processing
        conditioning = temb
        shift, scale = self.dit.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.dit.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]

        if return_all_hidden_states:
            return self.dit.proj_out_2(hidden_states), all_hidden_states
        return self.dit.proj_out_2(hidden_states)

    def _cached_block_forward(
        self,
        block_idx: int,
        block: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        temb: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Forward through a single block with caching."""
        block_cache = self.block_caches[block_idx]

        if block_cache.min_skip_steps <= step:
            # Check if we can reuse
            if block_cache.should_reuse(hidden_states, step, encoder_hidden_states):
                out = block_cache.cached_output
                if out.device != hidden_states.device:
                    out = out.to(hidden_states.device)
                return out

        # Compute normally
        output = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
        )

        # Update cache for next step
        if step >= self.warmup_steps - 1:  # only cache after warmup
            block_cache.cached_input = hidden_states.detach()
            block_cache.cached_output = output.detach()
            if encoder_hidden_states is not None:
                block_cache.cached_encoder_input = encoder_hidden_states.detach()

        return output

    def reset(self):
        """Reset caches between episodes."""
        for bc in self.block_caches:
            bc.cached_input = None
            bc.cached_output = None
            bc.cached_encoder_input = None
            bc.hits = 0
            bc.misses = 0
            bc.total_calls = 0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total_hits = sum(bc.hits for bc in self.block_caches)
        total_misses = sum(bc.misses for bc in self.block_caches)
        return {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_calls": total_hits + total_misses,
            "overall_hit_rate": total_hits / (total_hits + total_misses + 1e-8),
            "per_block": [bc.stats() for bc in self.block_caches],
        }


# ── Hook into FlowmatchingActionHead ─────────────────────────────────

def apply_dit_cache_to_action_head(
    action_head,
    num_steps: int = 8,
    sim_threshold: float = 0.995,
    warmup_steps: int = 2,
    cache_self_attn: bool = True,
    cache_cross_attn: bool = True,
    early_blocks: int = 4,
) -> DiTStepCacheManager:
    """Wrap the DiT inside the action head with block-level caching.

    Monkey-patches action_head.get_action to use cached DiT forward.
    """
    dit = action_head.model
    cache_mgr = DiTStepCacheManager(
        dit,
        num_steps=num_steps,
        sim_threshold=sim_threshold,
        warmup_steps=warmup_steps,
        cache_self_attn=cache_self_attn,
        cache_cross_attn=cache_cross_attn,
        early_blocks=early_blocks,
    )

    # Store original get_action
    original_get_action = action_head.get_action

    @torch.no_grad()
    def cached_get_action(backbone_output, action_input):
        """Cached version of get_action."""
        action_head.process_backbone_output(backbone_output)
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        state_features = action_head.state_encoder(action_input.state, embodiment_id)

        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, action_head.config.action_horizon, action_head.config.action_dim),
            dtype=vl_embs.dtype, device=device,
        )

        dt = 1.0 / num_steps
        future_tokens = action_head.future_tokens.weight.unsqueeze(0).expand(
            vl_embs.shape[0], -1, -1
        )

        for step in range(num_steps):
            t_discretized = int(step / num_steps * action_head.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = action_head.action_encoder(
                actions, timesteps_tensor, embodiment_id
            )
            if action_head.config.add_pos_embed:
                pos_ids = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=device
                )
                pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Use cached DiT forward
            model_output = cache_mgr.forward(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
                step=step,
            )

            pred = action_head.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -action_head.action_horizon:]
            actions = actions + dt * pred_velocity

        # Reset cache for next get_action call
        cache_mgr.reset()

        from transformers.feature_extraction_utils import BatchFeature
        return BatchFeature(data={"action_pred": actions})

    # Apply monkey patch
    action_head._original_get_action = original_get_action
    action_head.get_action = cached_get_action
    action_head._dit_cache_mgr = cache_mgr

    return cache_mgr


def remove_dit_cache_from_action_head(action_head):
    """Restore original get_action."""
    if hasattr(action_head, "_original_get_action"):
        action_head.get_action = action_head._original_get_action
        del action_head._original_get_action
    if hasattr(action_head, "_dit_cache_mgr"):
        del action_head._dit_cache_mgr
