# SPDX-FileCopyrightText: Copyright (c) 2025 GR00T-Cache Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Dummy GR00T-like model for testing GR00T-Cache without real weights.

Architecture (mimics GR00T N1.5):
  1. Vision encoder: Conv2d patch embed + LayerNorm
  2. Vision→LLM projector: nn.Linear
  3. Backbone LLM: text_embed + N×TransformerLayer (self-attn + FFN)
  4. Backbone→ActionHead projector: nn.Linear
  5. Action head: DiT with cross-attention to vision-language features
  6. Flow-matching denoising loop

All dimensions are consistent and configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DummyGR00TConfig:
    """Configuration for the dummy GR00T model.

    All dimension fields must be consistent:
      vision_hidden_size → projector → backbone_hidden_size
      backbone_hidden_size → backbone_projector → action_head_condition_dim
    """
    # ── Vision ──
    image_size: int = 224
    patch_size: int = 16
    vision_hidden_size: int = 768
    n_views: int = 2  # external + wrist

    # ── Backbone (VLM LLM) ──
    backbone_num_layers: int = 6
    backbone_hidden_size: int = 512
    backbone_num_heads: int = 8
    backbone_ffn_dim: int = 2048
    vocab_size: int = 1000  # dummy vocab

    # ── Projector (vision → backbone) ──
    projector_hidden_size: int = 512  # vision output → backbone input

    # ── Action head condition dimension (backbone output) ──
    condition_dim: int = 512

    # ── Action head (DiT) ──
    dit_num_layers: int = 4
    dit_hidden_size: int = 256
    dit_num_heads: int = 8
    dit_ffn_dim: int = 1024

    # ── Text ──
    text_tokens: int = 50

    # ── Action ──
    action_horizon: int = 16
    action_dim: int = 7
    state_dim: int = 7
    num_inference_timesteps: int = 4
    num_future_tokens: int = 8  # learned query tokens (reduced for dummy)

    # ── Device / dtype ──
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # ── Derived ──
    @property
    def n_patches_per_view(self) -> int:
        return (self.image_size // self.patch_size) ** 2


# ---------------------------------------------------------------------------
# Vision modules
# ---------------------------------------------------------------------------

class DummyVisionEncoder(nn.Module):
    """Simple conv-based patch embedder, like a ViT patch projection."""
    def __init__(self, config: DummyGR00TConfig):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            3, config.vision_hidden_size,
            kernel_size=config.patch_size, stride=config.patch_size,
        )
        self.norm = nn.LayerNorm(config.vision_hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [total_views, 3, H, W]
        x = self.patch_embed(pixel_values)          # [V_total, D_vis, h, w]
        x = x.flatten(2).transpose(1, 2)            # [V_total, n_patches, D_vis]
        x = self.norm(x)
        return x


class DummyVisionProjector(nn.Module):
    """Linear projector: vision_hidden_size → projector_hidden_size."""
    def __init__(self, config: DummyGR00TConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_hidden_size, config.projector_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Backbone (VLM LLM) modules
# ---------------------------------------------------------------------------

class DummyLLMSelfAttention(nn.Module):
    """Multi-head self-attention matching HuggingFace-style API.

    Uses separate q_proj/k_proj/v_proj/o_proj so that CachedAttentionWrapper
    can intercept them.
    """
    def __init__(self, config: DummyGR00TConfig):
        super().__init__()
        self.hidden_size = config.backbone_hidden_size
        self.num_heads = config.backbone_num_heads
        self.head_dim = config.backbone_hidden_size // config.backbone_num_heads
        self.num_key_value_heads = config.backbone_num_heads  # no GQA in dummy

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        B, T, C = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # SDPA
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=True if attention_mask is None else False,
        )

        # Compute attention weights for collection
        attn_weights = None
        if output_attentions:
            scale = 1.0 / (self.head_dim ** 0.5)
            attn_weights = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, T, C)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class DummyLLMDecoderLayer(nn.Module):
    """Single decoder layer with self-attention + FFN, HF-compatible layout."""
    def __init__(self, config: DummyGR00TConfig):
        super().__init__()
        self.self_attn = DummyLLMSelfAttention(config)
        self.input_layernorm = nn.LayerNorm(config.backbone_hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.backbone_hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.backbone_hidden_size, config.backbone_ffn_dim),
            nn.GELU(),
            nn.Linear(config.backbone_ffn_dim, config.backbone_hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_result = self.self_attn(
            hidden_states, attention_mask, position_ids,
            past_key_value, output_attentions, use_cache,
        )
        # Handle both tuple and tensor return types (wrapper may return tuple or tensor)
        if isinstance(attn_result, tuple):
            attn_out, attn_weights, _ = attn_result
        else:
            attn_out, attn_weights = attn_result, None
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, attn_weights, None


class DummyBackbone(nn.Module):
    """VLM backbone: vision encoder + projector + LLM."""
    def __init__(self, config: DummyGR00TConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = DummyVisionEncoder(config)
        self.vision_projector = DummyVisionProjector(config)
        self.text_embed = nn.Embedding(config.vocab_size, config.backbone_hidden_size)
        self.layers = nn.ModuleList([
            DummyLLMDecoderLayer(config)
            for _ in range(config.backbone_num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.backbone_hidden_size)
        # Project backbone output to condition dimension for DiT
        self.to_condition = nn.Linear(config.backbone_hidden_size, config.condition_dim)

    @property
    def eagle_model(self):
        """Compatibility shim — returns self so that apply_cache_to_backbone finds layers."""
        return self

    @property
    def language_model(self):
        """Compatibility shim."""
        return self

    @property
    def model(self):
        """Compatibility shim for HuggingFace-style access."""
        return self

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
    ) -> dict[str, torch.Tensor]:
        B = input_ids.shape[0]
        V_total = pixel_values.shape[0]
        n_views = V_total // B

        # ── Vision ──
        vis = self.vision_encoder(pixel_values)                    # [V_total, P, D_vis]
        vis = self.vision_projector(vis)                           # [V_total, P, D_proj]
        P = vis.shape[1]
        vis = vis.reshape(B, n_views * P, self.config.projector_hidden_size)  # [B, n_views*P, D_proj]

        # ── Project visual to backbone dim ──
        if self.config.projector_hidden_size != self.config.backbone_hidden_size:
            vis = F.linear(
                vis,
                torch.eye(
                    self.config.projector_hidden_size,
                    self.config.backbone_hidden_size,
                    device=vis.device, dtype=vis.dtype,
                )[:self.config.projector_hidden_size],
            )

        # ── Text ──
        text = self.text_embed(input_ids)                          # [B, T_text, D_backbone]

        # ── Merge ──
        seq = torch.cat([text, vis], dim=1)                        # [B, T_text + V_tokens, D_backbone]

        # ── Transformer layers ──
        hidden_states = [seq]
        h = seq
        for layer in self.layers:
            h, _, _ = layer(h)
            hidden_states.append(h)

        h = self.final_norm(h)
        condition = self.to_condition(h)                           # [B, total_tokens, D_cond]

        return {
            "backbone_features": condition,
            "backbone_attention_mask": attention_mask,
            "hidden_states": hidden_states,
        }


# ---------------------------------------------------------------------------
# Action head (DiT) modules
# ---------------------------------------------------------------------------

class DummyDiTAttention(nn.Module):
    """Diffusers-style Attention for DiT blocks.

    Uses to_q/to_k/to_v/to_out so CachedAttentionWrapper can hook in.
    When encoder_hidden_states is passed, does cross-attention.
    When None, does self-attention.
    """
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        cross_attention_dim: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim or query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim or query_dim, inner_dim, bias=bias)

        self.to_out = nn.ModuleList([
            nn.Linear(inner_dim, query_dim, bias=bias),
            nn.Dropout(0.0),
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T_q, C = hidden_states.shape
        T_kv = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else T_q

        q = self.to_q(hidden_states)
        kv_input = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        # head_to_batch_dim
        q = q.view(B, T_q, self.heads, self.head_dim).transpose(1, 2)  # [B, heads, T_q, head_dim]
        k = k.view(B, T_kv, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_kv, self.heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale)

        # batch_to_head_dim
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, T_q, C)
        attn_out = self.to_out[0](attn_out)
        attn_out = self.to_out[1](attn_out)
        return attn_out

    def head_to_batch_dim(self, t: torch.Tensor) -> torch.Tensor:
        B, T, C = t.shape
        return t.view(B, T, self.heads, self.head_dim).transpose(1, 2)

    def batch_to_head_dim(self, t: torch.Tensor) -> torch.Tensor:
        B, H, T, D = t.shape
        return t.transpose(1, 2).contiguous().reshape(B, T, H * D)


class DummyDiTBlock(nn.Module):
    """DiT transformer block with cross-attention.

    Mirrors the real GR00T's BasicTransformerBlock structure:
      - attn1: self/cross-attention (depending on encoder_hidden_states)
      - ff: feed-forward
    """
    def __init__(self, config: DummyGR00TConfig, cross_attention_dim: Optional[int] = None):
        super().__init__()
        self.attn1 = DummyDiTAttention(
            query_dim=config.dit_hidden_size,
            heads=config.dit_num_heads,
            dim_head=config.dit_hidden_size // config.dit_num_heads,
            cross_attention_dim=cross_attention_dim,
        )
        self.norm1 = nn.LayerNorm(config.dit_hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(config.dit_hidden_size, config.dit_ffn_dim),
            nn.GELU(),
            nn.Linear(config.dit_ffn_dim, config.dit_hidden_size),
        )
        self.norm2 = nn.LayerNorm(config.dit_hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        x = self.norm1(hidden_states)
        x = self.attn1(x, encoder_hidden_states=encoder_hidden_states)
        hidden_states = residual + x

        residual = hidden_states
        x = self.norm2(hidden_states)
        x = residual + self.ff(x)
        return x


class DummyActionHead(nn.Module):
    """Flow-matching action head with DiT.

    Mimics the GR00T FlowmatchingActionHead:
      1. Encode state, action, timestep
      2. Add learned future/query tokens
      3. Run DiT transformer blocks (cross-attention to backbone features)
      4. Decode to action space
    """
    def __init__(self, config: DummyGR00TConfig):
        super().__init__()
        self.config = config

        # Encoders
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.dit_hidden_size),
            nn.ReLU(),
            nn.Linear(config.dit_hidden_size, config.dit_hidden_size),
        )
        self.action_embed = nn.Linear(config.action_dim, config.dit_hidden_size)
        self.time_embed = nn.Embedding(1000, config.dit_hidden_size)

        # Learned query tokens (like GR00T's future_tokens)
        self.future_tokens = nn.Embedding(config.num_future_tokens, config.dit_hidden_size)

        # Position embedding for action sequence
        self.pos_embed = nn.Embedding(config.action_horizon, config.dit_hidden_size)

        # DiT blocks (cross-attend to condition features)
        self.blocks = nn.ModuleList([
            DummyDiTBlock(config, cross_attention_dim=config.condition_dim)
            for _ in range(config.dit_num_layers)
        ])

        # Output
        self.norm_out = nn.LayerNorm(config.dit_hidden_size)
        self.decoder = nn.Linear(config.dit_hidden_size, config.action_dim)

        # NOTE: Do NOT set self.model = self — it creates a reference cycle
        # that makes nn.Module.to() recurse infinitely.
        # Instead, CachedAttentionWrapper finds blocks via action_head.model.transformer_blocks
        # or falls back to action_head.blocks.

    @property
    def transformer_blocks(self):
        return self.blocks

    def forward(
        self,
        backbone_features: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        B = backbone_features.shape[0]
        device = backbone_features.device
        T_a = actions.shape[1]

        # Encode
        state_emb = self.state_encoder(state).unsqueeze(1)                # [B, 1, D_dit]
        time_emb = self.time_embed(timestep).unsqueeze(1)                 # [B, 1, D_dit]
        action_emb = self.action_embed(actions)                           # [B, T_a, D_dit]

        # Add position embeddings to actions
        pos_ids = torch.arange(T_a, device=device).unsqueeze(0).expand(B, -1)
        action_emb = action_emb + self.pos_embed(pos_ids)

        # Future/query tokens
        future = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)  # [B, N_future, D_dit]

        # Query sequence: [state, future, time, actions]
        query = torch.cat([state_emb, future, time_emb, action_emb], dim=1)

        # DiT blocks
        for block in self.blocks:
            query = block(query, encoder_hidden_states=backbone_features)

        # Decode
        query = self.norm_out(query)
        pred = self.decoder(query)

        # Extract action portion
        pred_actions = pred[:, -T_a:]
        return pred_actions


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class DummyGR00TModel(nn.Module):
    """Self-contained dummy GR00T N1.5 model.

    Pipeline:
        images → vision encoder → projector → [backbone LLM] → condition features
        state/action → encoders → [DiT × denoising steps] → action

    For testing GR00T-Cache:
      - Backbone layers use DummyLLMSelfAttention (q_proj/k_proj/v_proj/o_proj)
        which CachedAttentionWrapper can hook into.
      - DiT blocks use DummyDiTAttention (to_q/to_k/to_v/to_out)
        which CachedAttentionWrapper can also hook into.
    """

    def __init__(self, config: Optional[DummyGR00TConfig] = None):
        super().__init__()
        self.config = config or DummyGR00TConfig()
        self.backbone = DummyBackbone(self.config)
        self.action_head = DummyActionHead(self.config)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device(self.config.device)

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return self.config.dtype

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Full forward pass with denoising loop.

        Args:
            obs: dict with:
                "pixel_values": [B*V, 3, H, W] float32
                "input_ids": [B, T_text] int64
                "state": [B, D_state] float32

        Returns:
            Action [B, action_horizon, action_dim]
        """
        model_dtype = self.dtype
        pixel_values = obs["pixel_values"].to(device=self.device, dtype=model_dtype)
        input_ids = obs["input_ids"].to(device=self.device, dtype=torch.long)
        state = obs["state"].to(device=self.device, dtype=model_dtype)
        attention_mask = obs.get("attention_mask")

        # Backbone
        backbone_out = self.backbone(pixel_values, input_ids, attention_mask)
        cond = backbone_out["backbone_features"]

        B = state.shape[0]
        cfg = self.config

        # Init noise
        actions = torch.randn(
            B, cfg.action_horizon, cfg.action_dim,
            device=self.device, dtype=cond.dtype,
        )
        dt = 1.0 / cfg.num_inference_timesteps

        # Denoising loop
        for t in range(cfg.num_inference_timesteps):
            t_disc = int(t / cfg.num_inference_timesteps * 999)
            ts = torch.full((B,), t_disc, device=self.device, dtype=torch.long)
            pred_vel = self.action_head(cond, actions, state, ts)
            actions = actions + dt * pred_vel

        return actions

    def get_action(self, obs: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Inference API compatible with Gr00tPolicy.get_action.

        Returns dict with "action" → numpy array.
        """
        with torch.inference_mode():
            # Convert numpy → torch if needed
            batch = {}
            for key in ["pixel_values", "input_ids", "state"]:
                val = obs[key]
                if isinstance(val, np.ndarray):
                    t = torch.from_numpy(val)
                    if key == "input_ids":
                        t = t.to(dtype=torch.long)
                    elif key == "state":
                        t = t.to(dtype=torch.float32)
                    batch[key] = t
                else:
                    batch[key] = val

            action = self.forward(batch)

        return {"action": action.float().cpu().numpy()}


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_dummy_gr00t_model(
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    **overrides,
) -> DummyGR00TModel:
    """Create a dummy GR00T model.

    Args:
        device: 'cuda' or 'cpu'. Default: auto-detect.
        dtype: torch.bfloat16 / float32. Default: auto-detect.
        **overrides: Any DummyGR00TConfig field.

    Returns:
        DummyGR00TModel in eval mode.
    """
    kwargs: dict = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    kwargs.update(overrides)

    config = DummyGR00TConfig(**kwargs)
    model = DummyGR00TModel(config)
    model = model.to(device=config.device)
    if config.dtype != torch.float32:
        model = model.to(dtype=config.dtype)
    model.eval()
    return model


def create_dummy_observation(
    batch_size: int = 1,
    image_size: int = 224,
    n_views: int = 2,
    text_tokens: int = 50,
    state_dim: int = 7,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create a dummy observation dict.

    Returns dict with "pixel_values", "input_ids", "state" as numpy arrays.
    """
    rng = np.random.default_rng(seed)
    return {
        "pixel_values": rng.random(
            (batch_size * n_views, 3, image_size, image_size)
        ).astype(np.float32),
        "input_ids": rng.integers(0, 1000, (batch_size, text_tokens)).astype(np.int64),
        "state": rng.random((batch_size, state_dim)).astype(np.float32),
    }
