"""FastV-style visual token pruning utilities for GR00T System 2."""

from __future__ import annotations

import types
from functools import partial
from typing import Any, Optional

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() not in ("0", "false", "no", "off", "")
    return bool(value)


def fastv_prune_hidden_states(
    hidden_states,
    attention_scores,
    attention_mask,
    position_ids,
    image_token_start_index: int,
    image_token_length: int,
    fastv_r: float = 0.5,
):
    """Prune visual tokens in one contiguous image-token interval.

    Odd visual-token counts use floor via ``int(image_token_length * keep_ratio)``,
    with a minimum of one kept visual token.
    """

    if image_token_start_index is None or image_token_length is None:
        raise ValueError("image_token_start_index and image_token_length are required for FastV.")

    batch_size = hidden_states.shape[0]
    device = hidden_states.device
    img_start = int(image_token_start_index)
    img_end = img_start + int(image_token_length)
    visual_token_indices = torch.arange(img_start, img_end, device=device).unsqueeze(0)
    visual_token_indices = visual_token_indices.expand(batch_size, -1)
    return _fastv_prune_by_visual_indices(
        hidden_states=hidden_states,
        attention_scores=attention_scores,
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_token_indices=visual_token_indices,
        fastv_r=fastv_r,
    )


def _fastv_prune_by_visual_indices(
    hidden_states: torch.Tensor,
    attention_scores: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    visual_token_indices: torch.Tensor,
    fastv_r: float = 0.5,
):
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must be [B, S, D], got {tuple(hidden_states.shape)}")
    if attention_scores is None or attention_scores.ndim != 4:
        raise ValueError("FastV requires attention_scores with shape [B, H, S, S].")

    batch_size, seq_len, hidden_dim = hidden_states.shape
    if visual_token_indices.ndim != 2 or visual_token_indices.shape[0] != batch_size:
        raise ValueError("visual_token_indices must have shape [B, M].")

    image_token_length = int(visual_token_indices.shape[1])
    keep_num = max(1, int(image_token_length * (1.0 - float(fastv_r))))
    keep_num = min(keep_num, image_token_length)

    gather_index = visual_token_indices[:, None, None, :].expand(
        -1, attention_scores.shape[1], attention_scores.shape[2], -1
    )
    visual_attn = torch.gather(attention_scores, dim=-1, index=gather_index)
    importance = visual_attn.mean(dim=(1, 2))

    top_local = torch.topk(importance, k=keep_num, dim=-1).indices
    top_visual_indices = torch.gather(visual_token_indices, dim=1, index=top_local)

    all_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
    visual_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=hidden_states.device)
    visual_mask.scatter_(1, visual_token_indices, True)
    keep_mask = ~visual_mask
    keep_mask.scatter_(1, top_visual_indices, True)

    kept_indices = []
    for batch_idx in range(batch_size):
        kept_indices.append(all_indices[batch_idx][keep_mask[batch_idx]])
    kept_indices = torch.stack(kept_indices, dim=0)
    kept_indices = torch.sort(kept_indices, dim=1).values

    hidden_gather = kept_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
    pruned_hidden_states = torch.gather(hidden_states, dim=1, index=hidden_gather)
    pruned_attention_mask = prune_attention_mask(attention_mask, kept_indices, seq_len)
    pruned_position_ids = prune_position_ids(position_ids, kept_indices, batch_size)
    return pruned_hidden_states, pruned_attention_mask, pruned_position_ids, kept_indices


def prune_position_ids(
    position_ids: Optional[torch.Tensor], kept_indices: torch.Tensor, batch_size: int
) -> Optional[torch.Tensor]:
    if position_ids is None:
        return None
    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.shape[0] == 1 and batch_size > 1:
        position_ids = position_ids.expand(batch_size, -1)
    return torch.gather(position_ids, dim=1, index=kept_indices.to(position_ids.device))


def prune_attention_mask(
    attention_mask: Optional[torch.Tensor], kept_indices: torch.Tensor, original_seq_len: int
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None

    kept_indices = kept_indices.to(attention_mask.device)
    batch_size, kept_len = kept_indices.shape

    if attention_mask.ndim == 2 and attention_mask.shape[-1] == original_seq_len:
        return torch.gather(attention_mask, dim=1, index=kept_indices)

    if attention_mask.ndim == 3 and attention_mask.shape[-1] == original_seq_len:
        index = kept_indices[:, None, :].expand(-1, attention_mask.shape[1], -1)
        return torch.gather(attention_mask, dim=2, index=index)

    if attention_mask.ndim == 4 and attention_mask.shape[-1] == original_seq_len:
        key_index = kept_indices[:, None, None, :].expand(
            -1, attention_mask.shape[1], attention_mask.shape[2], -1
        )
        pruned = torch.gather(attention_mask, dim=3, index=key_index)
        if attention_mask.shape[-2] == original_seq_len:
            query_index = kept_indices[:, None, :, None].expand(
                -1, pruned.shape[1], -1, pruned.shape[3]
            )
            pruned = torch.gather(pruned, dim=2, index=query_index)
        return pruned

    return attention_mask


def infer_visual_token_indices(
    input_ids: torch.Tensor,
    image_token_index: int,
    image_token_start_index: Optional[int] = None,
    image_token_length: Optional[int] = None,
) -> tuple[torch.Tensor, int, int, bool]:
    """Infer per-batch visual token positions from GR00T Eagle input ids."""

    if image_token_start_index is not None and image_token_length is not None:
        start = int(image_token_start_index)
        length = int(image_token_length)
        indices = torch.arange(start, start + length, device=input_ids.device).unsqueeze(0)
        return indices.expand(input_ids.shape[0], -1), start, length, True

    image_mask = input_ids == int(image_token_index)
    counts = image_mask.sum(dim=1)
    if counts.numel() == 0 or int(counts.min().item()) <= 0:
        raise ValueError("FastV could not find image tokens in input_ids.")
    if not torch.all(counts == counts[0]):
        raise ValueError(f"FastV requires equal visual-token counts per batch, got {counts.tolist()}.")

    visual_token_indices = []
    contiguous = True
    starts = []
    for batch_idx in range(input_ids.shape[0]):
        positions = torch.nonzero(image_mask[batch_idx], as_tuple=False).flatten()
        starts.append(int(positions[0].item()))
        if positions.numel() > 1 and not torch.all(positions[1:] == positions[:-1] + 1):
            contiguous = False
        visual_token_indices.append(positions)
    return torch.stack(visual_token_indices, dim=0), starts[0], int(counts[0].item()), contiguous


def install_fastv_qwen3_forward(qwen3_model: torch.nn.Module) -> None:
    if getattr(qwen3_model, "_gr00t_fastv_forward_installed", False):
        return
    qwen3_model._gr00t_original_forward = qwen3_model.forward
    qwen3_model.forward = types.MethodType(_qwen3_fastv_forward, qwen3_model)
    qwen3_model._gr00t_fastv_forward_installed = True


def set_attention_implementation(module: torch.nn.Module, implementation: str) -> None:
    for submodule in module.modules():
        config = getattr(submodule, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = implementation


def _qwen3_fastv_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    fastv_config: Optional[dict[str, Any]] = None,
    **flash_attn_kwargs: Any,
) -> BaseModelOutputWithPast:
    use_fastv = _as_bool((fastv_config or {}).get("use_fastv"), False)
    if not use_fastv:
        return self._gr00t_original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = False
    past_key_values = None
    self._fastv_last_kept_indices = None
    self._fastv_last_attention_mask = None

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if self.gradient_checkpointing and self.training:
        raise ValueError("FastV pruning is not supported with training gradient checkpointing.")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, True
    )
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    fastv_k = int(fastv_config.get("fastv_k", 2))
    fastv_r = float(fastv_config.get("fastv_r", 0.5))
    verbose = _as_bool(fastv_config.get("verbose"), False)
    pruned_once = False

    for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        need_attn_for_fastv = (
            not pruned_once
            and layer_idx + 1 == fastv_k
            and fastv_config.get("visual_token_indices") is not None
        )
        layer_output_attentions = bool(output_attentions or need_attn_for_fastv)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=layer_output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **flash_attn_kwargs,
        )
        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        if need_attn_for_fastv:
            visual_token_indices = fastv_config["visual_token_indices"].to(hidden_states.device)
            original_seq_len = hidden_states.shape[1]
            hidden_states, causal_mask, position_ids, kept_indices = _fastv_prune_by_visual_indices(
                hidden_states=hidden_states,
                attention_scores=layer_outputs[1],
                attention_mask=causal_mask,
                position_ids=position_ids,
                visual_token_indices=visual_token_indices,
                fastv_r=fastv_r,
            )
            self._fastv_last_kept_indices = kept_indices.detach()
            self._fastv_last_attention_mask = prune_attention_mask(
                attention_mask, kept_indices, original_seq_len
            )
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            pruned_once = True
            if verbose:
                print(
                    "[FastV] "
                    f"layer={layer_idx + 1} seq_len {original_seq_len}->{hidden_states.shape[1]} "
                    f"visual_tokens {visual_token_indices.shape[1]}->{max(1, int(visual_token_indices.shape[1] * (1.0 - fastv_r)))}"
                )

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
