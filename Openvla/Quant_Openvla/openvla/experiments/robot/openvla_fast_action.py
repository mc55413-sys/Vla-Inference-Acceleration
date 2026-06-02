"""Low-latency action generation helpers for OpenVLA.

OpenVLA actions are represented by a small suffix of vocabulary tokens. During
inference we only need logits for those action tokens, but Hugging Face
``generate`` computes the full vocabulary projection for every prefill and
decode step. These helpers call the underlying language backbone directly and
apply ``lm_head`` only to the action-token rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FastActionOutputs:
    generated_action_token_ids: torch.Tensor
    past_key_values: Any


@dataclass
class LastTokenLogitsOutputs:
    generated_action_token_ids: torch.Tensor
    past_key_values: Any


def can_use_fast_action_head(model: Any) -> bool:
    language_model = getattr(model, "language_model", None)
    lm_backbone = get_language_backbone(language_model)
    lm_head = get_lm_head(language_model)
    return (
        language_model is not None
        and lm_backbone is not None
        and lm_head is not None
        and hasattr(lm_head, "weight")
        and hasattr(model, "bin_centers")
        and hasattr(model, "vocab_size")
    )


def can_use_last_token_logits(model: Any) -> bool:
    language_model = getattr(model, "language_model", None)
    return (
        language_model is not None
        and get_language_backbone(language_model) is not None
        and get_lm_head(language_model) is not None
    )


def get_language_backbone(language_model: Any) -> Optional[torch.nn.Module]:
    if language_model is None:
        return None
    for attr_name in ("model", "transformer"):
        backbone = getattr(language_model, attr_name, None)
        if backbone is not None:
            return backbone
    return None


def get_lm_head(language_model: Any) -> Optional[torch.nn.Module]:
    if language_model is None:
        return None
    if hasattr(language_model, "get_output_embeddings"):
        lm_head = language_model.get_output_embeddings()
        if lm_head is not None:
            return lm_head
    return getattr(language_model, "lm_head", None)


def limit_llm_layers(model: Any, max_layers: int = 0, strategy: str = "first") -> Tuple[int, int]:
    if max_layers <= 0:
        return 0, 0

    language_model = getattr(model, "language_model", None)
    backbone = get_language_backbone(language_model)
    layers = getattr(backbone, "layers", None)
    if backbone is None or layers is None:
        raise AttributeError("Expected language backbone to expose a `layers` ModuleList")

    original_layers = len(layers)
    if max_layers >= original_layers:
        return original_layers, original_layers

    indices = select_layer_indices(original_layers, max_layers, strategy)
    backbone.layers = nn.ModuleList([layers[index] for index in indices])
    for config in (
        getattr(backbone, "config", None),
        getattr(language_model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
    ):
        if config is not None and hasattr(config, "num_hidden_layers"):
            setattr(config, "num_hidden_layers", len(indices))
    model._latency_layer_indices = indices
    return original_layers, len(indices)


def select_layer_indices(total_layers: int, max_layers: int, strategy: str) -> Tuple[int, ...]:
    if strategy == "first":
        return tuple(range(max_layers))
    if strategy == "last":
        return tuple(range(total_layers - max_layers, total_layers))
    if strategy != "uniform":
        raise ValueError(f"Unsupported LLM layer strategy: {strategy}")

    if max_layers == 1:
        return (total_layers - 1,)
    raw = torch.linspace(0, total_layers - 1, steps=max_layers).round().to(dtype=torch.long).tolist()
    indices = []
    for index in raw:
        index = int(index)
        if index not in indices:
            indices.append(index)
    next_index = 0
    while len(indices) < max_layers:
        if next_index not in indices:
            indices.append(next_index)
        next_index += 1
    return tuple(sorted(indices[:max_layers]))


def build_multimodal_inputs(
    model: Any,
    input_ids: torch.Tensor,
    projected_patch_embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_embeddings = model.get_input_embeddings()(input_ids)
    multimodal_embeddings = torch.cat(
        [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
    )

    multimodal_attention_mask = None
    if attention_mask is not None:
        projected_patch_attention_mask = torch.ones(
            (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        multimodal_attention_mask = torch.cat(
            [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
        )
    return multimodal_embeddings, multimodal_attention_mask


def reduce_vision_tokens(
    projected_patch_embeddings: torch.Tensor,
    max_vision_tokens: int = 0,
    strategy: str = "uniform",
) -> torch.Tensor:
    if max_vision_tokens <= 0 or projected_patch_embeddings.shape[1] <= max_vision_tokens:
        return projected_patch_embeddings

    token_count = projected_patch_embeddings.shape[1]
    if strategy == "first":
        return projected_patch_embeddings[:, :max_vision_tokens, :]
    if strategy == "pool":
        pooled = F.adaptive_avg_pool1d(projected_patch_embeddings.transpose(1, 2), max_vision_tokens)
        return pooled.transpose(1, 2).contiguous()
    if strategy != "uniform":
        raise ValueError(f"Unsupported vision token reduction strategy: {strategy}")

    indices = torch.div(
        torch.arange(max_vision_tokens, device=projected_patch_embeddings.device) * token_count,
        max_vision_tokens,
        rounding_mode="floor",
    )
    return projected_patch_embeddings.index_select(dim=1, index=indices)


@torch.inference_mode()
def generate_action_tokens_fast(
    model: Any,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    unnorm_key: Optional[str],
) -> FastActionOutputs:
    language_model = model.language_model
    lm_backbone = get_language_backbone(language_model)
    if lm_backbone is None:
        raise AttributeError("Expected language_model to expose a `.model` or `.transformer` backbone")

    action_dim = model.get_action_dim(unnorm_key)
    generated_tokens = []

    outputs = lm_backbone(
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=multimodal_embeddings,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    next_token = select_next_action_token(model, outputs.last_hidden_state[:, -1, :])
    generated_tokens.append(next_token)
    past_key_values = outputs.past_key_values

    for _ in range(action_dim - 1):
        outputs = lm_backbone(
            input_ids=next_token[:, None],
            attention_mask=None,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        next_token = select_next_action_token(model, outputs.last_hidden_state[:, -1, :])
        generated_tokens.append(next_token)
        past_key_values = outputs.past_key_values

    return FastActionOutputs(
        generated_action_token_ids=torch.stack(generated_tokens, dim=1),
        past_key_values=past_key_values,
    )


@torch.inference_mode()
def generate_action_tokens_last_logits(
    model: Any,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    unnorm_key: Optional[str],
) -> LastTokenLogitsOutputs:
    """Generate actions exactly, but only project the final hidden state.

    ``LlamaForCausalLM`` applies ``lm_head`` to every prefill token even though
    OpenVLA only consumes ``logits[:, -1, :]``. Calling the backbone directly
    keeps the same KV cache and full-vocabulary argmax while avoiding that
    unused prefill projection.
    """

    language_model = model.language_model
    lm_backbone = get_language_backbone(language_model)
    lm_head = get_lm_head(language_model)
    if lm_backbone is None or lm_head is None:
        raise AttributeError("Expected language_model to expose a backbone and lm_head")

    action_dim = model.get_action_dim(unnorm_key)
    generated_tokens = []

    outputs = lm_backbone(
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=multimodal_embeddings,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    next_token = select_next_token_full_vocab(lm_head, outputs.last_hidden_state[:, -1, :])
    generated_tokens.append(next_token)
    past_key_values = outputs.past_key_values

    for _ in range(action_dim - 1):
        outputs = lm_backbone(
            input_ids=next_token[:, None],
            attention_mask=None,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        next_token = select_next_token_full_vocab(lm_head, outputs.last_hidden_state[:, -1, :])
        generated_tokens.append(next_token)
        past_key_values = outputs.past_key_values

    return LastTokenLogitsOutputs(
        generated_action_token_ids=torch.stack(generated_tokens, dim=1),
        past_key_values=past_key_values,
    )


@torch.inference_mode()
def predict_action_fast(
    model: Any,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    unnorm_key: Optional[str] = None,
    max_vision_tokens: int = 0,
    vision_token_strategy: str = "uniform",
    **_: Any,
) -> np.ndarray:
    if not torch.all(input_ids[:, -1] == 29871):
        empty_token = torch.full((input_ids.shape[0], 1), 29871, dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat((input_ids, empty_token), dim=1)
        if attention_mask is not None:
            extra_mask = torch.ones(
                (attention_mask.shape[0], 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat((attention_mask, extra_mask), dim=1)

    patch_features = model.vision_backbone(pixel_values)
    projected_patch_embeddings = model.projector(patch_features)
    projected_patch_embeddings = reduce_vision_tokens(
        projected_patch_embeddings,
        max_vision_tokens=max_vision_tokens,
        strategy=vision_token_strategy,
    )
    multimodal_embeddings, multimodal_attention_mask = build_multimodal_inputs(
        model, input_ids, projected_patch_embeddings, attention_mask
    )
    fast_outputs = generate_action_tokens_fast(model, multimodal_embeddings, multimodal_attention_mask, unnorm_key)
    return decode_action_tokens(model, fast_outputs.generated_action_token_ids[0], unnorm_key)


@torch.inference_mode()
def predict_action_last_logits(
    model: Any,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    unnorm_key: Optional[str] = None,
    drop_full_attention_mask: bool = True,
    **_: Any,
) -> np.ndarray:
    if not torch.all(input_ids[:, -1] == 29871):
        empty_token = torch.full((input_ids.shape[0], 1), 29871, dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat((input_ids, empty_token), dim=1)
        if attention_mask is not None:
            extra_mask = torch.ones(
                (attention_mask.shape[0], 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat((attention_mask, extra_mask), dim=1)

    if drop_full_attention_mask and torch.is_tensor(attention_mask) and bool(torch.all(attention_mask != 0)):
        attention_mask = None

    patch_features = model.vision_backbone(pixel_values)
    projected_patch_embeddings = model.projector(patch_features)
    multimodal_embeddings, multimodal_attention_mask = build_multimodal_inputs(
        model, input_ids, projected_patch_embeddings, attention_mask
    )
    outputs = generate_action_tokens_last_logits(model, multimodal_embeddings, multimodal_attention_mask, unnorm_key)
    return decode_action_tokens(model, outputs.generated_action_token_ids[0], unnorm_key)


def select_next_token_full_vocab(lm_head: torch.nn.Module, last_hidden_state: torch.Tensor) -> torch.Tensor:
    logits = lm_head(last_hidden_state)
    return logits.argmax(dim=-1)


def select_next_action_token(model: Any, last_hidden_state: torch.Tensor) -> torch.Tensor:
    action_token_ids = get_action_token_ids(model, last_hidden_state.device)
    language_model = model.language_model
    lm_head = get_lm_head(language_model)
    if lm_head is None or not hasattr(lm_head, "weight"):
        logits = language_model.lm_head(last_hidden_state)
        local_token_idx = logits.index_select(dim=-1, index=action_token_ids).argmax(dim=-1)
        return action_token_ids.index_select(0, local_token_idx)

    weight, bias = get_cached_action_lm_head(model, lm_head, action_token_ids, last_hidden_state.dtype)
    action_logits = F.linear(last_hidden_state, weight, bias)
    local_token_idx = action_logits.argmax(dim=-1)
    return action_token_ids.index_select(0, local_token_idx)


def get_cached_action_lm_head(
    model: Any,
    lm_head: torch.nn.Module,
    action_token_ids: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    cache = getattr(model, "_fast_action_lm_head_cache", {})
    bias_tensor = getattr(lm_head, "bias", None)
    cache_key = (
        lm_head.weight.data_ptr(),
        action_token_ids.device.type,
        action_token_ids.device.index,
        dtype,
        int(action_token_ids.numel()),
        bool(bias_tensor is not None),
    )
    if cache_key in cache:
        return cache[cache_key]

    weight = lm_head.weight.index_select(0, action_token_ids).to(dtype=dtype).contiguous()
    bias = None
    if bias_tensor is not None:
        bias = bias_tensor.index_select(0, action_token_ids).to(dtype=dtype).contiguous()
    cache[cache_key] = (weight, bias)
    model._fast_action_lm_head_cache = cache
    return weight, bias


def get_action_token_ids(model: Any, device: torch.device) -> torch.Tensor:
    num_action_bins = int(np.asarray(model.bin_centers).shape[0])
    return torch.arange(model.vocab_size - 1, model.vocab_size - num_action_bins - 1, -1, device=device)


def decode_action_tokens(model: Any, generated_action_token_ids: torch.Tensor, unnorm_key: Optional[str]) -> np.ndarray:
    predicted_action_token_ids = generated_action_token_ids.detach().cpu().numpy()
    discretized_actions = model.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    normalized_actions = model.bin_centers[discretized_actions]

    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    return np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
