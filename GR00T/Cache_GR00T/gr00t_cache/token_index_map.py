# SPDX-FileCopyrightText: Copyright (c) 2025 GR00T-Cache Contributors
# SPDX-License-Identifier: Apache-2.0
"""Token index mapping for GR00T multimodal sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class TokenIndexMap:
    """Maps token positions in the multimodal sequence to their modalities.

    GR00T concatenates text, visual, state, and action tokens into a single
    sequence. This class tracks which positions correspond to which modality.

    For the backbone (Eagle VLM):
        - Text tokens are at positions where input_ids != image_token_index
        - Visual tokens are at positions where input_ids == image_token_index
        - Visual tokens come from multiple camera views, each producing a grid
          of patches determined by the Eagle VLM tiling logic.

    For the action head (DiT):
        - Query tokens: state_features + future_tokens + action_features
        - Condition tokens: vl_embs (vision-language features from backbone)
          which contain text + visual tokens interleaved.
    """

    text_indices: Optional[torch.Tensor] = None  # [n_text_tokens]
    visual_indices: Optional[torch.Tensor] = None  # [n_visual_tokens]
    visual_indices_by_view: dict[str, torch.Tensor] = field(default_factory=dict)
    state_indices: Optional[torch.Tensor] = None
    action_indices: Optional[torch.Tensor] = None
    condition_indices: Optional[torch.Tensor] = None  # For cross-attention

    # Per-view patch grid info
    view_patch_grids: dict[str, tuple[int, int]] = field(default_factory=dict)
    view_token_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)

    # Sizes
    n_text: int = 0
    n_visual: int = 0
    n_visual_per_view: dict[str, int] = field(default_factory=dict)
    n_total: int = 0

    @classmethod
    def from_backbone_inputs(
        cls,
        input_ids: torch.Tensor,
        image_token_index: int,
        view_info: Optional[dict[str, Any]] = None,
    ) -> "TokenIndexMap":
        """Build index map from backbone (VLM) input.

        Args:
            input_ids: [batch, seq_len] or [seq_len] tokenized input.
            image_token_index: The token ID for <image> placeholder.
            view_info: Optional dict mapping view names to
                (num_tiles, tokens_per_tile, start_offset).

        Returns:
            TokenIndexMap with text and visual indices.
        """
        if input_ids.dim() == 2:
            ids = input_ids[0]  # Take first batch item
        else:
            ids = input_ids

        is_image = ids == image_token_index
        text_indices = torch.where(~is_image)[0]
        visual_indices = torch.where(is_image)[0]

        idx_map = cls(
            text_indices=text_indices,
            visual_indices=visual_indices,
            n_text=len(text_indices),
            n_visual=len(visual_indices),
            n_total=len(ids),
        )

        # If view info is provided, split visual indices per view
        if view_info is not None:
            for view_name, info in view_info.items():
                if isinstance(info, (tuple, list)) and len(info) >= 2:
                    start, end = int(info[0]), int(info[1])
                    view_vis = visual_indices[
                        (visual_indices >= start) & (visual_indices < end)
                    ]
                    idx_map.visual_indices_by_view[view_name] = view_vis
                    idx_map.n_visual_per_view[view_name] = len(view_vis)
                    idx_map.view_token_ranges[view_name] = (start, end)
                elif isinstance(info, int):
                    # Just the count — infer contiguous layout
                    cumsum = sum(idx_map.n_visual_per_view.values())
                    view_vis = visual_indices[cumsum : cumsum + info]
                    idx_map.visual_indices_by_view[view_name] = view_vis
                    idx_map.n_visual_per_view[view_name] = info

        return idx_map

    @classmethod
    def from_action_head_inputs(
        cls,
        query_seq_len: int,
        condition_seq_len: int,
        n_state_tokens: int = 1,
        n_future_tokens: int = 32,
        n_action_tokens: int = 16,
    ) -> "TokenIndexMap":
        """Build index map for action head (DiT) tokens.

        The query sequence is: [state, future_tokens, action_features]
        The condition sequence is: [text + visual tokens from backbone]

        Args:
            query_seq_len: Total query sequence length.
            condition_seq_len: Total condition sequence length.
            n_state_tokens: Number of state (proprioception) tokens.
            n_future_tokens: Number of learned future/query tokens.
            n_action_tokens: Number of action trajectory tokens.
        """
        state_indices = torch.arange(n_state_tokens)
        future_indices = torch.arange(n_state_tokens, n_state_tokens + n_future_tokens)
        action_indices = torch.arange(
            n_state_tokens + n_future_tokens,
            n_state_tokens + n_future_tokens + n_action_tokens,
        )
        condition_indices = torch.arange(condition_seq_len)

        return cls(
            state_indices=state_indices,
            action_indices=action_indices,
            condition_indices=condition_indices,
            n_total=query_seq_len,
            n_text=n_future_tokens,  # future tokens are learned, not text
            n_visual=condition_seq_len,
        )

    def get_visual_subset(
        self, view_name: Optional[str] = None
    ) -> torch.Tensor:
        """Get visual token indices, optionally filtered by view."""
        if view_name is not None and view_name in self.visual_indices_by_view:
            return self.visual_indices_by_view[view_name]
        return self.visual_indices

    def debug_summary(self) -> str:
        """Return a human-readable debug summary."""
        lines = [
            "TokenIndexMap Summary:",
            f"  Total tokens: {self.n_total}",
            f"  Text tokens: {self.n_text}",
            f"  Visual tokens: {self.n_visual}",
        ]
        for view_name, n in self.n_visual_per_view.items():
            lines.append(f"    {view_name}: {n} tokens")
        if self.state_indices is not None:
            lines.append(f"  State tokens: {len(self.state_indices)}")
        if self.action_indices is not None:
            lines.append(f"  Action tokens: {len(self.action_indices)}")
        return "\n".join(lines)

    def to(self, device: torch.device) -> "TokenIndexMap":
        """Move all tensors to the specified device."""
        for field_name in [
            "text_indices", "visual_indices", "state_indices",
            "action_indices", "condition_indices",
        ]:
            val = getattr(self, field_name, None)
            if val is not None:
                setattr(self, field_name, val.to(device))
        for key in self.visual_indices_by_view:
            self.visual_indices_by_view[key] = self.visual_indices_by_view[key].to(device)
        return self
