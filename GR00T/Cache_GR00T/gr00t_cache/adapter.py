# SPDX-FileCopyrightText: Copyright (c) 2025 GR00T-Cache Contributors
# SPDX-License-Identifier: Apache-2.0
"""
RealGR00TAdapter: Bridge between GR00T-Cache and the real GR00T model.

Provides a minimal adapter that hooks into the GR00T N1.5 model's
forward pass to enable cross-timestep visual token KV caching.

Key design principles:
1. Minimum intrusion — wraps existing modules, doesn't rewrite forward()
2. Hook-based profiling — uses forward hooks to measure per-component latency
3. Safe fallback — returns original output when cache is disabled
4. No hardcoded paths — all paths passed as arguments
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import tree

from .attention_wrapper import (
    CachedAttentionWrapper,
    apply_cache_to_backbone,
    apply_cache_to_action_head,
    remove_cache_from_model,
)
from .cache_manager import GR00TCacheManager
from .config import CacheMode, GR00TCacheConfig
from .token_index_map import TokenIndexMap


class RealGR00TAdapter:
    """Adapter for integrating GR00T-Cache with the real GR00T N1.5 model.

    This adapter:
    1. Loads a GR00T policy via Gr00tPolicy
    2. Wraps attention layers for KV caching
    3. Manages cache lifecycle across policy steps
    4. Provides profiling instrumentation

    Usage:
        adapter = RealGR00TAdapter(
            model_path="nvidia/GR00T-N1.5-3B",
            cache_config=GR00TCacheConfig(...),
            ...
        )
        adapter.setup()
        action = adapter.get_action(observation)
        adapter.reset_cache()  # Between episodes
    """

    def __init__(
        self,
        model_path: str,
        data_config_path: Optional[str] = "examples.Libero.custom_data_config:LiberoDataConfig",
        embodiment_tag: str = "new_embodiment",
        denoising_steps: int = 8,
        device: str = "cuda",
        cache_config: Optional[GR00TCacheConfig] = None,
        image_token_index: int = -200,  # Eagle VLM default image token
    ):
        """
        Args:
            model_path: HuggingFace hub ID or local path to GR00T model.
            data_config_path: Import path to data config (e.g.,
                "examples.Libero.custom_data_config:LiberoDataConfig").
            embodiment_tag: Embodiment tag for the robot.
            denoising_steps: Number of denoising steps for flow matching.
            device: Device to run on.
            cache_config: GR00T-Cache configuration. If None, caching is disabled.
            image_token_index: Token ID for <image> placeholder in VLM.
        """
        self.model_path = model_path
        self.data_config_path = data_config_path
        self.embodiment_tag = embodiment_tag
        self.denoising_steps = denoising_steps
        self.device = device
        self.cache_config = cache_config or GR00TCacheConfig(enabled=False)
        self.image_token_index = image_token_index

        # Will be set by setup()
        self.policy = None
        self.model = None
        self.cache_manager: Optional[GR00TCacheManager] = None
        self._backbone_wrappers: dict = {}
        self._action_head_wrappers: dict = {}
        self._setup_done = False

        # Per-step tracking
        self._current_step_images: Optional[torch.Tensor] = None
        self._current_instruction: Optional[str] = None
        self._current_input_ids: Optional[torch.Tensor] = None
        self._current_token_map: Optional[TokenIndexMap] = None
        self._collected_attention_maps: dict[int, torch.Tensor] = {}

    def setup(self) -> None:
        """Initialize the GR00T policy and apply cache wrappers.

        This must be called before get_action().
        """
        print(f"[GR00T-Adapter] Loading model from {self.model_path}...")
        start = time.perf_counter()

        # Import GR00T policy
        from gr00t.model.policy import Gr00tPolicy
        from gr00t.experiment.data_config import load_data_config

        data_config = load_data_config(self.data_config_path)
        self.policy = Gr00tPolicy(
            model_path=self.model_path,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag=self.embodiment_tag,
            denoising_steps=self.denoising_steps,
            device=self.device,
        )
        self.model = self.policy.model

        load_time = time.perf_counter() - start
        print(f"[GR00T-Adapter] Model loaded in {load_time:.1f}s")

        # Setup cache manager
        self.cache_manager = GR00TCacheManager(self.cache_config)

        # Apply cache wrappers if enabled
        if self.cache_config.enabled:
            self._apply_cache_wrappers()

        self._setup_done = True

    def _apply_cache_wrappers(self) -> None:
        """Wrap attention layers with caching wrappers."""
        mode = self.cache_config.cache_mode

        # Build token index map from first forward pass
        # We'll need to do a dummy forward to discover token layout
        # This is deferred to the first get_action call

        if mode in (CacheMode.BACKBONE_VISUAL_KV_CACHE, CacheMode.FULL_CACHE):
            if self.cache_config.debug:
                print("[GR00T-Adapter] Applying backbone visual KV cache...")

            # Create a placeholder token map — will be updated on first step
            dummy_map = TokenIndexMap(n_visual=256)  # Placeholder

            self._backbone_wrappers = apply_cache_to_backbone(
                self.model,
                self.cache_manager,
                dummy_map,
                self.cache_config,
            )

        if mode in (CacheMode.ACTION_HEAD_CONDITION_KV_CACHE, CacheMode.FULL_CACHE):
            if self.cache_config.debug:
                print("[GR00T-Adapter] Applying action head condition KV cache...")

            self._action_head_wrappers = apply_cache_to_action_head(
                self.model,
                self.cache_manager,
                self.cache_config,
            )

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        """Get action with optional cache management.

        Args:
            observations: Raw observation dict from environment.

        Returns:
            Action dict with keys like "action" or robot-specific keys.
        """
        if not self._setup_done:
            self.setup()

        cfg = self.cache_config

        if not cfg.enabled:
            # Passthrough — no cache overhead
            return self.policy.get_action(observations)

        # Build token index map if needed
        self._build_token_index_map(observations)

        # Extract images and instruction for cache management
        self._extract_cache_context(observations)

        # Compute reuse plan
        reuse_plan = self.cache_manager.get_reuse_plan(
            current_images=self._current_step_images,
            current_instruction=self._current_instruction,
            current_proprio=self._get_proprio(observations),
            current_token_map=self._current_token_map,
            attention_maps=self._collected_attention_maps,
        )

        self.cache_manager._current_reuse_plan = reuse_plan

        if cfg.debug and reuse_plan.get("should_cache"):
            print(
                f"[GR00T-Cache] Step {self.cache_manager.cache_step_id}: "
                f"reuse_ratio={reuse_plan['reuse_ratio']:.2f}, "
                f"cache_age={reuse_plan['cache_age']}"
            )

        # Run policy with cache active
        result = self.policy.get_action(observations)

        # Collect attention maps from wrappers
        self._collect_attention_maps()

        # Update cache for next step
        self._update_cache_after_step(observations)

        return result

    def get_action_profiled(self, observations: dict[str, Any]) -> dict[str, Any]:
        """Get action with detailed profiling.

        Returns dict with '__action__' and '__timing__' keys.
        """
        if not self._setup_done:
            self.setup()

        return self.policy.get_action_profiled(observations)

    def reset_cache(self) -> None:
        """Reset the cache (call between episodes)."""
        if self.cache_manager is not None:
            self.cache_manager.reset()
            if self.cache_config.debug:
                print("[GR00T-Cache] Cache reset for new episode")

    def reset(self) -> None:
        """Alias for reset_cache."""
        self.reset_cache()

    def _build_token_index_map(self, observations: dict[str, Any]) -> None:
        """Build or update the token index map from current inputs.

        This infers the token layout from the VLM processor output.
        """
        # Process observations through transforms to get VLM inputs
        obs_copy = observations.copy()
        from gr00t.model.policy import unsqueeze_dict_values
        if not self.policy._check_state_is_batched(obs_copy):
            obs_copy = unsqueeze_dict_values(obs_copy)
        for k, v in obs_copy.items():
            if not isinstance(v, torch.Tensor) and not isinstance(v, list):
                obs_copy[k] = torch.tensor(v) if not isinstance(v, (torch.Tensor, list)) else v

        normalized = self.policy.apply_transforms(obs_copy)

        # Get input_ids from normalized input
        if "eagle_input_ids" in normalized or "input_ids" in normalized:
            input_ids_key = "eagle_input_ids" if "eagle_input_ids" in normalized else "input_ids"
            input_ids = normalized[input_ids_key]

            view_info = self._infer_view_info(normalized, input_ids)
            self._current_token_map = TokenIndexMap.from_backbone_inputs(
                input_ids=input_ids,
                image_token_index=self.image_token_index,
                view_info=view_info,
            )
            self._current_input_ids = input_ids

    def _infer_view_info(
        self, normalized: dict, input_ids: torch.Tensor
    ) -> dict[str, Any]:
        """Infer per-view visual token ranges from normalized inputs.

        Uses pixel_values/image_sizes to determine how many visual tokens
        each camera view contributes.
        """
        view_info = {}
        # Try to get image sizes information
        if "eagle_image_sizes" in normalized:
            image_sizes = normalized["eagle_image_sizes"]
            # Each image contributes num_tiles * 256 visual tokens
            patch_size = 14
            tokens_per_tile = 256  # (448/14)^2 * 0.5^2 = 256

            offset = 0
            for v_idx in range(len(image_sizes) if image_sizes.dim() > 0 else 0):
                # Determine number of tiles
                h, w = image_sizes[v_idx].tolist() if image_sizes.dim() > 1 else (448, 448)
                n_tiles_w = max(1, w // 448)
                n_tiles_h = max(1, h // 448)
                n_tiles = n_tiles_w * n_tiles_h
                n_tokens = n_tiles * tokens_per_tile

                view_name = f"view_{v_idx}"
                if v_idx == 1:
                    view_name = "wrist"  # Second camera is typically wrist
                view_info[view_name] = (offset, offset + n_tokens)
                view_info[f"{view_name}_ntiles"] = n_tiles
                view_info[f"{view_name}_tokens_per_tile"] = tokens_per_tile
                offset += n_tokens
        else:
            # Fallback: assume split evenly
            n_vis = (input_ids == self.image_token_index).sum().item()
            n_views = 2  # Default: external + wrist
            per_view = n_vis // n_views
            view_info = {
                "external": (0, per_view),
                "wrist": (per_view, n_vis),
            }

        return view_info

    def _extract_cache_context(self, observations: dict[str, Any]) -> None:
        """Extract images and instruction from observations for cache management."""
        # Extract images
        if "video" in observations:
            video = observations["video"]
            if isinstance(video, torch.Tensor):
                video = video.cpu().numpy()
            # video: [T, V, H, W, C] or [B, T, V, H, W, C]
            if video.ndim >= 5:
                # Take last timestep, all views
                idx = -1
                # Select last timestep
                if video.ndim == 5:
                    frames = video[idx]  # [V, H, W, C]
                else:
                    frames = video[0, idx]  # [V, H, W, C]

                # Convert to [V, 3, H, W] float32
                frames = torch.from_numpy(frames).float() / 255.0
                if frames.shape[-1] == 3:
                    frames = frames.permute(0, 3, 1, 2)  # [V, 3, H, W]
                self._current_step_images = frames

        # Extract instruction
        if "annotation" in observations:
            ann = observations["annotation"]
            if isinstance(ann, dict):
                for k, v in ann.items():
                    if "task" in k or "instruction" in k:
                        self._current_instruction = str(v[0]) if hasattr(v, '__len__') else str(v)
                        break
            elif isinstance(ann, (list, np.ndarray)):
                self._current_instruction = str(ann[0])

    def _get_proprio(self, observations: dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract proprioception state from observations."""
        state_parts = []
        for k, v in observations.items():
            if "state" in k.lower():
                if isinstance(v, np.ndarray):
                    state_parts.append(torch.from_numpy(v).float().flatten())
                elif isinstance(v, torch.Tensor):
                    state_parts.append(v.float().flatten())

        if state_parts:
            return torch.cat(state_parts)
        return None

    def _collect_attention_maps(self) -> None:
        """Collect attention weights from all cache wrappers."""
        self._collected_attention_maps = {}

        for path, wrapper in self._backbone_wrappers.items():
            weights = wrapper.get_attention_weights()
            if weights is not None:
                layer_idx = int(path.split("_")[-1]) if "_" in path else 0
                self._collected_attention_maps[layer_idx] = weights

        for path, wrapper in self._action_head_wrappers.items():
            weights = wrapper.get_attention_weights()
            if weights is not None:
                # Use negative indices for DiT layers
                layer_idx = -1 - len(self._collected_attention_maps)
                self._collected_attention_maps[layer_idx] = weights

    def _update_cache_after_step(self, observations: dict[str, Any]) -> None:
        """Store current step data into cache manager."""
        if self.cache_manager is None:
            return

        # Store backbone K/V
        for wrapper in self._backbone_wrappers.values():
            wrapper.store_backbone_kv(wrapper.layer_idx)

        # Store condition K/V
        for wrapper in self._action_head_wrappers.values():
            wrapper.store_condition_kv(wrapper.layer_idx)

        self.cache_manager.update_cache(
            current_images=self._current_step_images,
            current_visual_tokens=None,  # Will be filled by wrapper
            current_token_map=self._current_token_map,
            current_instruction=self._current_instruction,
            current_proprio=self._get_proprio(observations),
            attention_maps=self._collected_attention_maps,
            layer_kv=self.cache_manager.layer_kv_cache,
            condition_kv=self.cache_manager.action_head_condition_kv,
        )

    def cleanup(self) -> None:
        """Remove cache wrappers and restore original model."""
        remove_cache_from_model(self.model, self._backbone_wrappers)
        remove_cache_from_model(self.model, self._action_head_wrappers)
        self._backbone_wrappers = {}
        self._action_head_wrappers = {}

    @property
    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        if self.cache_manager is None:
            return {}
        return self.cache_manager.stats()
