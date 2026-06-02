# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from gr00t.atm import ensure_dit_attention_patch, enable_dit_atm_if_configured
from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.gr00t_n1 import GR00T_N1_5

COMPUTE_DTYPE = torch.bfloat16


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to get the action for a given state.

        Args:
            observations: The observations from the environment.

        Returns:
            The action to take in the environment in dictionary format.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Return the modality config of the policy.
        """
        raise NotImplementedError


class Gr00tPolicy(BasePolicy):
    """
    A wrapper for Gr00t model checkpoints that handles loading the model, applying transforms,
    making predictions, and unapplying transforms. This loads some custom configs, stats
    and metadata related to the model checkpoints used
    in the Gr00t model.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        denoising_steps: Optional[int] = None,
        device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Gr00tPolicy.

        Args:
            model_path (str): Path to the model checkpoint directory or the huggingface hub id.
            modality_config (Dict[str, ModalityConfig]): The modality config for the model.
            modality_transform (ComposedModalityTransform): The modality transform for the model.
            embodiment_tag (Union[str, EmbodimentTag]): The embodiment tag for the model.
            denoising_steps: Number of denoising steps to use for the action head.
            device (Union[int, str]): Device to run the model on.
        """
        try:
            # NOTE(YL) this returns the local path to the model which is normally
            # saved in ~/.cache/huggingface/hub/
            model_path = snapshot_download(model_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {model_path}"
            )

        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self._modality_transform.eval()  # set this to eval mode
        self.model_path = Path(model_path)
        self.device = device

        # Convert string embodiment tag to EmbodimentTag enum if needed
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        # Load model
        self._load_model(model_path)
        # Load transforms
        self._load_metadata(self.model_path / "experiment_cfg")
        # Load horizons
        self._load_horizons()

        if denoising_steps is not None:
            if hasattr(self.model, "action_head") and hasattr(
                self.model.action_head, "num_inference_timesteps"
            ):
                self.model.action_head.num_inference_timesteps = denoising_steps
                print(f"Set action denoising steps to {denoising_steps}")

    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        # Ensure correct dimensions before applying transforms
        return self._modality_transform(obs)

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        return self._modality_transform.unapply(action)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
            "annotation.<>": np.ndarray, # (T, )
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
            "annotation.<>": np.ndarray, # (B, T, )
        }

        Returns:
            Dict[str, Any]: The predicted action.
        """
        # Create a copy to avoid mutating input
        obs_copy = observations.copy()

        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        normalized_input = self.apply_transforms(obs_copy)
        normalized_action = self._get_action_from_normalized_input(normalized_input)
        unnormalized_action = self._get_unnormalized_action(normalized_action)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    def get_action_profiled(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Run get_action and return both action and server-side timing breakdown.

        This endpoint is intended for real client/server LIBERO rollouts. It uses
        the actual observation sent by the LIBERO client and measures server-side
        preprocessing, System 2 backbone, System 1 action head, and postprocess.
        """

        def sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        def elapsed_ms(start):
            sync()
            return (time.perf_counter() - start) * 1000.0

        timings: Dict[str, float] = {}
        sync()
        total_start = time.perf_counter()

        pack_start = time.perf_counter()
        obs_copy = observations.copy()
        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)
        timings["server_input_pack_ms"] = elapsed_ms(pack_start)

        start = time.perf_counter()
        normalized_input, transform_timings = self._apply_transforms_profiled(obs_copy, sync)
        timings["server_preprocess_transform_ms"] = elapsed_ms(start)
        timings.update(transform_timings)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            normalized_action, model_timings = self._get_action_from_normalized_input_profiled(
                normalized_input
            )
        timings.update(model_timings)

        start = time.perf_counter()
        unnormalized_action = self._get_unnormalized_action(normalized_action)
        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        timings["server_postprocess_untransform_ms"] = elapsed_ms(start)
        timings["server_policy_total_ms"] = elapsed_ms(total_start)

        return {
            "__action__": unnormalized_action,
            "__timing__": timings,
            "__memory__": self._server_memory_snapshot(),
        }

    def _model_parameter_buffer_bytes(self) -> int:
        cached = getattr(self, "_model_parameter_buffer_bytes_cache", None)
        if cached is not None:
            return int(cached)
        component_memory = self._model_component_memory_bytes()
        total = component_memory.get("server_model_component_total_bytes")
        if total is not None:
            return int(total)
        seen: set[int] = set()
        total = 0
        for tensor in list(self.model.parameters()) + list(self.model.buffers()):
            ptr = tensor.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            total += tensor.numel() * tensor.element_size()
        self._model_parameter_buffer_bytes_cache = int(total)
        return int(total)

    @staticmethod
    def _component_for_tensor_name(name: str) -> str:
        if name.startswith("backbone.eagle_model.language_model."):
            return "llm"
        if name.startswith("action_head.model."):
            return "dit"
        if name.startswith("backbone.eagle_model.vision_model."):
            return "vision"
        if name.startswith("backbone."):
            return "backbone_other"
        if name.startswith("action_head."):
            return "action_head_other"
        return "other"

    def _model_component_memory_bytes(self) -> Dict[str, int]:
        cached = getattr(self, "_model_component_memory_bytes_cache", None)
        if cached is not None:
            return dict(cached)

        components = {
            "llm": 0,
            "dit": 0,
            "vision": 0,
            "backbone_other": 0,
            "action_head_other": 0,
            "other": 0,
        }
        seen: set[int] = set()
        for name, tensor in list(self.model.named_parameters()) + list(self.model.named_buffers()):
            ptr = tensor.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            component = self._component_for_tensor_name(name)
            components[component] += int(tensor.numel() * tensor.element_size())

        total = sum(components.values())
        out = {
            "server_model_component_total_bytes": int(total),
            "server_model_component_llm_bytes": int(components["llm"]),
            "server_model_component_dit_bytes": int(components["dit"]),
            "server_model_component_llm_plus_dit_bytes": int(
                components["llm"] + components["dit"]
            ),
            "server_model_component_vision_bytes": int(components["vision"]),
            "server_model_component_backbone_other_bytes": int(components["backbone_other"]),
            "server_model_component_action_head_other_bytes": int(components["action_head_other"]),
            "server_model_component_other_bytes": int(components["other"]),
        }
        self._model_component_memory_bytes_cache = dict(out)
        self._model_parameter_buffer_bytes_cache = int(total)
        return out

    def _server_memory_snapshot(self) -> Dict[str, float]:
        component_memory = self._model_component_memory_bytes()
        memory: Dict[str, float] = {
            **{key: float(value) for key, value in component_memory.items()},
            "server_model_parameter_buffer_bytes": float(
                component_memory["server_model_component_total_bytes"]
            ),
            "server_duquant_linear_modules": float(self._count_duquant_modules()),
            "server_duquant_packed_modules": float(self._count_duquant_modules(storage_mode="packed")),
            "server_duquant_fake_modules": float(self._count_duquant_modules(storage_mode="fake")),
            "server_atm_env_enabled": float(
                os.environ.get("GR00T_ATM_ENABLE", "0") not in ("0", "false", "False", "")
            ),
            "server_ohb_env_enabled": float(
                os.environ.get("GR00T_OHB_ENABLE", "0") not in ("0", "false", "False", "")
            ),
        }
        return memory

    def _count_duquant_modules(self, storage_mode: str | None = None) -> int:
        try:
            from gr00t.quantization import DuQuantLinear
        except Exception:
            return 0
        total = 0
        for module in self.model.modules():
            if not isinstance(module, DuQuantLinear):
                continue
            if storage_mode is not None and getattr(module, "storage_mode", None) != storage_mode:
                continue
            total += 1
        return total

    def _apply_transforms_profiled(
        self, obs: Dict[str, Any], sync
    ) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Apply modality transforms and expose a per-transform timing breakdown."""
        transform = self._modality_transform
        if not hasattr(transform, "transforms"):
            start = time.perf_counter()
            out = self.apply_transforms(obs)
            sync()
            return out, {"server_preprocess_unclassified_ms": (time.perf_counter() - start) * 1000.0}

        timings: Dict[str, float] = {}
        grouped = {
            "server_preprocess_video_ms": 0.0,
            "server_preprocess_state_action_ms": 0.0,
            "server_preprocess_concat_ms": 0.0,
            "server_preprocess_gr00t_vlm_ms": 0.0,
            "server_preprocess_other_ms": 0.0,
        }
        data = obs
        for idx, step in enumerate(transform.transforms):
            name = step.__class__.__name__
            start = time.perf_counter()
            try:
                data = step(data)
            except Exception as exc:
                raise ValueError(f"Error applying transform {idx} to data: {exc}") from exc
            sync()
            elapsed = (time.perf_counter() - start) * 1000.0
            timings[f"server_preprocess_{idx:02d}_{name}_ms"] = elapsed
            if name.startswith("Video"):
                grouped["server_preprocess_video_ms"] += elapsed
            elif name.startswith("StateAction"):
                grouped["server_preprocess_state_action_ms"] += elapsed
            elif name == "ConcatTransform":
                grouped["server_preprocess_concat_ms"] += elapsed
            elif name == "GR00TTransform":
                grouped["server_preprocess_gr00t_vlm_ms"] += elapsed
            else:
                grouped["server_preprocess_other_ms"] += elapsed
        timings.update(grouped)
        return data, timings

    def _get_action_from_normalized_input(self, normalized_input: Dict[str, Any]) -> torch.Tensor:
        # Set up autocast context if needed
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = self.model.get_action(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def _get_action_from_normalized_input_profiled(
        self, normalized_input: Dict[str, Any]
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        def sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        def elapsed_ms(start):
            sync()
            return (time.perf_counter() - start) * 1000.0

        def add_module_timer(module, key, handles, timer_totals):
            if module is None:
                return
            state = {"starts": [], "total_ms": 0.0}

            def pre_hook(_module, _inputs):
                sync()
                state["starts"].append(time.perf_counter())

            def post_hook(_module, _inputs, _output):
                sync()
                start = state["starts"].pop() if state["starts"] else None
                if start is not None:
                    state["total_ms"] += (time.perf_counter() - start) * 1000.0

            handles.append(module.register_forward_pre_hook(pre_hook))
            handles.append(module.register_forward_hook(post_hook))
            timer_totals[key] = state

        timings: Dict[str, float] = {}
        timer_totals: Dict[str, Dict[str, Any]] = {}
        timer_handles = []
        backbone = self.model.backbone
        eagle_model = getattr(backbone, "eagle_model", None)
        if eagle_model is not None:
            add_module_timer(
                getattr(eagle_model, "vision_model", None),
                "server_system2_vision_model_ms",
                timer_handles,
                timer_totals,
            )
            add_module_timer(
                getattr(eagle_model, "mlp1", None),
                "server_system2_vision_projector_ms",
                timer_handles,
                timer_totals,
            )
            add_module_timer(
                getattr(eagle_model, "language_model", None),
                "server_system2_reasoning_ms",
                timer_handles,
                timer_totals,
            )
        add_module_timer(
            getattr(backbone, "eagle_linear", None),
            "server_system2_to_system1_bridge_ms",
            timer_handles,
            timer_totals,
        )
        action_head = self.model.action_head
        add_module_timer(
            getattr(action_head, "vlln", None),
            "server_system1_vision_norm_ms",
            timer_handles,
            timer_totals,
        )
        add_module_timer(
            getattr(action_head, "vl_self_attention", None),
            "server_system1_vision_attention_ms",
            timer_handles,
            timer_totals,
        )

        sync()
        total_start = time.perf_counter()

        try:
            start = time.perf_counter()
            backbone_inputs, action_inputs = self.model.prepare_input(normalized_input)
            timings["server_model_prepare_input_to_device_ms"] = elapsed_ms(start)

            start = time.perf_counter()
            backbone_outputs = self.model.backbone(backbone_inputs)
            timings["server_system2_backbone_ms"] = elapsed_ms(start)

            start = time.perf_counter()
            action_head_outputs = self.model.action_head.get_action(backbone_outputs, action_inputs)
            timings["server_system1_action_head_ms"] = elapsed_ms(start)

            start = time.perf_counter()
            self.model.validate_data(action_head_outputs, backbone_outputs, is_training=False)
            normalized_action = action_head_outputs["action_pred"].float()
            timings["server_model_validate_cast_ms"] = elapsed_ms(start)
            timings["server_model_total_ms"] = elapsed_ms(total_start)
        finally:
            for handle in timer_handles:
                handle.remove()

        for key, state in timer_totals.items():
            timings[key] = float(state["total_ms"])
        timings["server_system2_vision_ms"] = (
            timings.get("server_system2_vision_model_ms", 0.0)
            + timings.get("server_system2_vision_projector_ms", 0.0)
        )
        timings["server_system2_other_ms"] = max(
            0.0,
            timings.get("server_system2_backbone_ms", 0.0)
            - (
                timings.get("server_system2_vision_ms", 0.0)
                + timings.get("server_system2_reasoning_ms", 0.0)
                + timings.get("server_system2_to_system1_bridge_ms", 0.0)
            ),
        )
        timings["server_system1_vision_ms"] = (
            timings.get("server_system1_vision_norm_ms", 0.0)
            + timings.get("server_system1_vision_attention_ms", 0.0)
        )
        timings["server_system1_action_ms"] = max(
            0.0,
            timings.get("server_system1_action_head_ms", 0.0)
            - timings.get("server_system1_vision_ms", 0.0),
        )
        return normalized_action, timings

    def _get_unnormalized_action(self, normalized_action: torch.Tensor) -> Dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()})

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Get the modality config for the model, overrides the base class method
        """
        return self._modality_config

    @property
    def modality_config(self) -> Dict[str, ModalityConfig]:
        return self._modality_config

    @property
    def modality_transform(self) -> ComposedModalityTransform:
        return self._modality_transform

    @property
    def video_delta_indices(self) -> np.ndarray:
        """Get the video delta indices."""
        return self._video_delta_indices

    @property
    def state_delta_indices(self) -> np.ndarray | None:
        """Get the state delta indices."""
        return self._state_delta_indices

    @property
    def denoising_steps(self) -> int:
        """Get the number of denoising steps."""
        return self.model.action_head.num_inference_timesteps

    @denoising_steps.setter
    def denoising_steps(self, value: int):
        """Set the number of denoising steps."""
        self.model.action_head.num_inference_timesteps = value

    def _check_state_is_batched(self, obs: Dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def _load_model(self, model_path):
        model = GR00T_N1_5.from_pretrained(model_path, torch_dtype=COMPUTE_DTYPE)
        model.eval()  # Set model to eval mode

        # Update action_horizon to match modality config
        # Get the expected action horizon from the modality config
        expected_action_horizon = len(self._modality_config["action"].delta_indices)

        if expected_action_horizon != model.action_head.config.action_horizon:
            print(
                f"Policy: Recreating action head with action_horizon {expected_action_horizon} (was {model.action_head.config.action_horizon})"
            )

            # Update the action head config
            new_action_head_config = model.action_head.config
            new_action_head_config.action_horizon = expected_action_horizon

            # Import the FlowmatchingActionHead class
            from gr00t.model.action_head.flow_matching_action_head import (
                FlowmatchingActionHead,
            )

            # Create new action head with updated config
            new_action_head = FlowmatchingActionHead(new_action_head_config)

            # Copy the weights from the old action head to the new one
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)

            # Replace the action head
            model.action_head = new_action_head

            # Update model config AND the action_head_cfg dictionary that gets saved
            model.config.action_horizon = expected_action_horizon
            model.action_horizon = expected_action_horizon
            model.config.action_head_cfg["action_horizon"] = expected_action_horizon

            print("[GR00T] Action head was recreated - will re-apply DuQuant if configured")

        # Ensure DiT attention processors are ATM-aware before any quantization or scaling is applied
        try:
            ensure_dit_attention_patch(model)
        except Exception as e:
            print(f"[GR00T] Failed to patch attention for ATM support: {e}")

        # Apply selective FP8 Linear conversion for large LLM matmuls
        # (gate/up/down proj) when GR00T_FP8_MODE=1 — replaces DuQuant/ATM.
        if os.environ.get("GR00T_FP8_MODE", "0") == "1":
            from gr00t.quantization.fp8_linear import convert_to_fp8_linear
            print("[GR00T] Selective FP8 mode: converting large LLM matmuls to FP8...")
            n_fp8 = convert_to_fp8_linear(model, verbose=True)
            print(f"[GR00T] FP8 conversion done: {n_fp8} layers converted")
        else:
            # Apply DuQuant W4A8 quantization if configured via environment variables
            # This must be done BEFORE moving model to device
            # IMPORTANT: This is called AFTER action_head recreation to ensure DiT layers are quantized
            try:
                from gr00t.quantization import enable_duquant_if_configured
                enable_duquant_if_configured(model)
            except Exception as e:
                print(f"[GR00T] DuQuant not enabled or failed to apply: {e}")

            # Apply ATM scaling if configured (uses pre-loaded alpha JSON)
            try:
                enable_dit_atm_if_configured(model)
            except Exception as e:
                print(f"[GR00T] ATM not enabled or failed to apply: {e}")

        model.to(device=self.device)  # type: ignore

        # torch.compile strategy (GR00T_TORCH_COMPILE=1):
        #   DiT (action_head): max-autotune + CUDA graphs — fixed shapes → 1.45x
        #   LLM (backbone):    default mode — BUT skip if DuQuant is active
        #                      (DuQuant custom layers cause graph breaks)
        if os.environ.get("GR00T_TORCH_COMPILE", "0") == "1":
            import torch
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.enable_flash_sdp(True)

            # DiT action_head: always safe to compile (BF16, fixed shapes)
            if hasattr(model, 'action_head') and model.action_head is not None:
                try:
                    model.action_head = torch.compile(
                        model.action_head, mode="max-autotune"
                    )
                    print("[GR00T]   action_head (DiT): max-autotune + CUDA graphs")
                except Exception as e:
                    print(f"[GR00T]   action_head compile failed: {e}")

            # LLM backbone: compile by default even with DuQuant. The custom
            # layers may graph-break, but the surrounding Eagle blocks still
            # benefit on Blackwell/Ada GPUs; keep an env kill switch for
            # machines where this is unstable.
            if hasattr(model, 'backbone') and model.backbone is not None:
                duquant_active = os.environ.get("GR00T_DUQUANT_WBITS_DEFAULT", "").strip() not in ("", "0")
                compile_duquant_backbone = os.environ.get(
                    "GR00T_DUQUANT_COMPILE_BACKBONE", "1"
                ) not in ("0", "false", "False")
                if duquant_active and not compile_duquant_backbone:
                    print("[GR00T]   backbone: SKIP compile (DuQuant custom layers active)")
                else:
                    try:
                        model.backbone = torch.compile(
                            model.backbone, mode="default"
                        )
                        if duquant_active:
                            print("[GR00T]   backbone (DuQuant LLM): default")
                        else:
                            print("[GR00T]   backbone (LLM): default")
                    except Exception as e:
                        print(f"[GR00T]   backbone compile failed: {e}")

            print("[GR00T] torch.compile applied successfully")

        self.model = model

    def _load_metadata(self, exp_cfg_dir: Path):
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)

        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

    def _load_horizons(self):
        """Load the horizons needed for the model."""
        # Get modality configs
        # Video horizons
        self._video_delta_indices = np.array(self._modality_config["video"].delta_indices)
        self._assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        # State horizons (if used)
        if "state" in self._modality_config:
            self._state_delta_indices = np.array(self._modality_config["state"].delta_indices)
            self._assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None

    def _assert_delta_indices(self, delta_indices: np.ndarray):
        """Assert that the delta indices are valid."""
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"


#######################################################################################################


# Helper functions
def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.expand_dims(np.array(v), axis=0)  # Fixed
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v, axis=0)  # Fixed: only remove batch dim
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze(0)  # Fixed: only remove batch dim
        else:
            squeezed_data[k] = v
    return squeezed_data
