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
import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

import gr00t
from gr00t.model.fastv import (
    infer_visual_token_indices,
    install_fastv_qwen3_forward,
    set_attention_implementation,
)

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


class EagleBackbone(nn.Module):

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str | None = None,
        project_to_dim: int = 1536,
        use_fastv: bool = False,
        fastv_k: int = 2,
        fastv_r: float = 0.5,
        image_token_start_index: int | None = None,
        image_token_length: int | None = None,
        fastv_verbose: bool = False,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        env_fastv = os.environ.get("GR00T_FASTV_ENABLE", "")
        use_fastv = use_fastv or env_fastv.lower() in ("1", "true", "yes", "on")
        attn_implementation = os.environ.get("GR00T_ATTN_IMPLEMENTATION", "sdpa")
        if use_fastv:
            attn_implementation = "eager"
        config = AutoConfig.from_pretrained(
            DEFAULT_EAGLE_PATH,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
        config._attn_implementation = attn_implementation
        if hasattr(config, "vision_config"):
            config.vision_config._attn_implementation = attn_implementation
        if hasattr(config, "text_config"):
            config.text_config._attn_implementation = attn_implementation
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.configure_fastv(
            use_fastv=use_fastv,
            fastv_k=int(os.environ.get("GR00T_FASTV_K", fastv_k)),
            fastv_r=float(os.environ.get("GR00T_FASTV_R", fastv_r)),
            image_token_start_index=image_token_start_index,
            image_token_length=image_token_length,
            verbose=fastv_verbose
            or os.environ.get("GR00T_FASTV_VERBOSE", "").lower() in ("1", "true", "yes", "on"),
        )
        self.set_trainable_parameters(tune_llm, tune_visual)

    def configure_fastv(
        self,
        use_fastv: bool = False,
        fastv_k: int = 2,
        fastv_r: float = 0.5,
        image_token_start_index: int | None = None,
        image_token_length: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.fastv_config = {
            "use_fastv": bool(use_fastv),
            "fastv_k": int(fastv_k),
            "fastv_r": float(fastv_r),
            "image_token_start_index": image_token_start_index,
            "image_token_length": image_token_length,
            "preserve_text_tokens": True,
            "preserve_special_tokens": True,
            "preserve_state_tokens": True,
            "preserve_action_tokens": True,
            "verbose": bool(verbose),
        }
        if use_fastv:
            qwen3_model = self.eagle_model.language_model.model
            install_fastv_qwen3_forward(qwen3_model)
            set_attention_implementation(self.eagle_model.language_model, "eager")
            print(
                "[FastV] enabled for System 2 VLM "
                f"(k={fastv_k}, r={fastv_r}, attention=eager)"
            )

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]

        fastv_config = None
        if self.fastv_config.get("use_fastv", False):
            visual_indices, image_start, image_length, contiguous = infer_visual_token_indices(
                eagle_input["input_ids"],
                self.eagle_model.image_token_index,
                self.fastv_config.get("image_token_start_index"),
                self.fastv_config.get("image_token_length"),
            )
            fastv_config = dict(self.fastv_config)
            fastv_config["visual_token_indices"] = visual_indices
            fastv_config["image_token_start_index"] = image_start
            fastv_config["image_token_length"] = image_length
            if self.fastv_config.get("verbose", False):
                print(
                    "[FastV] image tokens "
                    f"start={image_start} length={image_length} contiguous={contiguous}"
                )

        eagle_output = self.eagle_model(
            **eagle_input,
            output_hidden_states=True,
            return_dict=True,
            fastv_config=fastv_config,
        )
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        qwen3_model = self.eagle_model.language_model.model
        eagle_mask = getattr(qwen3_model, "_fastv_last_attention_mask", None)
        if eagle_mask is None:
            eagle_mask = eagle_input["attention_mask"]
        return eagle_features, eagle_mask

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask = self.forward_eagle(vl_input)

        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used in the forward pass for DDP compatibility
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=eagle_embeds.device, dtype=eagle_embeds.dtype, requires_grad=True
            )
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            eagle_embeds = eagle_embeds + dummy_term

        return BatchFeature(
            data={"backbone_features": eagle_embeds, "backbone_attention_mask": eagle_mask}
        )  # [B, T2, hidden_size]
