"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import importlib.util
import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

OPENVLA_ROOT = Path(__file__).resolve().parents[1]
if str(OPENVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENVLA_ROOT))

from experiments.robot.openvla_direct_quant import (
    DirectQuantConfig,
    estimate_model_size_mb,
    get_direct_quant_info,
    quantize_openvla_language_model,
)
from experiments.robot.openvla_fast_action import (
    can_use_fast_action_head,
    can_use_last_token_logits,
    limit_llm_layers,
    predict_action_fast,
    predict_action_last_logits,
)

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class OpenVLAServer:
    def __init__(
        self,
        openvla_path: Union[str, Path],
        attn_implementation: Optional[str] = "auto",
        direct_quant_mode: str = "none",
        direct_quant_group_size: int = 128,
        bnb_int8_threshold: float = 0.0,
        bnb_4bit_compute_dtype: str = "auto",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        fp8_activation_scale: float = 1.0,
        fast_action_head: bool = False,
        last_token_logits: bool = True,
        drop_full_attention_mask: bool = True,
        max_vision_tokens: int = 0,
        vision_token_strategy: str = "uniform",
        llm_max_layers: int = 0,
        llm_layer_strategy: str = "first",
    ) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.openvla_path, self.attn_implementation = openvla_path, attn_implementation
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.direct_quant_mode = direct_quant_mode
        self.fast_action_head_requested = fast_action_head
        self.last_token_logits_requested = last_token_logits
        self.drop_full_attention_mask = drop_full_attention_mask
        self.max_vision_tokens = max_vision_tokens
        self.vision_token_strategy = vision_token_strategy
        attn_implementation = resolve_attn_implementation(attn_implementation, self.device)
        self.attn_implementation = attn_implementation

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        patch_local_transformers_config(self.vla)
        original_layers, active_layers = limit_llm_layers(self.vla, llm_max_layers, llm_layer_strategy)
        if active_layers and active_layers < original_layers:
            print(f"[latency] LLM layers limited: {original_layers}->{active_layers} ({llm_layer_strategy})")

        report = None
        if self.direct_quant_mode not in {None, "", "none", "fp16", "bf16"}:
            report = quantize_openvla_language_model(
                self.vla,
                DirectQuantConfig(
                    mode=self.direct_quant_mode,
                    group_size=direct_quant_group_size,
                    bnb_int8_threshold=bnb_int8_threshold,
                    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                    fp8_activation_scale=fp8_activation_scale,
                ),
            )

        self.vla = self.vla.to(self.device)
        self.fast_action_head = self.fast_action_head_requested and can_use_fast_action_head(self.vla)
        self.last_token_logits = self.last_token_logits_requested and can_use_last_token_logits(self.vla)
        if self.fast_action_head_requested and not self.fast_action_head:
            print("[fast-action] action-only lm_head path is unavailable for this model; using full vocab logits.")
        if self.last_token_logits_requested and not self.last_token_logits:
            print("[last-logits] backbone-only prefill path is unavailable; using full language_model logits.")
        refresh_direct_quant_report_sizes(self.vla)
        if report is not None:
            updated_report = get_direct_quant_info(self.vla)
            print(
                "[*] Direct quantization summary: "
                f"mode={self.direct_quant_mode}, replaced={report.replaced_linear_layers}, "
                f"model_size={updated_report['original_model_size_mb']:.2f}->"
                f"{updated_report['quantized_model_size_mb']:.2f} MB"
            )

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if os.path.isdir(self.openvla_path):
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)

    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            image, instruction = payload["image"], payload["instruction"]
            unnorm_key = payload.get("unnorm_key", None)

            # Run VLA Inference
            prompt = get_openvla_prompt(instruction, self.openvla_path)
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            if self.fast_action_head:
                action = predict_action_fast(
                    self.vla,
                    **inputs,
                    unnorm_key=unnorm_key,
                    max_vision_tokens=self.max_vision_tokens,
                    vision_token_strategy=self.vision_token_strategy,
                )
            elif self.last_token_logits:
                action = predict_action_last_logits(
                    self.vla,
                    **inputs,
                    unnorm_key=unnorm_key,
                    drop_full_attention_mask=self.drop_full_attention_mask,
                )
            else:
                action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off
    openvla_path: Union[str, Path] = "openvla/openvla-7b"               # HF Hub Path (or path to local run directory)
    attn_implementation: str = "auto"                                   # auto | sdpa | eager | flash_attention_2
    direct_quant_mode: str = "none"                                     # none | w4a8 | w8a16 | w8a8 | w4a16 | bnb_*
    direct_quant_group_size: int = 128                                  # Group size for direct W4A16 quantization
    bnb_int8_threshold: float = 0.0                                     # 0.0 fastest; 6.0 accuracy-oriented
    bnb_4bit_compute_dtype: str = "auto"                                # auto | bf16 | fp16 | fp32
    bnb_4bit_quant_type: str = "nf4"                                    # nf4 | fp4
    bnb_4bit_use_double_quant: bool = True                              # Compress 4-bit quantization statistics
    fp8_activation_scale: float = 1.0                                   # Fixed activation scale for w4a8
    fast_action_head: bool = False                                      # Non-quant action-head optimization
    last_token_logits: bool = True                                      # Exact optimization: project only final prefill hidden state
    drop_full_attention_mask: bool = True                               # Drop all-ones attention masks
    max_vision_tokens: int = 0                                          # 0 keeps all visual tokens
    vision_token_strategy: str = "uniform"                              # uniform | pool | first
    llm_max_layers: int = 0                                             # 0 keeps all LLM layers
    llm_layer_strategy: str = "first"                                   # first | uniform | last

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port

    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(
        cfg.openvla_path,
        attn_implementation=cfg.attn_implementation,
        direct_quant_mode=cfg.direct_quant_mode,
        direct_quant_group_size=cfg.direct_quant_group_size,
        bnb_int8_threshold=cfg.bnb_int8_threshold,
        bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        fp8_activation_scale=cfg.fp8_activation_scale,
        fast_action_head=cfg.fast_action_head,
        last_token_logits=cfg.last_token_logits,
        drop_full_attention_mask=cfg.drop_full_attention_mask,
        max_vision_tokens=cfg.max_vision_tokens,
        vision_token_strategy=cfg.vision_token_strategy,
        llm_max_layers=cfg.llm_max_layers,
        llm_layer_strategy=cfg.llm_layer_strategy,
    )
    server.run(cfg.host, port=cfg.port)


def patch_local_transformers_config(model: torch.nn.Module) -> None:
    """Fill optional config fields expected by locally patched transformers builds."""
    configs = [getattr(model, "config", None)]
    if hasattr(model, "language_model"):
        configs.append(getattr(model.language_model, "config", None))
    configs.append(getattr(getattr(model, "config", None), "text_config", None))
    for config in configs:
        if config is not None and not hasattr(config, "proportion_attn_var"):
            setattr(config, "proportion_attn_var", None)


def refresh_direct_quant_report_sizes(model: torch.nn.Module) -> None:
    report = getattr(model, "_direct_quant_report", None)
    if not isinstance(report, dict) or report.get("mode") == "none":
        return

    report["quantized_model_size_mb"] = estimate_model_size_mb(model)
    if hasattr(model, "language_model"):
        report["quantized_target_size_mb"] = estimate_model_size_mb(model.language_model)

    original_target_size_mb = float(report.get("original_target_size_mb", 0.0) or 0.0)
    original_model_size_mb = float(report.get("original_model_size_mb", 0.0) or 0.0)
    quantized_target_size_mb = float(report.get("quantized_target_size_mb", 0.0) or 0.0)
    quantized_model_size_mb = float(report.get("quantized_model_size_mb", 0.0) or 0.0)
    report["target_size_reduction_percent"] = _percent_reduction(original_target_size_mb, quantized_target_size_mb)
    report["model_size_reduction_percent"] = _percent_reduction(original_model_size_mb, quantized_model_size_mb)


def _percent_reduction(baseline: float, current: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - current) / baseline * 100.0


def resolve_attn_implementation(attn_implementation: Optional[str], device: torch.device) -> str:
    normalized = str(attn_implementation or "auto").lower()
    if normalized == "auto":
        normalized = "flash_attention_2" if device.type == "cuda" else "eager"
    if normalized == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        print("[attn] flash_attention_2 requested but flash_attn is not installed; falling back to sdpa.", flush=True)
        return "sdpa"
    return normalized


if __name__ == "__main__":
    deploy()
