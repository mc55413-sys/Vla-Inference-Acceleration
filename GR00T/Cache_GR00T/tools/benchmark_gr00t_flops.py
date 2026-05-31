#!/usr/bin/env python3
"""Estimate dense-equivalent theoretical FLOPs for one GR00T get_action call."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.quantvla_bench_common import (
    count_duquant_modules,
    load_libero_policy,
    runtime_metadata,
    sample_libero_observation,
    save_json,
    sync_cuda,
)


def prod(values) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def category_for_name(name: str) -> str:
    if name.startswith("backbone.eagle_model.vision_model.") or name.startswith(
        "backbone.eagle_model.mlp1."
    ):
        return "system2_vision"
    if name.startswith("backbone.eagle_model.language_model."):
        return "system2_reasoning"
    if name.startswith("backbone.eagle_linear"):
        return "system2_to_system1_bridge"
    if name.startswith("backbone."):
        return "system2_other"
    if name.startswith("action_head.vlln.") or name.startswith("action_head.vl_self_attention."):
        return "system1_vision"
    if name.startswith("action_head.model."):
        return "system1_action"
    if name.startswith("action_head."):
        return "system1_action"
    return "other"


def linear_flops(module: Any, inputs: tuple[Any, ...], output: Any) -> int:
    if not inputs or not torch.is_tensor(inputs[0]):
        return 0
    x = inputs[0]
    if x.numel() == 0:
        return 0
    in_features = int(getattr(module, "in_features"))
    out_features = int(getattr(module, "out_features"))
    tokens = x.numel() // in_features
    return int(2 * tokens * in_features * out_features)


def conv2d_flops(module: nn.Conv2d, inputs: tuple[Any, ...], output: Any) -> int:
    if not torch.is_tensor(output):
        return 0
    batch = int(output.shape[0])
    out_channels = int(output.shape[1])
    out_h = int(output.shape[2])
    out_w = int(output.shape[3])
    kernel_ops = int(module.kernel_size[0] * module.kernel_size[1] * module.in_channels / module.groups)
    return int(2 * batch * out_channels * out_h * out_w * kernel_ops)


class FlopAccumulator:
    def __init__(self, model: nn.Module):
        self.model = model
        self.by_module: dict[str, int] = defaultdict(int)
        self.by_category: dict[str, int] = defaultdict(int)
        self.duquant_flops = 0
        self.handles = []
        try:
            from gr00t.quantization import DuQuantLinear
        except Exception:
            DuQuantLinear = None
        self.duquant_cls = DuQuantLinear

    def _hook(self, name: str, module: nn.Module):
        def inner(mod, inputs, output):
            flops = 0
            if isinstance(mod, nn.Linear):
                flops = linear_flops(mod, inputs, output)
            elif isinstance(mod, nn.Conv2d):
                flops = conv2d_flops(mod, inputs, output)
            elif self.duquant_cls is not None and isinstance(mod, self.duquant_cls):
                flops = linear_flops(mod, inputs, output)
                self.duquant_flops += flops
            if flops:
                self.by_module[name] += flops
                self.by_category[category_for_name(name)] += flops

        return inner

    def __enter__(self):
        for name, module in self.model.named_modules():
            is_target = isinstance(module, (nn.Linear, nn.Conv2d))
            if self.duquant_cls is not None and isinstance(module, self.duquant_cls):
                is_target = True
            if is_target:
                self.handles.append(module.register_forward_hook(self._hook(name, module)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_cpu and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Refusing to run FLOPs tracing on CPU. "
            "Run in the GPU-enabled conda environment, or pass --allow-cpu only for debugging."
        )

    policy = load_libero_policy(
        model_path=args.model_path,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )
    obs = sample_libero_observation(args.image_height, args.image_width, args.seed)

    for _ in range(args.warmup):
        policy.get_action(obs)
    sync_cuda()

    with FlopAccumulator(policy.model) as acc:
        policy.get_action(obs)
    sync_cuda()

    total = sum(acc.by_category.values())
    top_modules = sorted(acc.by_module.items(), key=lambda item: item[1], reverse=True)[
        : args.top_k
    ]
    return {
        "metadata": runtime_metadata(args.variant),
        "model": {
            "duquant_linear_modules": count_duquant_modules(policy.model),
            "denoising_steps": int(policy.denoising_steps),
        },
        "notes": [
            "Counts dense-equivalent FLOPs for nn.Linear, nn.Conv2d, and DuQuantLinear hooks.",
            "Functional attention matmuls/SDPA internals are not separately counted unless expressed through hooked modules.",
            "System split: system2_vision, system2_reasoning, system2_to_system1_bridge, system1_vision, system1_action.",
            "For packed-weight DuQuant, this is still dense-equivalent math, not fused low-bit kernel FLOPs.",
        ],
        "total_flops": int(total),
        "total_tflops": total / 1e12,
        "duquant_linear_dense_equivalent_flops": int(acc.duquant_flops),
        "duquant_linear_dense_equivalent_tflops": acc.duquant_flops / 1e12,
        "by_category": {
            key: {"flops": int(value), "tflops": value / 1e12}
            for key, value in sorted(acc.by_category.items())
        },
        "top_modules": [
            {"name": name, "flops": int(value), "tflops": value / 1e12}
            for name, value in top_modules
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="")
    parser.add_argument("--model-path", default="youliangtan/gr00t-n1.5-libero-long-posttrain")
    parser.add_argument(
        "--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig"
    )
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback for debugging. Real benchmark runs should not use this.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    save_json(args.output_json, result)
    print(f"Wrote FLOPs benchmark to {args.output_json}")


if __name__ == "__main__":
    main()
