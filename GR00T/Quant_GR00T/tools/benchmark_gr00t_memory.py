#!/usr/bin/env python3
"""Measure GR00T memory footprint for the current baseline/quantized environment."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.quantvla_bench_common import (
    count_duquant_modules,
    duquant_storage_summary,
    load_libero_policy,
    parameter_bytes,
    runtime_metadata,
    sample_libero_observation,
    save_json,
    sync_cuda,
)


def bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024**3)


def fmt_gib(value: int | None) -> str:
    gib = bytes_to_gib(value)
    if gib is None:
        return "n/a"
    return f"{gib:.3f}GiB"


def component_for_tensor(name: str) -> str:
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


def component_memory(model: torch.nn.Module) -> dict[str, Any]:
    components: dict[str, dict[str, int]] = {}
    seen: set[int] = set()
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        ptr = tensor.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        component = component_for_tensor(name)
        entry = components.setdefault(component, {"bytes": 0, "numel": 0, "tensors": 0})
        entry["bytes"] += int(tensor.numel() * tensor.element_size())
        entry["numel"] += int(tensor.numel())
        entry["tensors"] += 1

    total_bytes = sum(entry["bytes"] for entry in components.values())
    llm_bytes = components.get("llm", {}).get("bytes", 0)
    dit_bytes = components.get("dit", {}).get("bytes", 0)
    return {
        "total_bytes": int(total_bytes),
        "llm_bytes": int(llm_bytes),
        "dit_bytes": int(dit_bytes),
        "llm_plus_dit_bytes": int(llm_bytes + dit_bytes),
        "components": components,
    }


def memory_summary_lines(result: dict[str, Any]) -> list[str]:
    model = result.get("model", {})
    components = result.get("model_component_memory", {})
    lines = [
        "[memory] GR00T model component memory footprint",
        (
            "[memory] "
            f"total={fmt_gib(components.get('total_bytes'))}, "
            f"llm={fmt_gib(components.get('llm_bytes'))}, "
            f"dit={fmt_gib(components.get('dit_bytes'))}, "
            f"llm+dit={fmt_gib(components.get('llm_plus_dit_bytes'))}, "
            f"duquant_layers={model.get('duquant_linear_modules', 'n/a')}, "
            f"duquant_storage={model.get('duquant_storage', {})}, "
            f"denoising_steps={model.get('denoising_steps', 'n/a')}"
        ),
    ]
    for component, entry in sorted(components.get("components", {}).items()):
        lines.append(
            "[memory] "
            f"component={component} bytes={fmt_gib(entry.get('bytes'))} "
            f"numel={entry.get('numel', 0)} tensors={entry.get('tensors', 0)}"
        )
    return lines


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_cpu and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Refusing to run the memory benchmark on CPU. "
            "Run in the GPU-enabled conda environment, or pass --allow-cpu only for debugging."
        )

    result: dict[str, Any] = {"metadata": runtime_metadata(args.variant)}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    load_start = time.perf_counter()
    policy = load_libero_policy(
        model_path=args.model_path,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )
    sync_cuda()
    result["load_seconds"] = time.perf_counter() - load_start
    result["model"] = {
        "parameter_and_buffer_bytes": parameter_bytes(policy.model),
        "duquant_linear_modules": count_duquant_modules(policy.model),
        "duquant_storage": duquant_storage_summary(policy.model),
        "denoising_steps": int(policy.denoising_steps),
    }
    result["model_component_memory"] = component_memory(policy.model)

    obs = sample_libero_observation(args.image_height, args.image_width, args.seed)

    for _ in range(args.warmup):
        policy.get_action(obs)
    sync_cuda()
    result["warmup_iters"] = int(args.warmup)

    infer_start = time.perf_counter()
    for _ in range(args.iters):
        policy.get_action(obs)
    sync_cuda()
    result["inference_seconds"] = time.perf_counter() - infer_start
    result["inference_iters"] = int(args.iters)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="")
    parser.add_argument("--model-path", default="youliangtan/gr00t-n1.5-libero-long-posttrain")
    parser.add_argument(
        "--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig"
    )
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--append-log", default="")
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
    print(f"Wrote memory benchmark to {args.output_json}")
    lines = memory_summary_lines(result)
    for line in lines:
        print(line)
    if args.append_log:
        log_path = Path(args.append_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            for line in lines:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
