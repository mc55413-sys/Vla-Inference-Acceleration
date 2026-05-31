#!/usr/bin/env python3
"""Profile GR00T baseline or FastV with strict VLA latency fields."""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gr00t.experiment.data_config import load_data_config
from gr00t.model.fastv import estimate_transformer_tflops, infer_visual_token_indices
from gr00t.model.policy import Gr00tPolicy, squeeze_dict_values, unsqueeze_dict_values
from tools.pruning_bench_common import (
    parameter_bytes,
    runtime_metadata,
    sample_libero_observation,
    save_json,
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() not in ("0", "false", "no", "off", "")


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StageTimer:
    def __init__(self, use_cuda_event: bool):
        self.use_cuda_event = use_cuda_event and torch.cuda.is_available()
        self.elapsed_ms = 0.0

    def __enter__(self):
        if self.use_cuda_event:
            sync_cuda()
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.use_cuda_event:
            self.end_event.record()
            sync_cuda()
            self.elapsed_ms = float(self.start_event.elapsed_time(self.end_event))
        else:
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0


@contextlib.contextmanager
def module_stage_timer(module: torch.nn.Module | None, totals: dict[str, float], key: str):
    if module is None:
        yield
        return

    starts: list[Any] = []
    handles = []
    use_cuda_event = torch.cuda.is_available()

    def pre_hook(_module, _inputs):
        if use_cuda_event:
            sync_cuda()
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            starts.append(event)
        else:
            starts.append(time.perf_counter())

    def post_hook(_module, _inputs, _output):
        start = starts.pop() if starts else None
        if start is None:
            return
        if use_cuda_event:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            sync_cuda()
            totals[key] = totals.get(key, 0.0) + float(start.elapsed_time(end))
        else:
            totals[key] = totals.get(key, 0.0) + (time.perf_counter() - start) * 1000.0

    handles.append(module.register_forward_pre_hook(pre_hook))
    handles.append(module.register_forward_hook(post_hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def prepare_policy_input(policy: Gr00tPolicy, observations: dict[str, Any]):
    obs_copy = observations.copy()
    is_batch = policy._check_state_is_batched(obs_copy)
    if not is_batch:
        obs_copy = unsqueeze_dict_values(obs_copy)
    for key, value in obs_copy.items():
        if not isinstance(value, np.ndarray):
            obs_copy[key] = np.array(value)
    return obs_copy, is_batch


def apply_fastv_config(policy: Gr00tPolicy, args: argparse.Namespace) -> None:
    backbone = policy.model.backbone
    backbone.configure_fastv(
        use_fastv=args.use_fastv,
        fastv_k=args.fastv_k,
        fastv_r=args.fastv_r,
        image_token_start_index=args.image_token_start_index,
        image_token_length=args.image_token_length,
        verbose=args.verbose_steps,
    )


def model_size_mb(model: torch.nn.Module) -> float:
    return parameter_bytes(model) / (1024.0 * 1024.0)


def cuda_memory_stats() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {
            "peak_cuda_memory_mb": None,
            "allocated_cuda_memory_mb": None,
            "reserved_cuda_memory_mb": None,
        }
    return {
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
        "allocated_cuda_memory_mb": torch.cuda.memory_allocated() / (1024.0 * 1024.0),
        "reserved_cuda_memory_mb": torch.cuda.memory_reserved() / (1024.0 * 1024.0),
    }


def dtype_bits(model: torch.nn.Module) -> int | None:
    for tensor in model.parameters():
        if tensor.is_floating_point():
            return tensor.element_size() * 8
    return None


def infer_tflops_inputs(policy: Gr00tPolicy, backbone_inputs, args: argparse.Namespace) -> dict[str, Any]:
    language_model = policy.model.backbone.eagle_model.language_model
    text_config = language_model.config
    input_ids = backbone_inputs["eagle_input_ids"]
    visual_indices, _start, image_length, _contiguous = infer_visual_token_indices(
        input_ids,
        policy.model.backbone.eagle_model.image_token_index,
        args.image_token_start_index,
        args.image_token_length,
    )
    seq_len = int(input_ids.shape[1])
    m = int(image_length)
    n = seq_len - m
    rho = 1.0 - float(args.fastv_r) if args.use_fastv else 1.0
    k = int(args.fastv_k) if args.use_fastv else 1
    return {
        "N": n,
        "M": m,
        "rho": rho,
        "K": k,
        "T": len(language_model.model.layers),
        "d": int(text_config.hidden_size),
        "ffn_dim": int(text_config.intermediate_size),
        "visual_token_indices_shape": list(visual_indices.shape),
    }


def run_one_iteration(
    policy: Gr00tPolicy,
    args: argparse.Namespace,
    iteration: int,
    measured: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    prefix = "[Measure]" if measured else "[Warmup]"
    if args.verbose_steps:
        print(f"{prefix} iter={iteration} Data start", flush=True)
    with StageTimer(use_cuda_event=False) as timer:
        obs = sample_libero_observation(args.image_height, args.image_width, args.seed + iteration)
    data_ms = timer.elapsed_ms

    if args.verbose_steps:
        print(f"{prefix} iter={iteration} Preprocess start", flush=True)
    with StageTimer(use_cuda_event=False) as timer:
        obs_copy, is_batch = prepare_policy_input(policy, obs)
        normalized_input = policy.apply_transforms(obs_copy)
        backbone_inputs, action_inputs = policy.model.prepare_input(normalized_input)
    preprocess_ms = timer.elapsed_ms

    tflops_inputs = None
    if measured and iteration == 0:
        tflops_inputs = infer_tflops_inputs(policy, backbone_inputs, args)

    module_totals: dict[str, float] = {}
    eagle_model = policy.model.backbone.eagle_model
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if args.verbose_steps:
        print(f"{prefix} iter={iteration} Vision+LLM backbone start", flush=True)
    with torch.inference_mode():
        with module_stage_timer(eagle_model.vision_model, module_totals, "vision_model_ms"):
            with module_stage_timer(eagle_model.mlp1, module_totals, "vision_projector_ms"):
                with module_stage_timer(eagle_model.language_model, module_totals, "llm_backbone_ms"):
                    with module_stage_timer(policy.model.backbone.eagle_linear, module_totals, "llm_bridge_ms"):
                        backbone_outputs = policy.model.backbone(backbone_inputs)

        if args.verbose_steps:
            print(f"{prefix} iter={iteration} Action start", flush=True)
        with StageTimer(use_cuda_event=True) as timer:
            action_outputs = policy.model.action_head.get_action(backbone_outputs, action_inputs)
        action_ms = timer.elapsed_ms

        with StageTimer(use_cuda_event=False):
            normalized_action = action_outputs["action_pred"].float()
            unnormalized_action = policy._get_unnormalized_action(normalized_action)
            if not is_batch:
                _ = squeeze_dict_values(unnormalized_action)

    vision_ms = module_totals.get("vision_model_ms", 0.0) + module_totals.get("vision_projector_ms", 0.0)
    llm_ms = module_totals.get("llm_backbone_ms", 0.0) + module_totals.get("llm_bridge_ms", 0.0)
    model_latency_ms = vision_ms + llm_ms + action_ms
    end_to_end_latency_ms = data_ms + preprocess_ms + vision_ms + llm_ms + action_ms

    sample = {
        "iteration": iteration,
        "measured": measured,
        "model_name": args.model_path,
        "pruning_method": "fastv" if args.use_fastv else "none",
        "weight_bits": dtype_bits(policy.model),
        "activation_bits": dtype_bits(policy.model),
        "kv_cache_bits": None,
        "data_ms": data_ms,
        "preprocess_ms": preprocess_ms,
        "vision_ms": vision_ms,
        "llm_ms": llm_ms,
        "action_ms": action_ms,
        "model_latency_ms": model_latency_ms,
        "end_to_end_latency_ms": end_to_end_latency_ms,
        "vision_model_ms": module_totals.get("vision_model_ms", 0.0),
        "vision_projector_ms": module_totals.get("vision_projector_ms", 0.0),
        "llm_backbone_ms": module_totals.get("llm_backbone_ms", 0.0),
        "llm_bridge_ms": module_totals.get("llm_bridge_ms", 0.0),
        "model_size_mb": model_size_mb(policy.model),
        **cuda_memory_stats(),
    }
    if args.verbose_steps or measured:
        print(
            f"{prefix} iter={iteration} raw "
            f"data={data_ms:.3f} preprocess={preprocess_ms:.3f} "
            f"vision={vision_ms:.3f} llm={llm_ms:.3f} action={action_ms:.3f} "
            f"e2e={end_to_end_latency_ms:.3f} ms",
            flush=True,
        )
    return sample, tflops_inputs


def summarize(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    metric_cols = [
        "data_ms",
        "preprocess_ms",
        "vision_ms",
        "llm_ms",
        "action_ms",
        "model_latency_ms",
        "end_to_end_latency_ms",
        "peak_cuda_memory_mb",
        "allocated_cuda_memory_mb",
        "reserved_cuda_memory_mb",
        "model_size_mb",
    ]
    for key in metric_cols:
        values = df[key].dropna().astype(float).to_numpy()
        if len(values) == 0:
            continue
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(statistics.pstdev(values)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def load_baseline_summary(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def add_baseline_comparisons(
    samples_df: pd.DataFrame, summary: dict[str, Any], baseline: dict[str, Any] | None
) -> None:
    if baseline is None:
        speedup = 1.0
        memory_reduction = 0.0
        model_size_reduction = 0.0
    else:
        base_summary = baseline.get("summary", {})
        base_latency = base_summary.get("model_latency_ms", {}).get("mean")
        base_memory = base_summary.get("peak_cuda_memory_mb", {}).get("mean")
        base_size = base_summary.get("model_size_mb", {}).get("mean")
        cur_latency = summary.get("model_latency_ms", {}).get("mean")
        cur_memory = summary.get("peak_cuda_memory_mb", {}).get("mean")
        cur_size = summary.get("model_size_mb", {}).get("mean")
        speedup = base_latency / cur_latency if base_latency and cur_latency else None
        memory_reduction = (
            (1.0 - cur_memory / base_memory) * 100.0 if base_memory and cur_memory else None
        )
        model_size_reduction = (1.0 - cur_size / base_size) * 100.0 if base_size and cur_size else None

    samples_df["speedup_vs_baseline"] = speedup
    samples_df["memory_reduction_vs_baseline"] = memory_reduction
    samples_df["model_size_reduction_vs_baseline"] = model_size_reduction
    summary["speedup_vs_baseline"] = speedup
    summary["memory_reduction_vs_baseline"] = memory_reduction
    summary["model_size_reduction_vs_baseline"] = model_size_reduction


def run(args: argparse.Namespace) -> dict[str, Any]:
    print("[Setup] loading data config", flush=True)
    data_config = load_data_config(args.data_config)
    print("[Setup] loading GR00T policy", flush=True)
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )
    apply_fastv_config(policy, args)

    print(
        "[Setup] mode="
        f"{'FastV-50%' if args.use_fastv else 'baseline'} "
        f"warmup={args.warmup_steps} repeat={args.repeat_steps}",
        flush=True,
    )

    for i in range(args.warmup_steps):
        run_one_iteration(policy, args, i, measured=False)

    samples = []
    tflops_inputs = None
    for i in range(args.repeat_steps):
        sample, maybe_tflops_inputs = run_one_iteration(policy, args, i, measured=True)
        samples.append(sample)
        if maybe_tflops_inputs is not None:
            tflops_inputs = maybe_tflops_inputs

    df = pd.DataFrame(samples)
    tflops = {}
    if tflops_inputs is not None:
        tflops_args = {
            key: tflops_inputs[key] for key in ("T", "K", "N", "M", "rho", "d", "ffn_dim")
        }
        tflops = estimate_transformer_tflops(**tflops_args)
        df["theoretical_tflops"] = tflops["tflops_prune" if args.use_fastv else "tflops_full"]
    else:
        df["theoretical_tflops"] = None

    df["effective_tflops_per_second"] = df["theoretical_tflops"] / (
        df["model_latency_ms"] / 1000.0
    )
    summary = summarize(df)
    baseline = load_baseline_summary(args.baseline_json)
    add_baseline_comparisons(df, summary, baseline)

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    result = {
        "metadata": runtime_metadata(args.variant),
        "fastv_config": {
            "use_fastv": args.use_fastv,
            "fastv_k": args.fastv_k,
            "fastv_r": args.fastv_r,
            "image_token_start_index": args.image_token_start_index,
            "image_token_length": args.image_token_length,
        },
        "tflops_inputs": tflops_inputs,
        "tflops": tflops,
        "summary": summary,
        "raw_measurements": samples,
        "output_csv": str(output_csv),
    }
    save_json(output_json, result)
    print(f"[Done] wrote raw CSV to {output_csv}", flush=True)
    print(f"[Done] wrote summary JSON to {output_json}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="")
    parser.add_argument("--model-path", default="youliangtan/gr00t-n1.5-libero-long-posttrain")
    parser.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--repeat-steps", type=int, default=100)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_fastv", type=str2bool, default=False)
    parser.add_argument("--fastv_k", type=int, default=2)
    parser.add_argument("--fastv_r", type=float, default=0.5)
    parser.add_argument("--image_token_start_index", type=int, default=None)
    parser.add_argument("--image_token_length", type=int, default=None)
    parser.add_argument("--baseline-json", default="")
    parser.add_argument("--output-csv", default="results/fastv_profile/raw_measurements.csv")
    parser.add_argument("--output-json", default="results/fastv_profile/summary.json")
    parser.add_argument("--verbose-steps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({"summary": result["summary"], "tflops": result["tflops"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
