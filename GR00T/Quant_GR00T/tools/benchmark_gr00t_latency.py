#!/usr/bin/env python3
"""Benchmark GR00T policy latency with coarse and fine-grained breakdowns."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gr00t.eval.service import ExternalRobotInferenceClient
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy, squeeze_dict_values, unsqueeze_dict_values
from tools.quantvla_bench_common import runtime_metadata, save_json


def duquant_env_enabled() -> bool:
    return any(
        key.startswith("GR00T_DUQUANT_") and key != "GR00T_DUQUANT_PACKDIR"
        for key in os.environ
    )


def print_runtime_mode(mode: str) -> None:
    duquant_enabled = duquant_env_enabled()
    atm_enabled = os.environ.get("GR00T_ATM_ENABLE", "0") not in ("0", "false", "False")
    ohb_enabled = os.environ.get("GR00T_OHB_ENABLE", "0") not in ("0", "false", "False")
    print("Runtime mode:")
    print(f"  benchmark mode: {mode}")
    print(f"  DuQuant env enabled: {duquant_enabled}")
    print(f"  ATM env enabled: {atm_enabled}")
    print(f"  OHB env enabled: {ohb_enabled}")
    print(f"  GR00T_ATTN_IMPLEMENTATION: {os.environ.get('GR00T_ATTN_IMPLEMENTATION', 'sdpa')}")
    if mode == "client":
        print("  note: client mode measures the already-running server; server-side quantization depends on that server process env.")


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def now_ms() -> float:
    return time.perf_counter() * 1000.0


@contextlib.contextmanager
def maybe_autocast(dtype: torch.dtype):
    if torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield


def sample_libero_observation(height: int, width: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(1, height, width, 3), dtype=np.uint8)
    wrist_image = rng.integers(0, 256, size=(1, height, width, 3), dtype=np.uint8)

    obs = {
        "video.image": image,
        "video.wrist_image": wrist_image,
        # Match examples/Libero/eval/run_libero_eval.py: np.array(...) defaults to float64.
        "state.x": np.array([[0.0]], dtype=np.float64),
        "state.y": np.array([[0.0]], dtype=np.float64),
        "state.z": np.array([[0.0]], dtype=np.float64),
        "state.roll": np.array([[0.0]], dtype=np.float64),
        "state.pitch": np.array([[0.0]], dtype=np.float64),
        "state.yaw": np.array([[0.0]], dtype=np.float64),
        "state.gripper": np.array([[0.0, 0.0]], dtype=np.float64),
        "annotation.human.action.task_description": ["put the black bowl on the plate"],
    }
    return obs


def timed_call(fn, *args, **kwargs):
    sync_cuda()
    start = now_ms()
    out = fn(*args, **kwargs)
    sync_cuda()
    return out, now_ms() - start


def prepare_policy_input(policy: Gr00tPolicy, observations: dict[str, Any]):
    obs_copy = observations.copy()
    is_batch = policy._check_state_is_batched(obs_copy)
    if not is_batch:
        obs_copy = unsqueeze_dict_values(obs_copy)

    for key, value in obs_copy.items():
        if not isinstance(value, np.ndarray):
            obs_copy[key] = np.array(value)
    return obs_copy, is_batch


def profile_action_head_fine(action_head, backbone_output, action_input):
    timings: dict[str, float] = {}

    backbone_output, timings["action_process_backbone_ms"] = timed_call(
        action_head.process_backbone_output, backbone_output
    )

    vl_embs = backbone_output.backbone_features
    embodiment_id = action_input.embodiment_id
    batch_size = vl_embs.shape[0]
    device = vl_embs.device

    state_features, timings["action_state_encoder_ms"] = timed_call(
        action_head.state_encoder, action_input.state, embodiment_id
    )

    def init_noise():
        return torch.randn(
            size=(batch_size, action_head.config.action_horizon, action_head.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

    actions, timings["action_noise_init_ms"] = timed_call(init_noise)

    num_steps = action_head.num_inference_timesteps
    dt = 1.0 / num_steps
    action_encode_concat_ms = 0.0
    dit_ms = 0.0
    decode_update_ms = 0.0

    future_tokens = action_head.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)

    for step in range(num_steps):
        t_cont = step / float(num_steps)
        t_discretized = int(t_cont * action_head.num_timestep_buckets)
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)

        def encode_and_concat():
            action_features = action_head.action_encoder(actions, timesteps_tensor, embodiment_id)
            if action_head.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            return torch.cat((state_features, future_tokens, action_features), dim=1)

        sa_embs, elapsed = timed_call(encode_and_concat)
        action_encode_concat_ms += elapsed

        model_output, elapsed = timed_call(
            action_head.model,
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )
        dit_ms += elapsed

        def decode_and_update():
            nonlocal actions
            pred = action_head.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -action_head.action_horizon :]
            actions = actions + dt * pred_velocity

        _, elapsed = timed_call(decode_and_update)
        decode_update_ms += elapsed

    timings["action_denoising_encode_concat_ms"] = action_encode_concat_ms
    timings["action_denoising_dit_ms"] = dit_ms
    timings["action_denoising_decode_update_ms"] = decode_update_ms
    timings["action_head_total_profiled_ms"] = sum(timings.values())
    timings["action_dit_per_step_ms"] = dit_ms / num_steps
    timings["action_num_denoising_steps"] = float(num_steps)
    return {"action_pred": actions}, timings


def profile_model(policy: Gr00tPolicy, normalized_input: dict[str, Any], fine_grained: bool):
    model = policy.model
    timings: dict[str, float] = {}
    sync_cuda()
    model_start = now_ms()

    backbone_inputs, action_inputs = model.backbone.prepare_input(normalized_input), model.action_head.prepare_input(
        normalized_input
    )

    def to_device_with_maybe_dtype(x):
        if torch.is_floating_point(x):
            return x.to(model.device, dtype=model.action_head.dtype)
        return x.to(model.device)

    def move_inputs():
        return (
            tree.map_structure(to_device_with_maybe_dtype, backbone_inputs),
            tree.map_structure(to_device_with_maybe_dtype, action_inputs),
        )

    (backbone_inputs, action_inputs), timings["model_input_to_device_ms"] = timed_call(move_inputs)

    backbone_outputs, timings["backbone_eagle_ms"] = timed_call(model.backbone, backbone_inputs)

    if fine_grained:
        action_outputs, action_timings = profile_action_head_fine(
            model.action_head, backbone_outputs, action_inputs
        )
        timings.update(action_timings)
    else:
        action_outputs, timings["action_head_total_ms"] = timed_call(
            model.action_head.get_action, backbone_outputs, action_inputs
        )

    normalized_action = action_outputs["action_pred"].float()
    sync_cuda()
    timings["model_total_profiled_ms"] = now_ms() - model_start
    return normalized_action, timings


def profile_policy_once(policy: Gr00tPolicy, obs: dict[str, Any], fine_grained: bool):
    timings: dict[str, float] = {}
    sync_cuda()
    total_start = now_ms()

    (obs_copy, is_batch), timings["input_pack_ms"] = timed_call(prepare_policy_input, policy, obs)
    normalized_input, timings["preprocess_transform_ms"] = timed_call(policy.apply_transforms, obs_copy)

    with torch.inference_mode(), maybe_autocast(torch.bfloat16):
        normalized_action, model_timings = profile_model(policy, normalized_input, fine_grained)
    timings.update(model_timings)

    def postprocess():
        unnormalized_action = policy._get_unnormalized_action(normalized_action)
        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    _, timings["postprocess_untransform_ms"] = timed_call(postprocess)
    sync_cuda()
    timings["policy_total_profiled_ms"] = now_ms() - total_start
    return timings


def summarize(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        for key, value in sample.items():
            grouped[key].append(float(value))

    summary = {}
    for key, values in grouped.items():
        values_np = np.array(values, dtype=np.float64)
        summary[key] = {
            "mean_ms": float(values_np.mean()),
            "median_ms": float(np.median(values_np)),
            "p90_ms": float(np.percentile(values_np, 90)),
            "p95_ms": float(np.percentile(values_np, 95)),
            "min_ms": float(values_np.min()),
            "max_ms": float(values_np.max()),
            "std_ms": float(statistics.pstdev(values)),
        }
    return summary


def run_local(args):
    print_runtime_mode("local")
    data_config = load_data_config(args.data_config)
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )
    try:
        from gr00t.quantization import DuQuantLinear

        n_duquant = sum(1 for module in policy.model.modules() if isinstance(module, DuQuantLinear))
        print(f"  actual DuQuantLinear modules: {n_duquant}")
    except Exception as exc:
        print(f"  actual DuQuantLinear modules: unavailable ({exc})")
    obs = sample_libero_observation(args.image_height, args.image_width, args.seed)

    print(f"Warmup iterations: {args.warmup}")
    for _ in range(args.warmup):
        profile_policy_once(policy, obs, args.fine_grained)

    print(f"Measured iterations: {args.iters}")
    samples = [profile_policy_once(policy, obs, args.fine_grained) for _ in range(args.iters)]
    return summarize(samples)


def run_client(args):
    if not args.synthetic_client:
        raise ValueError(
            "Synthetic client latency is disabled by default. "
            "For real LIBERO observations, run examples/Libero/eval/run_libero_eval.py "
            "via ./run_libero_eval.sh with --output-json and optionally --profile-server. "
            "Pass --synthetic-client only for a quick ZMQ smoke test."
        )
    print_runtime_mode("client")
    client = ExternalRobotInferenceClient(host=args.host, port=args.port)
    obs = sample_libero_observation(args.image_height, args.image_width, args.seed)

    for _ in range(args.warmup):
        _, _ = timed_call(client.get_action, obs)

    samples = []
    for _ in range(args.iters):
        _, elapsed = timed_call(client.get_action, obs)
        samples.append({"client_zmq_end_to_end_ms": elapsed})
    return summarize(samples)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["local", "client"], default="local")
    parser.add_argument("--variant", default="")
    parser.add_argument("--model-path", default="youliangtan/gr00t-n1.5-libero-long-posttrain")
    parser.add_argument(
        "--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig"
    )
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--fine-grained", action="store_true")
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument(
        "--synthetic-client",
        action="store_true",
        help="Use a generated LIBERO-like observation for a quick ZMQ smoke test.",
    )
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_client(args) if args.mode == "client" else run_local(args)
    result = {
        "metadata": runtime_metadata(args.variant),
        "summary": summary,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        save_json(args.output_json, result)


if __name__ == "__main__":
    main()
