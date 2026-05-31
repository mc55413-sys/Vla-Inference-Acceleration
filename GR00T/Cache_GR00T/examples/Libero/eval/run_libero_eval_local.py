#!/usr/bin/env python3
"""
LIBERO evaluation with local GR00T model + GR00T-Cache.

Runs the LIBERO benchmark locally (no server), measures 5-stage latency,
and supports GR00T-Cache for cross-timestep visual token KV reuse.

Usage:
  # Baseline (cache disabled)
  python examples/Libero/eval/run_libero_eval_local.py \
      --model-path /path/to/gr00t-n1.5-libero \
      --task-suite-name libero_spatial

  # With GR00T-Cache
  python examples/Libero/eval/run_libero_eval_local.py \
      --model-path /path/to/gr00t-n1.5-libero \
      --task-suite-name libero_spatial \
      --cache-mode full_cache --max-reuse-ratio 0.5 --task-topk 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import tqdm

# Add repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.policy import Gr00tPolicy, squeeze_dict_values, unsqueeze_dict_values


# ── Logging ───────────────────────────────────────────────────────────

log_dir = "/tmp/logs"
os.makedirs(log_dir, exist_ok=True)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize_latency_values(latency_values):
    summary = {}
    for key, values in latency_values.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "count": int(arr.size),
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p95_ms": float(np.percentile(arr, 95)),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "std_ms": float(arr.std()),
        }
    return summary


# ── Local GR00T Policy ────────────────────────────────────────────────

class LocalGr00tPolicy:
    """Local GR00T policy — loads model directly, no server needed.

    Supports GR00T-Cache via the --cache-* CLI args.
    """

    LIBERO_CONFIG = {
        "proprio_size": 8,
        "state_key_mapping": {
            "x": 0, "y": 1, "z": 2,
            "roll": 3, "pitch": 4, "yaw": 5,
            "gripper": (6, 8),
        },
    }

    def __init__(
        self,
        model_path: str,
        data_config_path: str,
        embodiment_tag: str,
        denoising_steps: int,
        device: str = "cuda",
        cache_config: Optional["GR00TCacheConfig"] = None,
    ):
        from gr00t.experiment.data_config import load_data_config
        from gr00t.data.embodiment_tags import EmbodimentTag

        self.device = device

        data_config = load_data_config(data_config_path)
        self.policy = Gr00tPolicy(
            model_path=model_path,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag=embodiment_tag,
            denoising_steps=denoising_steps,
            device=device,
        )
        self.model = self.policy.model
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

        # Cache
        self.cache_config = cache_config
        self.cache_manager = None
        self._bb_wrappers = {}
        self._ah_wrappers = {}
        if cache_config and cache_config.enabled:
            self._setup_cache()

    def _setup_cache(self):
        from gr00t_cache.cache_manager import GR00TCacheManager
        from gr00t_cache.token_index_map import TokenIndexMap
        from gr00t_cache.attention_wrapper import (
            apply_cache_to_backbone, apply_cache_to_action_head,
        )

        self.cache_manager = GR00TCacheManager(self.cache_config)
        token_map = TokenIndexMap(n_visual=768, n_total=848)
        self._bb_wrappers = apply_cache_to_backbone(
            self.model, self.cache_manager, token_map, self.cache_config,
        )
        self._ah_wrappers = apply_cache_to_action_head(
            self.model, self.cache_manager, self.cache_config,
        )
        if self.cache_config.debug:
            print(f"[GR00T-Cache] Wrapped {len(self._bb_wrappers)} backbone + "
                  f"{len(self._ah_wrappers)} action-head layers")

    def reset_cache(self):
        if self.cache_manager:
            self.cache_manager.reset()

    def get_action_with_timing(self, observation_dict, lang: str):
        """Get action + 5-stage timing using the real LIBERO observation."""
        from examples.Libero.eval.utils import (
            get_libero_image, quat2axisangle, normalize_gripper_action,
        )

        timing = {}
        t_total_start = time.perf_counter()

        # ── Stage 1: Data (LIBERO obs → GR00T format) ──
        t0 = time.perf_counter()
        xyz = observation_dict["robot0_eef_pos"]
        rpy = quat2axisangle(observation_dict["robot0_eef_quat"])
        gripper = observation_dict["robot0_gripper_qpos"]
        img, wrist_img = get_libero_image(observation_dict)
        obs = {
            "video.image": np.expand_dims(img, axis=0),
            "video.wrist_image": np.expand_dims(wrist_img, axis=0),
            "state.x": np.array([[xyz[0]]]),
            "state.y": np.array([[xyz[1]]]),
            "state.z": np.array([[xyz[2]]]),
            "state.roll": np.array([[rpy[0]]]),
            "state.pitch": np.array([[rpy[1]]]),
            "state.yaw": np.array([[rpy[2]]]),
            "state.gripper": np.expand_dims(gripper, axis=0),
            "annotation.human.action.task_description": [lang],
        }
        timing["data_ms"] = (time.perf_counter() - t0) * 1000.0

        # ── Cache: update current images for reuse plan ──
        if self.cache_manager and self.cache_manager.config.enabled:
            current_images = torch.from_numpy(
                np.stack([img, wrist_img])
            ).float().permute(0, 3, 1, 2) / 255.0  # [2, 3, H, W]

            from gr00t_cache.token_index_map import TokenIndexMap
            token_map = TokenIndexMap(n_visual=768, n_total=848)

            reuse_plan = self.cache_manager.get_reuse_plan(
                current_images=current_images,
                current_proprio=torch.from_numpy(
                    np.array([xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2],
                              gripper[0], gripper[1]])
                ).float(),
                current_token_map=token_map,
                batch_size=1,
            )
            self.cache_manager._current_reuse_plan = reuse_plan

        # ── Stage 2: Preprocess ──
        cuda_sync()
        t0 = time.perf_counter()
        obs_copy = obs.copy()
        is_batch = self.policy._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)
        normalized_input, transform_timings = self.policy._apply_transforms_profiled(
            obs_copy, cuda_sync
        )
        cuda_sync()
        timing["preprocess_ms"] = (time.perf_counter() - t0) * 1000.0

        # ── Stages 3+4: Vision + LLM (backbone) ──
        with torch.inference_mode():
            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, model_timings = self.policy._get_action_from_normalized_input_profiled(
                        normalized_input
                    )
            else:
                _, model_timings = self.policy._get_action_from_normalized_input_profiled(
                    normalized_input
                )

        # Extract stage timings
        timing["vision_ms"] = model_timings.get("server_system2_vision_ms", 0.0)
        timing["llm_ms"] = model_timings.get("server_system2_reasoning_ms", 0.0)
        timing["action_ms"] = (
            model_timings.get("server_system1_action_head_ms", 0.0)
            + model_timings.get("server_postprocess_untransform_ms", 0.0)
        )

        # ── Stage 5: Postprocess (action conversion) ──
        t0 = time.perf_counter()
        action_chunk = self.policy._get_unnormalized_action(
            model_timings.get("_action_pred", None) if "_action_pred" in model_timings
            else self._extract_action(normalized_input)
        )
        timing["postprocess_ms"] = (time.perf_counter() - t0) * 1000.0

        # ── Action conversion to LIBERO format ──
        t0 = time.perf_counter()
        action_components = [
            np.atleast_1d(action_chunk[f"action.{key}"][0])[0]
            for key in self.action_keys
        ]
        action_array = np.array(action_components, dtype=np.float32)
        from examples.Libero.eval.utils import normalize_gripper_action
        action_array = normalize_gripper_action(action_array, binarize=True)
        timing["convert_to_libero_ms"] = (time.perf_counter() - t0) * 1000.0

        # ── Derived ──
        timing["model_ms"] = timing["vision_ms"] + timing["llm_ms"] + timing["action_ms"]
        timing["e2e_ms"] = (
            timing["data_ms"] + timing["preprocess_ms"]
            + timing["model_ms"] + timing["postprocess_ms"]
        )
        timing["total_ms"] = (time.perf_counter() - t_total_start) * 1000.0

        # Store raw internal timings for debugging
        timing["_vision_model_ms"] = model_timings.get("server_system2_vision_model_ms", 0.0)
        timing["_vision_projector_ms"] = model_timings.get("server_system2_vision_projector_ms", 0.0)
        timing["_backbone_total_ms"] = model_timings.get("server_system2_backbone_ms", 0.0)
        timing["_action_head_ms"] = model_timings.get("server_system1_action_head_ms", 0.0)
        timing["_model_total_ms"] = model_timings.get("server_model_total_ms", 0.0)

        # ── Update cache ──
        if self.cache_manager and self.cache_manager.config.enabled:
            for w in self._bb_wrappers.values():
                w.store_backbone_kv(w.layer_idx)
            for w in self._ah_wrappers.values():
                w.store_condition_kv(w.layer_idx)

        return action_array, timing

    def _extract_action(self, normalized_input):
        """Extract action prediction when not available from profiled path."""
        with torch.inference_mode():
            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model_pred = self.model.get_action(normalized_input)
            else:
                model_pred = self.model.get_action(normalized_input)
        return model_pred["action_pred"].float()

    @property
    def cache_stats(self) -> dict:
        if self.cache_manager:
            return self.cache_manager.stats()
        return {}


# ── Eval ───────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    model_path: str = ""
    data_config: str = "examples.Libero.custom_data_config:LiberoDataConfig"
    embodiment_tag: str = "new_embodiment"
    denoising_steps: int = 8
    device: str = "cuda"

    # LIBERO
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    headless: bool = False
    task_ids: Optional[list[int]] = None
    save_videos: bool = False

    # Cache
    cache_mode: str = "none"  # none, full_cache, backbone_visual_kv_cache, action_head_condition_kv_cache
    max_reuse_ratio: float = 0.5
    task_topk: Optional[int] = None
    entropy_scale: float = 1.0

    # Output
    output_json: Optional[str] = None
    print_step_latency: bool = True
    warmup_steps: int = 10


def eval_libero_local(cfg: EvalConfig):
    from libero.libero import benchmark
    from examples.Libero.eval.utils import (
        get_libero_dummy_action, get_libero_env, get_libero_image,
        normalize_gripper_action, quat2axisangle, save_rollout_video,
    )

    # ── Load model ──
    print(f"Loading GR00T model: {cfg.model_path}")
    from gr00t_cache.config import GR00TCacheConfig, CacheMode
    cache_config = None
    if cfg.cache_mode != "none":
        cache_config = GR00TCacheConfig(
            enabled=True,
            cache_mode=CacheMode(cfg.cache_mode),
            max_reuse_ratio=cfg.max_reuse_ratio,
            task_topk=cfg.task_topk,
            entropy_scale=cfg.entropy_scale,
            debug=False,
        )
        print(f"Cache mode: {cfg.cache_mode}, max_reuse={cfg.max_reuse_ratio}, task_topk={cfg.task_topk}")
    else:
        print("Cache disabled (baseline)")

    gr00t_policy = LocalGr00tPolicy(
        model_path=cfg.model_path,
        data_config_path=cfg.data_config,
        embodiment_tag=cfg.embodiment_tag,
        denoising_steps=cfg.denoising_steps,
        device=cfg.device,
        cache_config=cache_config,
    )

    # ── LIBERO setup ──
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    task_indices = cfg.task_ids or list(range(num_tasks))

    log_path = f"{log_dir}/libero_eval_local_{cfg.task_suite_name}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")

    total_episodes, total_successes = 0, 0
    started_at = time.time()
    latency_values = defaultdict(list)

    results = {
        "task_suite_name": cfg.task_suite_name,
        "cache_mode": cfg.cache_mode,
        "max_reuse_ratio": cfg.max_reuse_ratio,
        "task_indices": task_indices,
        "tasks": [],
        "episodes": [],
    }

    # Warmup
    print(f"\nWarming up ({cfg.warmup_steps} dummy steps)...")
    dummy_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    for _ in range(cfg.warmup_steps):
        dummy_obs = {
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0]),
            "robot0_gripper_qpos": np.array([0.0, 0.0]),
            "agentview_image": dummy_img,
            "robot0_eye_in_hand_image": dummy_img,
        }
        gr00t_policy.get_action_with_timing(dummy_obs, "warmup")

    print("Warmup complete.\n")

    try:
        for task_id in tqdm.tqdm(task_indices):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, resolution=256)

            # Reset cache per task
            gr00t_policy.reset_cache()

            task_episodes, task_successes = 0, 0
            max_trials = min(cfg.num_trials_per_task, len(initial_states))

            for episode_idx in tqdm.tqdm(range(max_trials)):
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                t, done = 0, False

                if cfg.task_suite_name == "libero_spatial":
                    max_steps = 220
                elif cfg.task_suite_name == "libero_object":
                    max_steps = 280
                elif cfg.task_suite_name == "libero_goal":
                    max_steps = 600
                elif cfg.task_suite_name == "libero_10":
                    max_steps = 1000
                elif cfg.task_suite_name == "libero_90":
                    max_steps = 400
                else:
                    max_steps = 300

                episode_latency = defaultdict(list)

                while t < max_steps + cfg.num_steps_wait:
                    try:
                        if t < cfg.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action())
                            t += 1
                            continue

                        # Get action with timing
                        action, step_timing = gr00t_policy.get_action_with_timing(
                            obs, task.language,
                        )

                        # Record timings
                        for k, v in step_timing.items():
                            latency_values[k].append(float(v))
                            episode_latency[k].append(float(v))

                        # Step environment
                        obs, reward, done, info = env.step(action.tolist())

                        if cfg.print_step_latency and t == cfg.num_steps_wait + 1:
                            # Print first real step timing
                            print(f"\n  First step timing (task={task_id}, ep={episode_idx}):")
                            for stage in ["data_ms", "preprocess_ms", "vision_ms",
                                          "llm_ms", "action_ms", "postprocess_ms",
                                          "model_ms", "e2e_ms"]:
                                v = step_timing.get(stage, 0)
                                print(f"    {stage:20s}: {v:8.2f} ms")

                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        print(f"Exception: {e}")
                        import traceback
                        traceback.print_exc()
                        break

                task_episodes += 1
                total_episodes += 1

                ep_summary = summarize_latency_values(episode_latency)
                results["episodes"].append({
                    "task_id": int(task_id),
                    "episode_idx": int(episode_idx),
                    "success": bool(done),
                    "steps": int(t),
                    "latency_summary": ep_summary,
                })

                print(f"  Episode {episode_idx}: success={done}, steps={t}")

            task_rate = task_successes / task_episodes if task_episodes else 0
            results["tasks"].append({
                "task_id": int(task_id),
                "episodes": int(task_episodes),
                "successes": int(task_successes),
                "success_rate": task_rate,
            })
            print(f"  Task {task_id}: success_rate={task_rate:.2f}")

    finally:
        results["total_episodes"] = int(total_episodes)
        results["total_successes"] = int(total_successes)
        results["success_rate"] = (
            float(total_successes) / float(total_episodes) if total_episodes else 0.0
        )
        results["elapsed_sec"] = time.time() - started_at
        results["latency_summary"] = summarize_latency_values(latency_values)
        results["cache_stats"] = gr00t_policy.cache_stats

        if cfg.output_json:
            os.makedirs(os.path.dirname(cfg.output_json), exist_ok=True)
            with open(cfg.output_json, "w") as f:
                json.dump(results, f, indent=2, default=str)

        log_file.close()

    # ── Print Summary ──
    summary = results.get("latency_summary", {})
    print("\n" + "=" * 80)
    print(f"  LATENCY SUMMARY — {cfg.task_suite_name} (cache={cfg.cache_mode})")
    print("=" * 80)
    stages = [
        ("data_ms", "1. Data"),
        ("preprocess_ms", "2. Preprocess"),
        ("vision_ms", "3. Vision"),
        ("llm_ms", "4. LLM"),
        ("action_ms", "5. Action (denoising + unnormalize)"),
        ("postprocess_ms", "   Postprocess (→LIBERO action)"),
        ("model_ms", "   Model (3+4+5)"),
        ("e2e_ms", "   End-to-End (1-5)"),
    ]
    for key, name in stages:
        s = summary.get(key, {})
        if s:
            print(f"  {name:42s}  mean={s['mean_ms']:8.2f}  p50={s['median_ms']:8.2f}  "
                  f"p95={s['p95_ms']:8.2f}  std={s['std_ms']:6.2f} ms")
    hz = 1000.0 / summary["e2e_ms"]["mean_ms"] if summary.get("e2e_ms", {}).get("mean_ms", 0) > 0 else 0
    print(f"  {'Control Frequency':42s}  {hz:.1f} Hz")
    print(f"  Success rate: {results['success_rate']:.2f} "
          f"({results['total_successes']}/{results['total_episodes']})")
    if results.get("cache_stats"):
        print(f"  Cache: {json.dumps(results['cache_stats'], indent=2)}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIBERO eval with local GR00T + Cache")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-config", type=str,
                        default="examples.Libero.custom_data_config:LiberoDataConfig")
    parser.add_argument("--embodiment-tag", type=str, default="new_embodiment")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task-suite-name", type=str, default="libero_spatial")
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--cache-mode", type=str, default="none")
    parser.add_argument("--max-reuse-ratio", type=float, default=0.5)
    parser.add_argument("--task-topk", type=int, default=None)
    parser.add_argument("--entropy-scale", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    cfg = EvalConfig(
        model_path=args.model_path,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=args.device,
        task_suite_name=args.task_suite_name,
        num_trials_per_task=args.num_trials_per_task,
        headless=args.headless,
        cache_mode=args.cache_mode,
        max_reuse_ratio=args.max_reuse_ratio,
        task_topk=args.task_topk,
        entropy_scale=args.entropy_scale,
        warmup_steps=args.warmup_steps,
        output_json=args.output_json,
    )
    eval_libero_local(cfg)
