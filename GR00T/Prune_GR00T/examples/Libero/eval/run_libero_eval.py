import json
import os
import pprint
import time
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import tqdm
import tyro
from libero.libero import benchmark

from examples.Libero.eval.utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    normalize_gripper_action,
    quat2axisangle,
    save_rollout_video,
)

log_dir = "/tmp/logs"
os.makedirs(log_dir, exist_ok=True)  # ensures directory exists


def summarize_obs(obs_dict):
    summary = {}
    for k, v in obs_dict.items():
        if isinstance(v, torch.Tensor):
            summary[k] = {"shape": tuple(v.shape), "dtype": v.dtype, "device": v.device}
        elif isinstance(v, np.ndarray):
            summary[k] = {"shape": v.shape, "dtype": v.dtype}
        else:
            summary[k] = type(v).__name__
    pprint.pprint(summary)


def show_obs_images_cv2(new_obs):
    # remove batch dim
    img_agent = new_obs["video.image"][0]
    img_wrist = new_obs["video.wrist_image"][0]

    # convert RGB -> BGR for OpenCV
    img_agent_bgr = cv2.cvtColor(img_agent, cv2.COLOR_RGB2BGR)
    img_wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)

    # show in separate windows
    cv2.imshow("Agent View", img_agent_bgr)
    cv2.imshow("Wrist View", img_wrist_bgr)
    cv2.waitKey(1)


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


def summarize_memory_values(memory_values):
    summary = {}
    for key, values in memory_values.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "count": int(arr.size),
            "mean_bytes": float(arr.mean()),
            "median_bytes": float(np.median(arr)),
            "min_bytes": float(arr.min()),
            "max_bytes": float(arr.max()),
            "mean_gib": float(arr.mean() / (1024**3)),
            "median_gib": float(np.median(arr) / (1024**3)),
            "max_gib": float(arr.max() / (1024**3)),
        }
    return summary


def bytes_to_gib(value):
    if value is None:
        return None
    return float(value) / (1024**3)


def get_mean_latency(latency_summary, key):
    stats = latency_summary.get(key)
    if not stats:
        return None
    return stats["mean_ms"]


def write_results_json(path, results, latency_values, memory_values):
    if not path:
        return
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    results["latency_summary"] = summarize_latency_values(latency_values)
    results["memory_summary"] = summarize_memory_values(memory_values)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def format_latency_stats(latency_summary, label, key):
    stats = latency_summary.get(key)
    if not stats:
        return None
    return f"{label} avg={stats['mean_ms']:.1f}ms p50={stats['median_ms']:.1f}ms p95={stats['p95_ms']:.1f}ms"


STAGE_LATENCY_KEYS = [
    ("Data", "dual_data_ms"),
    ("Preprocess", "dual_preprocess_ms"),
    ("System-2 Vision", "dual_s2_vision_ms"),
    ("System-2 Reasoning", "dual_s2_reasoning_ms"),
    ("Bridge", "dual_bridge_ms"),
    ("System-1 Vision", "dual_s1_vision_ms"),
    ("System-1 Action", "dual_s1_action_ms"),
    ("Post/Other", "dual_post_other_ms"),
]


def format_end_to_end_latency_line(latency_summary):
    keys = [
        ("policy", "client_policy_total_ms"),
        ("server", "server_policy_total_ms"),
        ("env", "libero_env_step_ms"),
        ("loop", "control_loop_step_ms"),
    ]
    parts = []
    for label, key in keys:
        part = format_latency_stats(latency_summary, label, key)
        if part:
            parts.append(part)
    return " | ".join(parts)


def format_dual_system_latency_line(latency_summary):
    parts = []
    for label, key in (
        *STAGE_LATENCY_KEYS,
        ("End to End Latency", "dual_stage_sum_ms"),
        ("Model Latency", "dual_model_ms"),
    ):
        part = format_latency_stats(latency_summary, label, key)
        if part:
            parts.append(part)
    return " | ".join(parts)


def format_model_latency_line(latency_summary):
    keys = [
        ("gr00t", "server_gr00t_latency_ms"),
        ("model", "server_model_total_ms"),
        ("data", "server_dual_data_ms"),
        ("s2_vision", "dual_s2_vision_ms"),
        ("s2_reason", "dual_s2_reasoning_ms"),
        ("bridge", "dual_bridge_ms"),
        ("s1_vision", "dual_s1_vision_ms"),
        ("s1_action", "dual_s1_action_ms"),
        ("post", "dual_post_ms"),
        ("model_other", "server_model_unaccounted_ms"),
    ]
    parts = []
    for label, key in keys:
        part = format_latency_stats(latency_summary, label, key)
        if part:
            parts.append(part)
    return " | ".join(parts)


def format_preprocess_latency_line(latency_summary):
    keys = [
        ("pre", "server_preprocess_transform_ms"),
        ("pre_video", "server_preprocess_video_ms"),
        ("pre_vlm", "server_preprocess_gr00t_vlm_ms"),
        ("pre_state", "server_preprocess_state_action_ms"),
        ("pre_concat", "server_preprocess_concat_ms"),
        ("post", "server_postprocess_untransform_ms"),
        ("srv_other", "server_unaccounted_ms"),
    ]
    parts = []
    for label, key in keys:
        part = format_latency_stats(latency_summary, label, key)
        if part:
            parts.append(part)
    return " | ".join(parts)


def format_memory_line(memory_summary):
    keys = [
        ("llm", "server_model_component_llm_bytes", "median_gib"),
        ("dit", "server_model_component_dit_bytes", "median_gib"),
        ("llm+dit", "server_model_component_llm_plus_dit_bytes", "median_gib"),
        ("total", "server_model_component_total_bytes", "median_gib"),
        ("vision", "server_model_component_vision_bytes", "median_gib"),
    ]
    parts = []
    for label, key, stat_key in keys:
        stats = memory_summary.get(key)
        if stats and stats.get(stat_key) is not None:
            parts.append(f"{label}={stats[stat_key]:.3f}GiB")
    return " | ".join(parts)


def format_step_latency(step_timing):
    keys = [
        ("policy", "client_policy_total_ms"),
        ("server", "server_policy_total_ms"),
        ("gr00t", "server_gr00t_latency_ms"),
        ("data", "server_dual_data_ms"),
        ("preprocess", "dual_preprocess_ms"),
        ("pre", "server_preprocess_transform_ms"),
        ("pre_video", "server_preprocess_video_ms"),
        ("pre_vlm", "server_preprocess_gr00t_vlm_ms"),
        ("to_dev", "server_model_prepare_input_to_device_ms"),
        ("s2_vision", "dual_s2_vision_ms"),
        ("s2_reason", "dual_s2_reasoning_ms"),
        ("bridge", "dual_bridge_ms"),
        ("s1_vision", "dual_s1_vision_ms"),
        ("s1_action", "dual_s1_action_ms"),
        ("model_latency", "dual_model_ms"),
        ("model_compute", "dual_model_compute_ms"),
        ("post", "server_postprocess_untransform_ms"),
        ("post_other", "dual_post_other_ms"),
        ("stage_sum", "dual_stage_sum_ms"),
        ("srv_other", "server_unaccounted_ms"),
        ("model_other", "server_model_unaccounted_ms"),
        ("env", "libero_env_step_ms"),
        ("loop", "control_loop_step_ms"),
    ]
    parts = []
    for label, key in keys:
        if key in step_timing:
            parts.append(f"{label}={float(step_timing[key]):.1f}ms")
    return " | ".join(parts)


def get_latest_and_avg(latency_values, key, current=None):
    values = latency_values.get(key, [])
    latest = float(current) if current is not None else (float(values[-1]) if values else None)
    avg = float(np.mean(values)) if values else None
    return latest, avg


def fmt_ms_pair(label, latest, avg):
    if latest is None or avg is None:
        return None
    return f"{label}={latest:.1f}ms avg={avg:.1f}ms"


def fmt_ms_current(label, latest):
    if latest is None:
        return None
    return f"{label}={latest:.1f}ms"


def has_server_profile_timing(step_timing):
    return any(
        key in step_timing
        for key in (
            "server_policy_total_ms",
            "server_model_total_ms",
            "server_system2_backbone_ms",
            "server_system1_action_head_ms",
        )
    )


def compute_dual_system_step_timings(step_timing):
    if not has_server_profile_timing(step_timing):
        step_timing["dual_stage_sum_ms"] = step_timing.get("client_policy_total_ms", 0.0)
        step_timing["dual_model_ms"] = step_timing.get("client_zmq_get_action_ms", 0.0)
        return

    step_timing["dual_preprocess_ms"] = step_timing.get(
        "server_preprocess_transform_ms", 0.0
    )
    server_data = (
        step_timing.get("server_input_pack_ms", 0.0)
        + step_timing.get("server_model_prepare_input_to_device_ms", 0.0)
    )
    step_timing["server_dual_data_ms"] = server_data
    step_timing["dual_data_ms"] = (
        step_timing.get("client_libero_obs_to_gr00t_obs_ms", 0.0) + server_data
    )
    step_timing["dual_s2_vision_ms"] = step_timing.get("server_system2_vision_ms", 0.0)
    step_timing["dual_s2_reasoning_ms"] = step_timing.get("server_system2_reasoning_ms", 0.0)
    step_timing["dual_bridge_ms"] = step_timing.get(
        "server_system2_to_system1_bridge_ms", 0.0
    )
    step_timing["dual_s1_vision_ms"] = step_timing.get("server_system1_vision_ms", 0.0)
    step_timing["dual_s1_action_ms"] = step_timing.get("server_system1_action_ms", 0.0)
    step_timing["dual_post_ms"] = (
        step_timing.get("server_postprocess_untransform_ms", 0.0)
        + step_timing.get("client_action_chunk_to_libero_action_ms", 0.0)
    )
    model_compute_ms = (
        step_timing.get("dual_s2_vision_ms", 0.0)
        + step_timing.get("dual_s2_reasoning_ms", 0.0)
        + step_timing.get("dual_bridge_ms", 0.0)
        + step_timing.get("dual_s1_vision_ms", 0.0)
        + step_timing.get("dual_s1_action_ms", 0.0)
    )
    if "client_policy_total_ms" in step_timing:
        step_timing["dual_post_other_ms"] = max(
            0.0,
            step_timing["client_policy_total_ms"]
            - step_timing.get("dual_data_ms", 0.0)
            - step_timing.get("dual_preprocess_ms", 0.0)
            - model_compute_ms,
        )
    else:
        step_timing["dual_post_other_ms"] = step_timing.get("dual_post_ms", 0.0)
    step_timing["dual_stage_sum_ms"] = sum(
        step_timing.get(key, 0.0) for _label, key in STAGE_LATENCY_KEYS
    )
    step_timing["dual_model_ms"] = max(
        0.0,
        step_timing.get("dual_stage_sum_ms", 0.0)
        - step_timing.get("dual_data_ms", 0.0)
        - step_timing.get("dual_preprocess_ms", 0.0),
    )
    step_timing["dual_model_compute_ms"] = model_compute_ms
    if "server_policy_total_ms" in step_timing:
        step_timing["server_gr00t_latency_ms"] = step_timing["server_policy_total_ms"]


def format_realtime_dual_system_latency_line(
    task_id,
    episode_idx,
    step,
    step_timing,
    latency_values,
):
    if not has_server_profile_timing(step_timing):
        client, client_avg = get_latest_and_avg(
            latency_values, "client_policy_total_ms", step_timing.get("client_policy_total_ms")
        )
        zmq, zmq_avg = get_latest_and_avg(
            latency_values, "client_zmq_get_action_ms", step_timing.get("client_zmq_get_action_ms")
        )
        parts = [
            fmt_ms_pair("Client Policy", client, client_avg),
            fmt_ms_pair("Client ZMQ", zmq, zmq_avg),
            "Server stages=n/a (run eval with --profile-server)",
        ]
        return (
            f"[latency] task={task_id} episode={episode_idx} step={step} "
            + " | ".join(part for part in parts if part)
        )

    parts = []
    for label, key in STAGE_LATENCY_KEYS:
        latest, _ = get_latest_and_avg(latency_values, key, step_timing.get(key))
        parts.append(fmt_ms_current(label, latest))
    stage_sum, stage_sum_avg = get_latest_and_avg(
        latency_values, "dual_stage_sum_ms", step_timing.get("dual_stage_sum_ms")
    )
    model, model_avg = get_latest_and_avg(
        latency_values, "dual_model_ms", step_timing.get("dual_model_ms")
    )
    parts.append(fmt_ms_pair("End to End Latency", stage_sum, stage_sum_avg))
    parts.append(fmt_ms_pair("Model Latency", model, model_avg))
    return (
        f"[latency] task={task_id} episode={episode_idx} step={step} "
        + " | ".join(part for part in parts if part)
    )


def format_realtime_tflops_line(
    task_id,
    episode_idx,
    step,
    step_timing,
    latency_values,
    dense_equiv_tflops_per_get_action,
):
    if dense_equiv_tflops_per_get_action is None:
        return None
    parts = [
        f"dense_equiv={float(dense_equiv_tflops_per_get_action):.4f}TFLOPs/call",
    ]
    return (
        f"[tflops] task={task_id} episode={episode_idx} step={step} "
        + " | ".join(part for part in parts if part)
    )


def format_gib_from_step_memory(step_memory, key):
    value = step_memory.get(key)
    if value is None:
        return "n/a"
    return f"{bytes_to_gib(value):.3f}GiB"


def format_realtime_memory_line(task_id, episode_idx, step, step_memory):
    parts = [
        f"llm={format_gib_from_step_memory(step_memory, 'server_model_component_llm_bytes')}",
        f"dit={format_gib_from_step_memory(step_memory, 'server_model_component_dit_bytes')}",
        (
            "llm+dit="
            + format_gib_from_step_memory(
                step_memory, "server_model_component_llm_plus_dit_bytes"
            )
        ),
        f"total={format_gib_from_step_memory(step_memory, 'server_model_component_total_bytes')}",
    ]
    return (
        f"[model component memory] task={task_id} episode={episode_idx} step={step} "
        + " | ".join(parts)
    )


def format_preprocess_detail(step_timing):
    detail_items = []
    for key, value in step_timing.items():
        if key.startswith("server_preprocess_") and key.endswith("_ms"):
            if key in {
                "server_preprocess_transform_ms",
                "server_preprocess_video_ms",
                "server_preprocess_state_action_ms",
                "server_preprocess_concat_ms",
                "server_preprocess_gr00t_vlm_ms",
                "server_preprocess_other_ms",
            }:
                continue
            label = key.removeprefix("server_preprocess_").removesuffix("_ms")
            detail_items.append((float(value), label))
    detail_items.sort(reverse=True)
    return " | ".join(f"{label}={value:.1f}ms" for value, label in detail_items)


def print_and_log(message, log_file):
    print(message)
    log_file.write(message + "\n")
    log_file.flush()


@dataclass
class GenerateConfig:
    # fmt: off
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1                     # Number of rollouts per task
    #################################################################################################################
    # fmt: on
    """Port to connect to."""
    port: int = 5555
    """Headless mode (no GUI)."""
    headless: bool = False
    """Run only the specified task indices (overrides order if provided)."""
    task_ids: list[int] | None = None
    """Run tasks in this explicit order."""
    task_order: list[int] | None = None
    """Optional JSON path for structured evaluation results."""
    output_json: str | None = None
    """Optional log file path. Defaults to /tmp/logs/libero_eval_<task>.log."""
    log_file: str | None = None
    """Save rollout videos."""
    save_videos: bool = True
    """Ask the GR00T server to return System 2/System 1 timing for each real LIBERO obs."""
    profile_server: bool = False
    """Maximum number of per-step timing records to keep in output_json."""
    max_step_timing_records: int = 20000
    """Write output_json after every episode, so partial results survive long runs."""
    live_json: bool = True
    """Print latency summary after every episode."""
    print_latency: bool = True
    """Print latency for every real LIBERO control step."""
    print_step_latency: bool = False
    """Print verbose per-step timing internals for debugging."""
    print_detailed_step_latency: bool = False
    """Dense-equivalent TFLOPs per get_action; used for live TFLOPs/call logs."""
    dense_equiv_tflops_per_get_action: float | None = None


class GR00TPolicy:
    """GR00T Policy wrapper for Libero environments."""

    LIBERO_CONFIG = {
        "proprio_size": 8,
        "state_key_mapping": {
            "x": 0,
            "y": 1,
            "z": 2,
            "roll": 3,
            "pitch": 4,
            "yaw": 5,
            "gripper": (6, 8),
        },
    }

    def __init__(self, host="localhost", port=5555, headless=False, profile_server=False):
        from gr00t.eval.service import ExternalRobotInferenceClient

        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.config = self.LIBERO_CONFIG
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        self.headless = headless
        self.profile_server = profile_server

    def get_action(self, observation_dict, lang: str):
        """Get action from GR00T policy given observation and language instruction."""
        action, _, _ = self.get_action_with_timing(observation_dict, lang)
        return action

    def get_action_with_timing(self, observation_dict, lang: str):
        """Get action and timing using the real LIBERO observation."""
        timing = {}
        memory = {}

        start = time.perf_counter()
        obs_dict = self._process_observation(observation_dict, lang)
        timing["client_libero_obs_to_gr00t_obs_ms"] = (time.perf_counter() - start) * 1000.0

        # summarize_obs(obs_dict)
        start = time.perf_counter()
        if self.profile_server:
            response = self.policy.get_action_profiled(obs_dict)
            action_chunk = response["__action__"]
            timing.update(response.get("__timing__", {}))
            memory.update(response.get("__memory__", {}))
        else:
            action_chunk = self.policy.get_action(obs_dict)
        timing["client_zmq_get_action_ms"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        action = self._convert_to_libero_action(action_chunk, 0)
        timing["client_action_chunk_to_libero_action_ms"] = (time.perf_counter() - start) * 1000.0
        timing["client_policy_total_ms"] = (
            timing["client_libero_obs_to_gr00t_obs_ms"]
            + timing["client_zmq_get_action_ms"]
            + timing["client_action_chunk_to_libero_action_ms"]
        )
        return action, timing, memory

    def _process_observation(self, obs, lang: str):
        """Convert Libero observation to GR00T format."""
        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        img, wrist_img = get_libero_image(obs)
        new_obs = {
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
        if not self.headless:
            show_obs_images_cv2(new_obs)
        return new_obs

    def _convert_to_libero_action(
        self, action_chunk: dict[str, np.array], idx: int = 0
    ) -> np.ndarray:
        """Convert GR00T action chunk to Libero format.

        Args:
            action_chunk: Dictionary of action components from GR00T policy
            idx: Index of action to extract from chunk (default: 0 for first action)

        Returns:
            7-dim numpy array: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        action_components = [
            np.atleast_1d(action_chunk[f"action.{key}"][idx])[0] for key in self.action_keys
        ]
        action_array = np.array(action_components, dtype=np.float32)
        action_array = normalize_gripper_action(action_array, binarize=True)
        assert len(action_array) == 7, f"Expected 7-dim action, got {len(action_array)}"
        return action_array


def eval_libero(cfg: GenerateConfig) -> None:
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_path = cfg.log_file or f"{log_dir}/libero_eval_{cfg.task_suite_name}.log"
    if os.path.dirname(log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    if cfg.print_step_latency and not cfg.profile_server:
        print_and_log(
            "[warning] --print_step_latency without --profile-server only measures client latency; "
            "server stages will be shown as n/a.",
            log_file,
        )

    # Decide which task indices to run
    if cfg.task_ids:
        task_indices = cfg.task_ids
    elif cfg.task_order:
        task_indices = cfg.task_order
    else:
        task_indices = list(range(num_tasks_in_suite))

    # Clamp indices to valid range and warn if needed
    task_indices = [idx for idx in task_indices if 0 <= idx < num_tasks_in_suite]

    # Start evaluation
    total_episodes, total_successes = 0, 0
    started_at = time.time()
    latency_values = defaultdict(list)
    memory_values = defaultdict(list)
    results = {
        "task_suite_name": cfg.task_suite_name,
        "num_trials_per_task": cfg.num_trials_per_task,
        "num_steps_wait": cfg.num_steps_wait,
        "port": cfg.port,
        "headless": cfg.headless,
        "task_indices": task_indices,
        "profile_server": cfg.profile_server,
        "tasks": [],
        "episodes": [],
        "step_timings": [],
        "step_memory": [],
    }
    try:
        task_iter = tqdm.tqdm(task_indices)
        for task_id in task_iter:
            # Get task
            task = task_suite.get_task(task_id)

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = get_libero_env(task, resolution=256)

            gr00t_policy = GR00TPolicy(
                host="localhost",
                port=cfg.port,
                headless=cfg.headless,
                profile_server=cfg.profile_server,
            )

            # Start episodes
            task_episodes, task_successes = 0, 0
            task_started_at = time.time()
            task_result = {
                "task_id": int(task_id),
                "task_description": task_description,
                "episodes": 0,
                "successes": 0,
                "success_rate": 0.0,
            }
            max_trials = min(cfg.num_trials_per_task, len(initial_states))
            for episode_idx in tqdm.tqdm(range(max_trials)):
                print(f"\nTask: {task_description}")
                log_file.write(f"\nTask: {task_description}\n")

                # Reset environment
                env.reset()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                done = False
                exception = None
                top_view = []
                wrist_view = []
                if cfg.task_suite_name == "libero_spatial":
                    max_steps = 220  # longest training demo has 193 steps
                elif cfg.task_suite_name == "libero_object":
                    max_steps = 280  # longest training demo has 254 steps
                elif cfg.task_suite_name == "libero_goal":
                    max_steps = 600  # longest training demo has 270 steps
                elif cfg.task_suite_name == "libero_10":
                    max_steps = 1000  # longest training demo has 505 steps
                elif cfg.task_suite_name == "libero_90":
                    max_steps = 400  # longest training demo has 373 steps

                print(f"Starting episode {task_episodes+1}...")
                log_file.write(f"Starting episode {task_episodes+1}...\n")
                episode_latency_values = defaultdict(list)
                episode_memory_values = defaultdict(list)
                while t < max_steps + cfg.num_steps_wait:
                    try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                        if t < cfg.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action())
                            t += 1
                            continue

                        if cfg.save_videos:
                            img, wrist_img = get_libero_image(obs)
                            top_view.append(img)
                            wrist_view.append(wrist_img)

                    # Query model to get action
                        action, step_timing, step_memory = gr00t_policy.get_action_with_timing(
                            obs,
                            task.language,
                        )

                    # Execute action in environment
                        env_step_start = time.perf_counter()
                        obs, reward, done, info = env.step(action.tolist())
                        step_timing["libero_env_step_ms"] = (
                            time.perf_counter() - env_step_start
                        ) * 1000.0
                        step_timing["control_loop_step_ms"] = (
                            step_timing["client_policy_total_ms"] + step_timing["libero_env_step_ms"]
                        )
                        if "server_policy_total_ms" in step_timing:
                            step_timing["server_unaccounted_ms"] = step_timing[
                                "server_policy_total_ms"
                            ] - (
                                step_timing.get("server_input_pack_ms", 0.0)
                                + step_timing.get("server_preprocess_transform_ms", 0.0)
                                + step_timing.get("server_model_total_ms", 0.0)
                                + step_timing.get("server_postprocess_untransform_ms", 0.0)
                            )
                        if "server_model_total_ms" in step_timing:
                            step_timing["server_model_unaccounted_ms"] = step_timing[
                                "server_model_total_ms"
                            ] - (
                                step_timing.get("server_model_prepare_input_to_device_ms", 0.0)
                                + step_timing.get("server_system2_backbone_ms", 0.0)
                                + step_timing.get("server_system1_action_head_ms", 0.0)
                                + step_timing.get("server_model_validate_cast_ms", 0.0)
                            )
                        compute_dual_system_step_timings(step_timing)
                        for key, value in step_timing.items():
                            latency_values[key].append(float(value))
                            episode_latency_values[key].append(float(value))
                        for key, value in step_memory.items():
                            memory_values[key].append(float(value))
                            episode_memory_values[key].append(float(value))
                        if cfg.print_step_latency:
                            preprocess_detail = format_preprocess_detail(step_timing)
                            print_and_log(
                                format_realtime_dual_system_latency_line(
                                    task_id,
                                    episode_idx,
                                    t,
                                    step_timing,
                                    latency_values,
                                ),
                                log_file,
                            )
                            tflops_line = format_realtime_tflops_line(
                                task_id,
                                episode_idx,
                                t,
                                step_timing,
                                latency_values,
                                cfg.dense_equiv_tflops_per_get_action,
                            )
                            if tflops_line:
                                print_and_log(tflops_line, log_file)
                            if cfg.print_detailed_step_latency:
                                if step_memory:
                                    print_and_log(
                                        format_realtime_memory_line(
                                            task_id,
                                            episode_idx,
                                            t,
                                            step_memory,
                                        ),
                                        log_file,
                                    )
                                print_and_log(
                                    (
                                        f"[latency detail] task={task_id} episode={episode_idx} step={t} "
                                        f"{format_step_latency(step_timing)}"
                                        + (
                                            f" | pre_detail: {preprocess_detail}"
                                            if preprocess_detail
                                            else ""
                                        )
                                    ),
                                    log_file,
                                )
                        if len(results["step_timings"]) < cfg.max_step_timing_records:
                            results["step_timings"].append(
                                {
                                    "task_id": int(task_id),
                                    "episode_idx": int(episode_idx),
                                    "step": int(t),
                                    **{k: float(v) for k, v in step_timing.items()},
                                }
                            )
                        if len(results["step_memory"]) < cfg.max_step_timing_records and step_memory:
                            results["step_memory"].append(
                                {
                                    "task_id": int(task_id),
                                    "episode_idx": int(episode_idx),
                                    "step": int(t),
                                    **{k: float(v) for k, v in step_memory.items()},
                                }
                            )
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        exception = str(e)
                        print(f"Caught exception: {e}")
                        log_file.write(f"Caught exception: {e}\n")
                        break

                task_episodes += 1
                total_episodes += 1
                task_result["episodes"] = int(task_episodes)
                task_result["successes"] = int(task_successes)
                task_result["success_rate"] = float(task_successes) / float(task_episodes)
                episode_summary = summarize_latency_values(episode_latency_values)
                episode_memory_summary = summarize_memory_values(episode_memory_values)
                results["episodes"].append(
                    {
                        "task_id": int(task_id),
                        "episode_idx": int(episode_idx),
                        "success": bool(done),
                        "steps": int(t),
                        "exception": exception,
                        "latency_summary": episode_summary,
                        "memory_summary": episode_memory_summary,
                    }
                )

            # Save a replay video of the episode
                if cfg.save_videos:
                    save_rollout_video(
                        top_view,
                        wrist_view,
                        total_episodes,
                        success=done,
                        task_description=task_description,
                        log_file=log_file,
                    )

            # Log current results
                print(f"Success: {done}")
                print(f"# episodes completed so far: {total_episodes}")
                print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                if cfg.print_latency:
                    total_summary = summarize_latency_values(latency_values)
                    total_memory_summary = summarize_memory_values(memory_values)
                    lines = [
                        (
                            "Episode latency: "
                            + format_dual_system_latency_line(episode_summary)
                        ),
                        (
                            "Running latency: "
                            + format_dual_system_latency_line(total_summary)
                        ),
                        "Running server memory: " + format_memory_line(total_memory_summary),
                    ]
                    for line in lines:
                        if line.rsplit(": ", 1)[-1]:
                            print_and_log(line, log_file)
                log_file.write(f"Success: {done}\n")
                log_file.write(f"# episodes completed so far: {total_episodes}\n")
                log_file.write(
                    f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n"
                )
                log_file.flush()

                results["total_episodes"] = int(total_episodes)
                results["total_successes"] = int(total_successes)
                results["success_rate"] = (
                    float(total_successes) / float(total_episodes) if total_episodes else 0.0
                )
                results["elapsed_sec"] = time.time() - started_at
                if cfg.live_json:
                    write_results_json(cfg.output_json, results, latency_values, memory_values)

        # Log final results
            print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
            log_file.write(
                f"Current task success rate: {float(task_successes) / float(task_episodes)}\n"
            )
            log_file.write(
                f"Current total success rate: {float(total_successes) / float(total_episodes)}\n"
            )
            log_file.flush()
            task_result["elapsed_sec"] = time.time() - task_started_at
            results["tasks"].append(task_result)
            if cfg.live_json:
                write_results_json(cfg.output_json, results, latency_values, memory_values)
    finally:
        results["total_episodes"] = int(total_episodes)
        results["total_successes"] = int(total_successes)
        results["success_rate"] = (
            float(total_successes) / float(total_episodes) if total_episodes else 0.0
        )
        results["elapsed_sec"] = time.time() - started_at
        write_results_json(cfg.output_json, results, latency_values, memory_values)
        log_file.close()

    # Save local log file


if __name__ == "__main__":
    cfg = tyro.cli(GenerateConfig)
    eval_libero(cfg)
