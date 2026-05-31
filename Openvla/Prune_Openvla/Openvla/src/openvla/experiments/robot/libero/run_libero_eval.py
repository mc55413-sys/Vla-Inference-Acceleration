"""
run_libero_eval.py
Runs a model in a LIBERO simulation environment.
"""

import os
import sys
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import imageio
import draccus
import numpy as np
import tqdm
import wandb

script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent.parent
libero_path = project_root / "LIBERO"
if str(libero_path) not in sys.path:
    sys.path.insert(0, str(libero_path))

from libero.libero import benchmark

# Append current directory so that interpreter can find experiments.robot
sys.path.insert(0, str(project_root))
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


def _new_latency_stats() -> dict:
    return {"model_sum": 0.0, "e2e_sum": 0.0, "count": 0}


def _update_latency_stats(stats: dict, model_latency_ms: float, e2e_latency_ms: float) -> None:
    if not (np.isfinite(model_latency_ms) and np.isfinite(e2e_latency_ms)):
        return
    stats["model_sum"] += model_latency_ms
    stats["e2e_sum"] += e2e_latency_ms
    stats["count"] += 1


def _latency_avg_metrics(stats: dict) -> dict:
    if stats["count"] == 0:
        return {"avg_model_latency_ms": float("nan"), "avg_e2e_latency_ms": float("nan")}
    return {
        "avg_model_latency_ms": stats["model_sum"] / stats["count"],
        "avg_e2e_latency_ms": stats["e2e_sum"] / stats["count"],
    }


def _format_ms(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.2f}"


def _format_latency_summary(prefix: str, stats: dict) -> str:
    avg = _latency_avg_metrics(stats)
    return (
        f"{prefix} avg_Model_Latency={_format_ms(avg['avg_model_latency_ms'])} ms | "
        f"avg_E2E_Latency={_format_ms(avg['avg_e2e_latency_ms'])} ms | count={stats['count']}"
    )


def _format_step_metrics(
    task_id: int, episode_idx: int, action_step: int, latency_ms: float, metrics: dict, warmup: bool = False
) -> str:
    tflops = metrics.get("tflops")
    og_tflops = metrics.get("og_tflops")
    prefix = "[metrics-warmup]" if warmup else "[metrics]"
    model_latency_ms = metrics.get("model_latency_ms", metrics.get("predict_latency_ms", latency_ms))
    data_ms = metrics.get("data_ms", 0.0)
    preprocess_ms = metrics.get("preprocess_ms", 0.0)
    e2e_latency_ms = metrics.get("e2e_latency_ms", model_latency_ms + data_ms + preprocess_ms)
    vision_ms = metrics.get("vision_ms", 0.0)
    llm_ms = metrics.get("llm_ms", 0.0)
    action_ms = metrics.get("action_ms", metrics.get("action_decode_ms", 0.0))
    avg_model_latency_ms = metrics.get("avg_model_latency_ms", float("nan"))
    avg_e2e_latency_ms = metrics.get("avg_e2e_latency_ms", float("nan"))
    latency_parts = (
        f"Model_Latency={_format_ms(model_latency_ms)} ms | E2E_Latency={_format_ms(e2e_latency_ms)} ms | "
        f"avg_Model_Latency={_format_ms(avg_model_latency_ms)} ms | "
        f"avg_E2E_Latency={_format_ms(avg_e2e_latency_ms)} ms | "
        f"data={_format_ms(data_ms)} preprocess={_format_ms(preprocess_ms)} vision={_format_ms(vision_ms)} "
        f"llm={_format_ms(llm_ms)} action={_format_ms(action_ms)}"
    )
    if tflops is None:
        return f"{prefix} task={task_id} episode={episode_idx} step={action_step} {latency_parts} | tflops=n/a"
    return (
        f"{prefix} task={task_id} episode={episode_idx} step={action_step} "
        f"{latency_parts} | tflops={tflops:.4f} T | og_tflops={og_tflops:.4f} T"
    )


@dataclass
class GenerateConfig:
    # Use VLA-Pruner for faster inference
    # FastV Token Pruning Configuration
    use_fastv: bool = True              #enable fastvforward for token pruning
    fastv_k: int = 3                    # Layer to start pruning
    fastv_r: float = 0.50               # Pruning ratio (tokens to remove)
    fastv_image_token_start_index: int = 1  # Image token start index
    fastv_image_token_length: int = 256     # Image token length
    sparsevlm: bool = False         # enable sparsevlm
    #VLA-Pruner Settings
    use_temporal: bool = True      # Whether to use temporal attention guidance
    temporal_w: int = 3           # Temporal window size
    temporal_gamma: float = 0.80  # Temporal weight decay factor
    #use test-to-vison attention or prefill attention
    use_text_vision_selection: bool = False
    use_prefil_attention: bool = False
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "checkpoints/openvla-7b-finetuned-libero-spatial"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    latency_warmup_steps: int = 1                    # Number of action latencies to mark as warmup per episode
    num_trials_per_task: int = 10                    # Number of rollouts per task
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under
    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    # Set random seed
    set_seed_everywhere(cfg.seed)
    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name
    # Load model
    model = get_model(cfg)
    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"
    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )
    log_file.write("=" * 80 + "\n")
    log_file.write("CONFIGURATION SUMMARY:\n")
    log_file.write("=" * 80 + "\n")
    for field_name, field_value in cfg.__dict__.items():
        log_file.write(f"  {field_name}: {field_value}\n")
    log_file.write("=" * 80 + "\n")
    log_file.write("\n")
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)
    # Start evaluation
    total_episodes, total_successes = 0, 0
    total_latency_stats = _new_latency_stats()
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task_latency_stats = _new_latency_stats()
        # Get task
        task = task_suite.get_task(task_id)
        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")
            # Reset environment
            env.reset()
            model.reset_av_history()
            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            # Setup
            t = 0
            replay_images = []
            replay_images_heatmap = []
            prev_img = None
            last_caches = None
            episode_latency_stats = _new_latency_stats()
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps
            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue
                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)
                    # Save previous image
                    if prev_img is None:
                        prev_img = img
                    else:
                        prev_img = replay_images[-1]
                    # Save preprocessed image for replay video
                    replay_images.append(img)
                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "prev_image": prev_img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    # Query model to get action
                    action, last_caches, result_image = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        last_caches=last_caches,
                    )
                    action_step = t - cfg.num_steps_wait + 1
                    step_metrics = dict(getattr(model, "last_vlapruner_metrics", {}) or {})
                    latency_ms = step_metrics.get("latency_ms", float("nan"))
                    model_latency_ms = step_metrics.get(
                        "model_latency_ms", step_metrics.get("predict_latency_ms", latency_ms)
                    )
                    e2e_latency_ms = step_metrics.get(
                        "e2e_latency_ms",
                        model_latency_ms + step_metrics.get("data_ms", 0.0) + step_metrics.get("preprocess_ms", 0.0),
                    )
                    is_warmup = action_step <= cfg.latency_warmup_steps
                    if not is_warmup:
                        _update_latency_stats(episode_latency_stats, model_latency_ms, e2e_latency_ms)
                        _update_latency_stats(task_latency_stats, model_latency_ms, e2e_latency_ms)
                        _update_latency_stats(total_latency_stats, model_latency_ms, e2e_latency_ms)
                        step_metrics.update(_latency_avg_metrics(total_latency_stats))
                    else:
                        step_metrics.update(
                            {"avg_model_latency_ms": float("nan"), "avg_e2e_latency_ms": float("nan")}
                        )
                    metrics_msg = _format_step_metrics(
                        task_id,
                        episode_idx + 1,
                        action_step,
                        latency_ms,
                        step_metrics,
                        warmup=is_warmup,
                    )
                    print(metrics_msg, flush=True)
                    log_file.write(metrics_msg + "\n")
                    log_file.flush()
                    replay_images_heatmap.append(result_image)
                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)
                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)
                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
            task_episodes += 1
            total_episodes += 1

            save_rollout_video(
                        replay_images_heatmap, total_episodes, success=done, task_description=task_description, log_file=log_file
                    )
            # Save a replay video of the episode
            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            episode_latency_msg = _format_latency_summary("[latency-episode]", episode_latency_stats)
            print(episode_latency_msg)
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.write(episode_latency_msg + "\n")
            log_file.flush()
        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        task_latency_msg = _format_latency_summary("[latency-task]", task_latency_stats)
        total_latency_msg = _format_latency_summary("[latency-total-running]", total_latency_stats)
        print(task_latency_msg)
        print(total_latency_msg)
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.write(task_latency_msg + "\n")
        log_file.write(total_latency_msg + "\n")
        log_file.flush()
        if cfg.use_wandb:
            task_latency_avg = _latency_avg_metrics(task_latency_stats)
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                    f"latency/avg_Model_Latency_ms/{task_description}": task_latency_avg["avg_model_latency_ms"],
                    f"latency/avg_E2E_Latency_ms/{task_description}": task_latency_avg["avg_e2e_latency_ms"],
                }
            )
    # Save local log file
    final_latency_msg = _format_latency_summary("[latency-final]", total_latency_stats)
    print(final_latency_msg)
    log_file.write(final_latency_msg + "\n")
    log_file.flush()
    log_file.close()
    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        final_latency_avg = _latency_avg_metrics(total_latency_stats)
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
                "latency/avg_Model_Latency_ms/total": final_latency_avg["avg_model_latency_ms"],
                "latency/avg_E2E_Latency_ms/total": final_latency_avg["avg_e2e_latency_ms"],
            }
        )
        wandb.save(local_log_filepath)

if __name__ == "__main__":
    eval_libero()
