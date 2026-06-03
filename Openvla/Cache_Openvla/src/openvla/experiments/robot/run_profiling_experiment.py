"""End-to-end OpenVLA/VLA-Cache profiling on LIBERO."""

import inspect
import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import draccus
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

THIS_FILE = Path(__file__).resolve()
OPENVLA_ROOT = THIS_FILE.parents[2]
SRC_ROOT = OPENVLA_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
LIBERO_ROOT = SRC_ROOT / "LIBERO"

for path in (OPENVLA_ROOT, PROJECT_ROOT, LIBERO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.openvla_utils import get_processor, get_vla
from experiments.robot.robot_utils import (
    DEVICE,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


PRIMARY_STAGES = [
    "e2e_observation_to_action_ready",
    "data_observation_to_model_tensors_prompt",
    "vision_image_tensor_to_projector_output",
    "llm_multimodal_prefix_forward",
    "action_decode_or_denoise_to_continuous",
]


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_mem_mb() -> float:
    return float(torch.cuda.memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0


def _tensor_numel(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_numel(v) for v in value)
    if isinstance(value, dict):
        return sum(_tensor_numel(v) for v in value.values())
    return 0


def _estimate_leaf_flops(module: nn.Module, output: Any) -> int:
    if isinstance(module, nn.Linear):
        return int(2 * _tensor_numel(output) * module.in_features)
    if isinstance(module, nn.Conv2d) and torch.is_tensor(output):
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        return int(2 * output.numel() * kernel_ops)
    return 0


def _estimate_llama_layer_flops(module: nn.Module, hidden_states: Any) -> int:
    if not torch.is_tensor(hidden_states) or hidden_states.ndim < 3:
        return 0
    n = int(hidden_states.shape[1])
    if n == 1:
        return 0
    d = int(hidden_states.shape[2])
    up_proj = getattr(getattr(module, "mlp", None), "up_proj", None)
    m = int(getattr(up_proj, "out_features", 0) or 0)
    return int(4 * n * (d**2) + 2 * (n**2) * d + 3 * n * d * m) if d > 0 and m > 0 else 0


class StageProfiler:
    def __init__(self) -> None:
        self.records: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self.active: Dict[str, Dict[str, float]] = {}
        self.model_accum: Optional[Dict[str, Dict[str, float]]] = None
        self.stage_stack: List[str] = []
        self.handles: List[Any] = []
        self.enabled = True

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        self.start(name)
        try:
            yield
        finally:
            self.end(name)

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        _sync_cuda()
        self.active[name] = {"start": time.perf_counter(), "mem": _gpu_mem_mb()}

    def end(self, name: str) -> Dict[str, float]:
        if not self.enabled:
            return {
                "latency_ms": 0.0,
                "gpu_mem_mb": 0.0,
                "flops": 0.0,
                "tflops": 0.0,
                "throughput_tflops_per_s": 0.0,
            }
        state = self.active.pop(name)
        _sync_cuda()
        rec = {
            "latency_ms": (time.perf_counter() - state["start"]) * 1000,
            "gpu_mem_mb": _gpu_mem_mb() - state["mem"],
            "flops": 0.0,
            "tflops": 0.0,
            "throughput_tflops_per_s": 0.0,
        }
        self.records[name].append(rec)
        return rec

    def begin_model_step(self) -> None:
        if not self.enabled:
            self.model_accum = None
            self.stage_stack = []
            return
        self.model_accum = defaultdict(lambda: {"latency_ms": 0.0, "gpu_mem_mb": 0.0, "flops": 0.0})
        self.stage_stack = []

    def end_model_step(self) -> Dict[str, Dict[str, float]]:
        accum = self.model_accum or {}
        for stage, vals in accum.items():
            latency_ms = vals["latency_ms"]
            flops = vals["flops"]
            vals["tflops"] = flops * 1e-12
            vals["throughput_tflops_per_s"] = flops / (latency_ms / 1000) / 1e12 if latency_ms > 0 and flops > 0 else 0.0
            self.records[stage].append(dict(vals))
        self.model_accum = None
        self.stage_stack = []
        return dict(accum)

    def add_synthetic(self, name: str, latency_ms: float, flops: float = 0.0) -> None:
        if not self.enabled:
            return
        latency_ms = max(float(latency_ms), 0.0)
        self.records[name].append(
            {
                "latency_ms": latency_ms,
                "gpu_mem_mb": 0.0,
                "flops": flops,
                "tflops": flops * 1e-12,
                "throughput_tflops_per_s": flops / (latency_ms / 1000) / 1e12 if latency_ms > 0 and flops > 0 else 0.0,
            }
        )

    def _push_stage(self, stage_name: str, module: nn.Module) -> None:
        _sync_cuda()
        module.__profile_start = time.perf_counter()
        module.__profile_mem = _gpu_mem_mb()
        self.stage_stack.append(stage_name)

    def _pop_stage(self, stage_name: str, module: nn.Module) -> None:
        _sync_cuda()
        if self.model_accum is not None:
            self.model_accum[stage_name]["latency_ms"] += (time.perf_counter() - module.__profile_start) * 1000
            self.model_accum[stage_name]["gpu_mem_mb"] += _gpu_mem_mb() - module.__profile_mem
        if self.stage_stack:
            self.stage_stack.pop()

    def _leaf_hook(self, module: nn.Module, args: tuple, output: Any) -> None:
        if self.model_accum is None or not self.stage_stack:
            return
        if self.stage_stack[-1] in {"llm_multimodal_prefix_forward", "action_token_decode_forward", "llm_language_forward"}:
            return
        flops = _estimate_leaf_flops(module, output)
        if flops:
            self.model_accum[self.stage_stack[-1]]["flops"] += float(flops)

    def _llama_pre_hook(self, module: nn.Module, args: tuple, kwargs: Dict[str, Any]) -> None:
        if self.model_accum is None or not self.stage_stack:
            return
        if self.stage_stack[-1] not in {"llm_multimodal_prefix_forward", "action_token_decode_forward", "llm_language_forward"}:
            return
        hidden_states = kwargs.get("hidden_states") if kwargs else None
        if hidden_states is None and args:
            hidden_states = args[0]
        flops = _estimate_llama_layer_flops(module, hidden_states)
        if flops:
            self.model_accum[self.stage_stack[-1]]["flops"] += float(flops)

    def install_model_hooks(self, model: nn.Module) -> None:
        self.remove_hooks()

        def pre_fixed(stage: str):
            return lambda module, args, kwargs: self._push_stage(stage, module)

        def post_fixed(stage: str):
            return lambda module, args, kwargs, output: self._pop_stage(stage, module)

        def pre_language(module: nn.Module, args: tuple, kwargs: Dict[str, Any]) -> None:
            if kwargs.get("inputs_embeds") is not None:
                stage = "llm_multimodal_prefix_forward"
            elif kwargs.get("past_key_values") is not None:
                stage = "action_token_decode_forward"
            else:
                stage = "llm_language_forward"
            module.__profile_stage_name = stage
            self._push_stage(stage, module)

        def post_language(module: nn.Module, args: tuple, kwargs: Dict[str, Any], output: Any) -> None:
            self._pop_stage(module.__profile_stage_name, module)

        for module, stage in ((model.vision_backbone, "vision_image_tensor_to_projector_output"), (model.projector, "vision_image_tensor_to_projector_output")):
            self.handles.append(module.register_forward_pre_hook(pre_fixed(stage), with_kwargs=True))
            self.handles.append(module.register_forward_hook(post_fixed(stage), with_kwargs=True))
        self.handles.append(model.language_model.register_forward_pre_hook(pre_language, with_kwargs=True))
        self.handles.append(model.language_model.register_forward_hook(post_language, with_kwargs=True))
        for root in (model.vision_backbone, model.projector):
            for module in root.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    self.handles.append(module.register_forward_hook(self._leaf_hook))
        decoder_parent = getattr(model.language_model, "model", model.language_model)
        for layer in getattr(decoder_parent, "layers", []):
            self.handles.append(layer.register_forward_pre_hook(self._llama_pre_hook, with_kwargs=True))

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for stage, records in self.records.items():
            lat = np.array([r["latency_ms"] for r in records], dtype=np.float64)
            mem = np.array([r["gpu_mem_mb"] for r in records], dtype=np.float64)
            flops = np.array([r["flops"] for r in records], dtype=np.float64)
            tflops = np.array([r["tflops"] for r in records], dtype=np.float64)
            thr = np.array([r.get("throughput_tflops_per_s", 0.0) for r in records], dtype=np.float64)
            out[stage] = {
                "count": len(records),
                "latency_ms": {"mean": float(lat.mean()), "std": float(lat.std()), "min": float(lat.min()), "max": float(lat.max()), "p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))},
                "gpu_mem_mb": {"mean": float(mem.mean()), "max": float(mem.max())},
                "flops": {"mean": float(flops.mean()), "max": float(flops.max())},
                "tflops": {"mean": float(tflops.mean()), "max": float(tflops.max())},
                "throughput_tflops_per_s": {"mean": float(thr.mean()), "max": float(thr.max())},
            }
        return out


@dataclass
class ProfilingConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "checkpoints/openvla-7b-finetuned-libero-spatial"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    use_vla_cache: bool = True
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 10
    max_tasks: int = 1
    max_steps: Optional[int] = None
    warmup_action_steps: int = 1
    static_patch_top_k: int = 150
    static_patch_sim_threshold: float = 0.996
    attention_top_k: int = 120
    output_dir: str = "./profiling_results"
    seed: int = 7
    check_only: bool = False


def _close_env(env: Any) -> None:
    with suppress(Exception):
        if env is not None:
            env.close()


def _check_environment(checkpoint_path: Path) -> None:
    import tokenizers
    import transformers
    import transformers.models.llama.modeling_llama as llama_modeling

    llama_forward = inspect.getsource(llama_modeling.LlamaModel.forward)
    has_hooks = "reusable_patches" in llama_forward and "proportion_attn_var" in llama_forward
    print(f"Python: {sys.executable}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Tokenizers: {tokenizers.__version__}")
    print(f"VLA-Cache Llama hooks: {has_hooks}")
    if not has_hooks:
        raise RuntimeError("Current transformers package does not contain VLA-Cache Llama hooks")
    print("Check passed.")


@draccus.wrap()
def run_profiling(cfg: ProfilingConfig) -> None:
    print("\n" + "=" * 80)
    print("OpenVLA/VLA-Cache end-to-end profiling")
    print(f"VLA-Cache: {'enabled' if cfg.use_vla_cache else 'disabled'}")
    print("=" * 80 + "\n")
    checkpoint_path = Path(cfg.pretrained_checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    if cfg.check_only:
        _check_environment(checkpoint_path)
        return

    cfg.pretrained_checkpoint = str(checkpoint_path)
    cfg.unnorm_key = cfg.task_suite_name
    set_seed_everywhere(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiler = StageProfiler()
    step_records: List[Dict[str, Any]] = []
    total_episodes = 0
    total_successes = 0

    print("[*] Loading model and processor...")
    with profiler.stage("model_loading"):
        model = get_vla(cfg)
        processor = get_processor(cfg)
    profiler.install_model_hooks(model)
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    num_tasks = min(cfg.max_tasks, task_suite.n_tasks)
    resize_size = get_image_resize_size(cfg)

    try:
        for task_id in range(num_tasks):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env = None
            try:
                env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
                print(f"\nTask {task_id}: {task_description}")
                for episode_idx in range(cfg.num_trials_per_task):
                    print(f"  Episode {episode_idx + 1}/{cfg.num_trials_per_task}")
                    env.reset()
                    obs = env.set_init_state(initial_states[episode_idx])
                    t = 0
                    prev_img = None
                    last_caches = None
                    max_steps = cfg.max_steps or (220 if cfg.task_suite_name == "libero_spatial" else 300)
                    while t < max_steps + cfg.num_steps_wait:
                        if t < cfg.num_steps_wait:
                            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                            t += 1
                            continue
                        action_step_idx = t - cfg.num_steps_wait
                        record_step = action_step_idx >= cfg.warmup_action_steps
                        profiler.set_enabled(record_step)

                        profiler.start("e2e_observation_to_action_ready")
                        with profiler.stage("data_observation_to_model_tensors_prompt"):
                            img = get_libero_image(obs, resize_size)
                            if prev_img is None:
                                prev_img = img
                            image = Image.fromarray(img).convert("RGB")
                            prev_image = Image.fromarray(prev_img).convert("RGB")
                            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
                            inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

                        prompt_cache = last_caches.get("past_key_values") if (cfg.use_vla_cache and last_caches) else None
                        model.language_model.config.reusable_patches = None
                        model.language_model.config.proportion_attn_var = None
                        stable_patch_count = reusable_token_count = 0
                        reuse_schedule_mean = 0.0
                        reuse_schedule_pruning_layers: List[float] = []

                        if cfg.use_vla_cache and last_caches is not None:
                            with profiler.stage("vla_cache_reuse_policy_prepare"):
                                from experiments.robot.vla_cache_utils import find_static_patches, get_layer_mask_schedule, task_relevant_selection

                                prev_attn = last_caches.get("attentions")
                                if prompt_cache is not None and prev_attn is not None:
                                    stable_patches = find_static_patches(image, prev_image, top_k=cfg.static_patch_top_k, sim_threshold=cfg.static_patch_sim_threshold)
                                    _, remaining_tokens = task_relevant_selection(prev_attn, image, stable_patches, top_k=cfg.attention_top_k)
                                    schedule_raw = get_layer_mask_schedule(prev_attn)
                                    schedule = schedule_raw.detach().cpu().tolist() if isinstance(schedule_raw, torch.Tensor) else list(schedule_raw or [])
                                    stable_patch_count = len(stable_patches)
                                    reusable_token_count = len(remaining_tokens)
                                    if schedule:
                                        reuse_schedule_mean = float(np.mean(schedule))
                                        reuse_schedule_pruning_layers = [float(schedule[i]) for i in (2, 6, 9, 11) if i < len(schedule)]
                                    model.language_model.config.reusable_patches = torch.tensor(remaining_tokens, device=DEVICE) if remaining_tokens else None
                                    model.language_model.config.proportion_attn_var = schedule

                        profiler.begin_model_step()
                        with profiler.stage("model_generate_total"):
                            action, last_caches = model.predict_action(
                                **inputs,
                                unnorm_key=cfg.unnorm_key,
                                do_sample=False,
                                return_dict_in_generate=True,
                                output_attentions=True,
                                past_key_values=prompt_cache,
                            )
                        model_accum = profiler.end_model_step()

                        vision_ms = model_accum.get("vision_image_tensor_to_projector_output", {}).get("latency_ms", 0.0)
                        prefix_ms = model_accum.get("llm_multimodal_prefix_forward", {}).get("latency_ms", 0.0)
                        action_ms = model_accum.get("action_token_decode_forward", {}).get("latency_ms", 0.0)
                        prefix_flops = model_accum.get("llm_multimodal_prefix_forward", {}).get("flops", 0.0)
                        prefix_tflops = model_accum.get("llm_multimodal_prefix_forward", {}).get("tflops", 0.0)
                        if record_step:
                            model_total = profiler.records["model_generate_total"][-1]["latency_ms"]
                            profiler.add_synthetic("action_decode_or_denoise_to_continuous", model_total - vision_ms - prefix_ms)

                        with profiler.stage("robot_action_postprocess"):
                            action = normalize_gripper_action(action, binarize=True)
                            action = invert_gripper_action(action)
                        profiler.end("e2e_observation_to_action_ready")

                        step_records.append(
                            {
                                "task_id": task_id,
                                "episode_idx": episode_idx,
                                "env_step": action_step_idx,
                                "recorded": record_step,
                                "used_vla_cache": bool(prompt_cache is not None),
                                "vision_ms": vision_ms,
                                "llm_prefix_ms": prefix_ms,
                                "llm_prefix_flops": prefix_flops,
                                "llm_prefix_tflops_sample": prefix_tflops,
                                "action_forward_ms": action_ms,
                                "stable_patch_count": stable_patch_count,
                                "reusable_token_count": reusable_token_count,
                                "reuse_schedule_mean": reuse_schedule_mean,
                                "reuse_schedule_pruning_layers": reuse_schedule_pruning_layers,
                            }
                        )
                        obs, _, done, _ = env.step(action.tolist())
                        prev_img = img
                        if done:
                            total_successes += 1
                            break
                        t += 1
                    total_episodes += 1
            finally:
                profiler.set_enabled(True)
                _close_env(env)
    finally:
        profiler.set_enabled(True)
        profiler.remove_hooks()

    recorded = sum(1 for r in step_records if r.get("recorded"))
    if recorded == 0:
        raise RuntimeError("No action steps were recorded. Lower WARMUP_ACTION_STEPS or increase MAX_STEPS.")
    summary = profiler.summary()
    summary["config"] = {
        "use_vla_cache": cfg.use_vla_cache,
        "task_suite": cfg.task_suite_name,
        "num_tasks": num_tasks,
        "num_episodes": total_episodes,
        "warmup_action_steps_per_episode": cfg.warmup_action_steps,
        "static_patch_top_k": cfg.static_patch_top_k,
        "static_patch_sim_threshold": cfg.static_patch_sim_threshold,
        "attention_top_k": cfg.attention_top_k,
        "num_recorded_action_steps": recorded,
        "success_rate": total_successes / total_episodes if total_episodes else 0.0,
        "checkpoint": str(checkpoint_path),
        "tflops_note": "The tflops field is theoretical TFLOPs/sample using the original VLA-Cache Llama decoder formula; it is not TFLOPS/s throughput.",
    }
    summary["per_step"] = step_records
    output_file = output_dir / f"profiling_{'with' if cfg.use_vla_cache else 'without'}_cache.json"
    output_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    run_profiling()
