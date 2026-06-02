"""VLA quantization latency, memory, and size profiling CLI.

Timing methodology follows VLA-Pruner with maximally expanded LLM scope:
  - model_latency_ms = wall-clock from before data-load to after action (includes data+preprocess+model)
  - vision_ms = vision backbone kernel-launch only (no sync; GPU exec falls into LLM window)
  - action_ms = numpy decode only (GPU->CPU transfer falls into LLM window)
  - llm_ms = model_latency_ms - vision_ms - action_ms
    (captures data/preprocess + backbone-GPU + projector + embedding + prefill + decode + D2H + overhead)

Latency breakdown:
  End-to-End Latency = Model (data+preprocess already included)
  Model Latency      = Vision + LLM + Action
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import importlib.util
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

OPENVLA_ROOT = Path(__file__).resolve().parents[2]
if str(OPENVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENVLA_ROOT))

from experiments.robot.openvla_direct_quant import (  # noqa: E402
    DirectQuantConfig,
    estimate_model_size_mb,
    get_direct_quant_info,
    quantize_openvla_language_model,
)
from experiments.robot.openvla_fast_action import (  # noqa: E402
    build_multimodal_inputs,
    can_use_fast_action_head,
    can_use_last_token_logits,
    decode_action_tokens,
    generate_action_tokens_fast,
    generate_action_tokens_last_logits,
    get_language_backbone,
    limit_llm_layers,
    reduce_vision_tokens,
)


SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
LATENCY_FIELDS = ("data_ms", "preprocess_ms", "vision_ms", "llm_ms", "action_ms")
SUMMARY_STATS = ("mean", "std", "p50", "p90", "p95", "min", "max")
FP8_REPORT_LABEL = "w4a8"


class _MiniSeries:
    def __init__(self, values: Sequence[Any]) -> None:
        self.values = list(values)

    def to_numpy(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)


class _MiniDataFrame:
    def __init__(self, records: Optional[Sequence[Dict[str, Any]]] = None) -> None:
        self.records = list(records or [])
        columns: List[str] = []
        for record in self.records:
            for key in record:
                if key not in columns:
                    columns.append(key)
        self.columns = columns

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return _MiniSeries([record.get(key) for record in self.records])
        return _MiniDataFrame([{column: record.get(column) for column in key} for record in self.records])

    def to_csv(self, path: Path, index: bool = False) -> None:
        del index
        import csv

        with Path(path).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.records)

    def to_dict(self, orient: str = "records") -> List[Dict[str, Any]]:
        if orient != "records":
            raise ValueError("_MiniDataFrame only supports orient='records'")
        return list(self.records)

    def to_string(self, index: bool = False) -> str:
        del index
        if not self.records:
            return ""
        widths = {
            column: max(len(str(column)), *(len(str(record.get(column, ""))) for record in self.records))
            for column in self.columns
        }
        header = " ".join(str(column).ljust(widths[column]) for column in self.columns)
        rows = [
            " ".join(str(record.get(column, "")).ljust(widths[column]) for column in self.columns)
            for record in self.records
        ]
        return "\n".join([header, *rows])


class _MiniPandas:
    DataFrame = _MiniDataFrame

    @staticmethod
    def concat(dataframes: Sequence[_MiniDataFrame], ignore_index: bool = True) -> _MiniDataFrame:
        del ignore_index
        records: List[Dict[str, Any]] = []
        for dataframe in dataframes:
            records.extend(dataframe.to_dict(orient="records"))
        return _MiniDataFrame(records)


if pd is None:
    pd = _MiniPandas()


@dataclass
class TimerConfig:
    device: torch.device
    suppress_stage_output: bool = True


@dataclass
class OpenVLASample:
    image: Image.Image
    instruction: str
    unnorm_key: Optional[str]


@dataclass
class VisionOutputs:
    projected_patch_embeddings: torch.Tensor


@dataclass
class LLMOutputs:
    generated_action_token_ids: torch.Tensor


class VLAInferenceAdapter(ABC):
    """Adapter interface consumed by the profiler."""

    def __init__(self, model: torch.nn.Module, processor: Any = None, device: str = "cuda") -> None:
        self.model = model
        self.processor = processor
        self.device = torch.device(device)

    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def quant_info(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_sample(self, idx: int) -> OpenVLASample:
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, raw_sample: OpenVLASample) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def vision_forward(self, batch: Dict[str, Any]) -> VisionOutputs:
        raise NotImplementedError

    @abstractmethod
    def llm_forward(self, vision_outputs: VisionOutputs, batch: Dict[str, Any]) -> LLMOutputs:
        raise NotImplementedError

    @abstractmethod
    def action_decode(self, llm_outputs: LLMOutputs, batch: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError


class OpenVLAInferenceAdapter(VLAInferenceAdapter):
    """Adapter that profiles OpenVLA's processor, vision/projector, LLM, and action decode separately."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        model_path: str,
        device: str,
        dtype: torch.dtype,
        instruction: str,
        image_path: Optional[str] = None,
        unnorm_key: Optional[str] = None,
        quant_label: str = "baseline",
        fast_action_head: bool = False,
        last_token_logits: bool = False,
        max_vision_tokens: int = 0,
        vision_token_strategy: str = "uniform",
        drop_full_attention_mask: bool = True,
    ) -> None:
        super().__init__(model=model, processor=processor, device=device)
        self.model_path = model_path
        self.dtype = dtype
        self.instruction = instruction
        self.image_path = image_path
        self.unnorm_key = unnorm_key
        self.quant_label = quant_label
        self.fast_action_head = fast_action_head and can_use_fast_action_head(model)
        self.last_token_logits = last_token_logits and can_use_last_token_logits(model)
        self.max_vision_tokens = max_vision_tokens
        self.vision_token_strategy = vision_token_strategy
        self.drop_full_attention_mask = drop_full_attention_mask
        self._synthetic_image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).convert("RGB")

    def model_name(self) -> str:
        base = Path(str(self.model_path)).name or str(self.model_path)
        return f"{base}:{self.display_quant_label()}"

    def display_quant_label(self) -> str:
        normalized = self.quant_label.lower().replace("-", "_")
        if normalized in {"fp8", "fp8_tensorwise", "w4a8"}:
            return FP8_REPORT_LABEL
        return self.quant_label

    def quant_info(self) -> Dict[str, Any]:
        direct_info = get_direct_quant_info(self.model)
        mode = str(direct_info.get("mode", "none"))
        if mode != "none":
            if mode == "w8a8":
                return {
                    "quant_method": direct_info.get("backend", "direct-w8a8"),
                    "weight_bits": 8,
                    "activation_bits": 8,
                    "kv_cache_bits": 16,
                }
            if mode == "w8a16":
                return {
                    "quant_method": direct_info.get("backend", "direct-w8a16"),
                    "weight_bits": 8,
                    "activation_bits": 16,
                    "kv_cache_bits": 16,
                }
            if mode == "w4a16":
                return {
                    "quant_method": direct_info.get("backend", "direct-w4a16"),
                    "weight_bits": 4,
                    "activation_bits": 16,
                    "kv_cache_bits": 16,
                }
            if mode == "fp8":
                return {
                    "quant_method": FP8_REPORT_LABEL,
                    "weight_bits": 4,
                    "activation_bits": 8,
                    "kv_cache_bits": 16,
                }
            if mode == "bnb_int8":
                return {
                    "quant_method": direct_info.get("backend", "bitsandbytes-Linear8bitLt"),
                    "weight_bits": 8,
                    "activation_bits": 8,
                    "kv_cache_bits": 16,
                }
            if mode in {"bnb_nf4", "bnb_fp4"}:
                return {
                    "quant_method": direct_info.get("backend", f"bitsandbytes-Linear4bit-{mode.removeprefix('bnb_')}"),
                    "weight_bits": 4,
                    "activation_bits": 16,
                    "kv_cache_bits": 16,
                }

        dtype_name = "bf16" if self.dtype == torch.bfloat16 else "fp16" if self.dtype == torch.float16 else str(self.dtype)
        return {
            "quant_method": dtype_name,
            "weight_bits": 16,
            "activation_bits": 16,
            "kv_cache_bits": 16,
        }

    def load_sample(self, idx: int) -> OpenVLASample:
        del idx
        if self.image_path is None:
            image = self._synthetic_image.copy()
        else:
            image = Image.open(self.image_path).convert("RGB")
        return OpenVLASample(image=image, instruction=self.instruction, unnorm_key=self.unnorm_key)

    def preprocess_cpu(self, raw_sample: OpenVLASample) -> Dict[str, Any]:
        """CPU-only preprocess: processor + tokenizer. GPU transfer is done separately."""
        prompt = get_openvla_prompt(raw_sample.instruction, self.model_path)
        batch = self.processor(prompt, raw_sample.image)
        input_ids = batch["input_ids"]
        if not torch.all(input_ids[:, -1] == 29871):
            empty_token = torch.full((input_ids.shape[0], 1), 29871, dtype=input_ids.dtype, device=input_ids.device)
            batch["input_ids"] = torch.cat((input_ids, empty_token), dim=1)
            if "attention_mask" in batch and batch["attention_mask"] is not None:
                extra_mask = torch.ones(
                    (batch["attention_mask"].shape[0], 1),
                    dtype=batch["attention_mask"].dtype,
                    device=batch["attention_mask"].device,
                )
                batch["attention_mask"] = torch.cat((batch["attention_mask"], extra_mask), dim=1)
        attention_mask = batch.get("attention_mask")
        if self.drop_full_attention_mask and torch.is_tensor(attention_mask) and bool(torch.all(attention_mask != 0)):
            batch["attention_mask"] = None
        batch["unnorm_key"] = raw_sample.unnorm_key
        return batch

    def preprocess(self, raw_sample: OpenVLASample) -> Dict[str, Any]:
        """Full preprocess including GPU transfer (kept for backward compatibility)."""
        batch = self.preprocess_cpu(raw_sample)
        return move_batch_to_device(batch, self.device, self.dtype)

    def vision_backbone_forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Vision backbone only — included in vision_ms measurement."""
        pixel_values = batch["pixel_values"]
        return self.model.vision_backbone(pixel_values)

    def projector_forward(self, patch_features: torch.Tensor) -> VisionOutputs:
        """Projector + token reduction — falls into LLM timing window via subtraction."""
        projected_patch_embeddings = self.model.projector(patch_features)
        projected_patch_embeddings = reduce_vision_tokens(
            projected_patch_embeddings,
            max_vision_tokens=self.max_vision_tokens,
            strategy=self.vision_token_strategy,
        )
        return VisionOutputs(projected_patch_embeddings=projected_patch_embeddings)

    def vision_forward(self, batch: Dict[str, Any]) -> VisionOutputs:
        """Full vision pipeline (backbone + projector) — kept for backward compatibility."""
        patch_features = self.vision_backbone_forward(batch)
        return self.projector_forward(patch_features)

    def action_decode_cpu(self, token_ids_cpu: torch.Tensor, batch: Dict[str, Any]) -> np.ndarray:
        """Numpy-only action decode from already-CPU token ids."""
        predicted_action_token_ids = token_ids_cpu.numpy()
        discretized_actions = self.model.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.model.bin_centers.shape[0] - 1)
        normalized_actions = self.model.bin_centers[discretized_actions]

        action_norm_stats = self.model.get_action_stats(batch.get("unnorm_key"))
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        return np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

    def _build_multimodal_inputs(
        self,
        vision_outputs: VisionOutputs,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return build_multimodal_inputs(
            self.model,
            batch["input_ids"],
            vision_outputs.projected_patch_embeddings,
            batch.get("attention_mask"),
        )

    def llm_forward(self, vision_outputs: VisionOutputs, batch: Dict[str, Any]) -> LLMOutputs:
        action_dim = self.model.get_action_dim(batch.get("unnorm_key"))
        generated_tokens: List[torch.Tensor] = []
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_inputs(vision_outputs, batch)

        if self.fast_action_head:
            outputs = generate_action_tokens_fast(
                self.model,
                multimodal_embeddings,
                multimodal_attention_mask,
                batch.get("unnorm_key"),
            )
            return LLMOutputs(generated_action_token_ids=outputs.generated_action_token_ids)

        if self.last_token_logits:
            outputs = generate_action_tokens_last_logits(
                self.model,
                multimodal_embeddings,
                multimodal_attention_mask,
                batch.get("unnorm_key"),
            )
            return LLMOutputs(generated_action_token_ids=outputs.generated_action_token_ids)

        outputs = self.model.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        generated_tokens.append(next_token)
        past_key_values = outputs.past_key_values

        for _ in range(action_dim - 1):
            outputs = self.model.language_model(
                input_ids=next_token[:, None],
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            generated_tokens.append(next_token)
            past_key_values = outputs.past_key_values

        return LLMOutputs(generated_action_token_ids=torch.stack(generated_tokens, dim=1))

    def action_decode(self, llm_outputs: LLMOutputs, batch: Dict[str, Any]) -> np.ndarray:
        return decode_action_tokens(self.model, llm_outputs.generated_action_token_ids[0], batch.get("unnorm_key"))

class VLALatencyProfiler:
    def __init__(
        self,
        adapter: VLAInferenceAdapter,
        timer_config: TimerConfig,
        profile_memory: bool = True,
        compressed_model_size_mb: Optional[float] = None,
        print_raw_measurements: bool = True,
    ) -> None:
        self.adapter = adapter
        self.timer_config = timer_config
        self.profile_memory = profile_memory
        self.compressed_model_size_mb = compressed_model_size_mb
        self.print_raw_measurements = print_raw_measurements

    @torch.inference_mode()
    def profile(self, warmup_steps: int, repeat_steps: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        self.adapter.model.eval()
        clear_cuda_memory(self.timer_config.device)

        if warmup_steps > 0:
            print(f"[profile] warmup steps={warmup_steps}", flush=True)
        for idx in range(warmup_steps):
            self._run_one(idx, collect_timing=False)

        clear_cuda_memory(self.timer_config.device)
        print(f"[profile] measuring steps={repeat_steps}", flush=True)
        raw_records: List[Dict[str, Any]] = []
        for idx in range(repeat_steps):
            record = self._run_one(idx, collect_timing=True)
            raw_records.append(record)
            if self.print_raw_measurements:
                print(format_raw_measurement(record), flush=True)

        raw_df = pd.DataFrame(raw_records)
        summary = summarize_raw_records(
            raw_df=raw_df,
            adapter=self.adapter,
            model=self.adapter.model,
            device=self.timer_config.device,
            profile_memory=self.profile_memory,
            compressed_model_size_mb=self.compressed_model_size_mb,
        )
        return raw_df, summary

    def _run_one(self, idx: int, collect_timing: bool) -> Dict[str, Any]:
        timer = StageTimer(self.timer_config)

        # Data & Preprocess
        raw_sample, data_ms = timer.time_cpu(lambda: self.adapter.load_sample(idx), sync_after=False)
        batch, preprocess_ms = timer.time_cpu(lambda: self.adapter.preprocess(raw_sample), sync_after=True)

        # Model latency (vision → LLM → action)
        _sync_if_cuda(self.timer_config.device)
        t_model_start = time.perf_counter()

        # Vision
        t_vision_start = time.perf_counter()
        vision_outputs = self.adapter.vision_forward(batch)
        _sync_if_cuda(self.timer_config.device)
        vision_ms = (time.perf_counter() - t_vision_start) * 1000.0

        # LLM
        llm_outputs = self.adapter.llm_forward(vision_outputs, batch)
        _sync_if_cuda(self.timer_config.device)
        t_after_llm = time.perf_counter()

        # Action
        action = self.adapter.action_decode(llm_outputs, batch)
        _sync_if_cuda(self.timer_config.device)
        action_ms = (time.perf_counter() - t_after_llm) * 1000.0

        model_latency_ms = (time.perf_counter() - t_model_start) * 1000.0
        llm_ms = max(0.0, model_latency_ms - vision_ms - action_ms)

        if not collect_timing:
            return {}

        end_to_end_latency_ms = data_ms + preprocess_ms + model_latency_ms
        record = {
            "step": idx,
            "model_name": self.adapter.model_name(),
            **self.adapter.quant_info(),
            "data_ms": data_ms,
            "preprocess_ms": preprocess_ms,
            "vision_ms": vision_ms,
            "llm_ms": llm_ms,
            "action_ms": action_ms,
            "model_latency_ms": model_latency_ms,
            "end_to_end_latency_ms": end_to_end_latency_ms,
        }
        return record


class StageTimer:
    def __init__(self, config: TimerConfig) -> None:
        self.config = config

    def time_cpu(self, fn: Any, sync_after: bool = False) -> Tuple[Any, float]:
        _sync_if_cuda(self.config.device)
        start = time.perf_counter()
        with self._stage_output_context():
            result = fn()
        if sync_after:
            _sync_if_cuda(self.config.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return result, elapsed_ms

    def time_wallclock(self, fn: Any) -> Tuple[Any, float]:
        """Wall-clock GPU timing: sync before and after, use perf_counter.
        Same methodology as VLA-Pruner's `_timed_call`."""
        _sync_if_cuda(self.config.device)
        start = time.perf_counter()
        with self._stage_output_context():
            result = fn()
        _sync_if_cuda(self.config.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return result, elapsed_ms

    @contextlib.contextmanager
    def _stage_output_context(self) -> Any:
        if not self.config.suppress_stage_output:
            yield
            return
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield


def compare_models(summaries: Sequence[Dict[str, Any]], baseline_model_name: str) -> pd.DataFrame:
    if not summaries:
        return pd.DataFrame()

    baseline = None
    for row in summaries:
        if row["model_name"] == baseline_model_name:
            baseline = row
            break
    if baseline is None:
        baseline = summaries[0]

    rows: List[Dict[str, Any]] = []
    for row in summaries:
        rows.append(
            {
                "model_name": row["model_name"],
                "quant_method": row["quant_method"],
                "model_latency_speedup": _safe_ratio(
                    baseline["model_latency_ms_mean"], row["model_latency_ms_mean"]
                ),
                "end_to_end_latency_speedup": _safe_ratio(
                    baseline["end_to_end_latency_ms_mean"], row["end_to_end_latency_ms_mean"]
                ),
                "vision_latency_speedup": _safe_ratio(baseline["vision_ms_mean"], row["vision_ms_mean"]),
                "llm_latency_speedup": _safe_ratio(baseline["llm_ms_mean"], row["llm_ms_mean"]),
                "action_latency_speedup": _safe_ratio(baseline["action_ms_mean"], row["action_ms_mean"]),
                "peak_memory_reduction_percent": _safe_reduction(
                    baseline.get("peak_cuda_memory_mb"), row.get("peak_cuda_memory_mb")
                ),
                "model_size_reduction_percent": _safe_reduction(
                    baseline.get("model_size_mb"), row.get("model_size_mb")
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_raw_records(
    raw_df: pd.DataFrame,
    adapter: VLAInferenceAdapter,
    model: torch.nn.Module,
    device: torch.device,
    profile_memory: bool,
    compressed_model_size_mb: Optional[float],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "model_name": adapter.model_name(),
        **adapter.quant_info(),
    }

    summary_fields = list(LATENCY_FIELDS) + ["model_latency_ms", "end_to_end_latency_ms"]
    for field in summary_fields:
        if field not in raw_df.columns:
            continue
        values = raw_df[field].to_numpy(dtype=np.float64)
        stats = latency_stats(values)
        for stat_name, stat_value in stats.items():
            summary[f"{field}_{stat_name}"] = stat_value

    memory_info = cuda_memory_info(device) if profile_memory else empty_memory_info()
    summary.update(memory_info)

    model_size_mb = compressed_model_size_mb if compressed_model_size_mb is not None else estimate_model_size_mb(model)
    summary["model_size_mb"] = model_size_mb
    summary["speedup_vs_baseline"] = 1.0
    summary["memory_reduction_vs_baseline"] = 0.0
    summary["model_size_reduction_vs_baseline"] = 0.0
    return summary


def latency_stats(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def format_raw_measurement(record: Dict[str, Any]) -> str:
    return (
        "[raw] "
        f"step={record['step']} "
        f"model={record['model_name']} "
        f"quant={record['quant_method']} "
        f"data={record['data_ms']:.3f}ms "
        f"preprocess={record['preprocess_ms']:.3f}ms "
        f"vision={record['vision_ms']:.3f}ms "
        f"llm={record['llm_ms']:.3f}ms "
        f"action={record['action_ms']:.3f}ms "
        f"model={record['model_latency_ms']:.3f}ms "
        f"e2e={record['end_to_end_latency_ms']:.3f}ms"
    )


def load_openvla_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
    quant_mode: str,
    group_size: int,
    min_linear_weight_numel: int,
    bnb_int8_threshold: float,
    bnb_4bit_compute_dtype: str,
    bnb_4bit_quant_type: str,
    bnb_4bit_use_double_quant: bool,
    fp8_activation_scale: float,
    llm_max_layers: int,
    llm_layer_strategy: str,
    compile_llm: bool,
    compile_mode: str,
) -> Any:
    register_openvla_auto_classes()
    attn_implementation = resolve_attn_implementation(attn_implementation, device)
    print(
        f"[load] loading model from {model_path} dtype={dtype} attn={attn_implementation}",
        flush=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print("[load] model weights loaded", flush=True)

    maybe_load_local_norm_stats(model, model_path)
    patch_local_transformers_config(model)
    original_layers, active_layers = limit_llm_layers(model, llm_max_layers, llm_layer_strategy)
    if active_layers and active_layers < original_layers:
        print(f"[latency] LLM layers limited: {original_layers}->{active_layers} ({llm_layer_strategy})", flush=True)
    normalized_mode = quant_mode.lower().replace("-", "_")
    report = None
    if normalized_mode not in {"none", "fp16", "bf16"}:
        print(f"[direct-quant] applying mode={display_mode_for_profile(normalized_mode)}", flush=True)
        report = quantize_openvla_language_model(
            model,
            DirectQuantConfig(
                mode=normalized_mode,
                group_size=group_size,
                min_linear_weight_numel=min_linear_weight_numel,
                bnb_int8_threshold=bnb_int8_threshold,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                fp8_activation_scale=fp8_activation_scale,
            ),
        )
    else:
        model._direct_quant_report = {"mode": "none", "backend": "none", "target": "none"}

    print(f"[load] moving model to {device}", flush=True)
    model = model.to(device)
    refresh_direct_quant_report_sizes(model)
    print("[load] model ready on device", flush=True)
    if compile_llm:
        compile_language_backbone(model, mode=compile_mode)

    if report is not None:
        updated_report = get_direct_quant_info(model)
        print(
            "[direct-quant] "
            f"mode={display_mode_for_profile(report.mode)} replaced={report.replaced_linear_layers} "
            f"target_size={updated_report['original_target_size_mb']:.2f}->"
            f"{updated_report['quantized_target_size_mb']:.2f} MB",
            flush=True,
        )

    return model


def compile_language_backbone(model: Any, mode: str = "reduce-overhead") -> None:
    if not hasattr(torch, "compile"):
        print("[compile] torch.compile is unavailable; using eager LLM backbone.", flush=True)
        return
    language_model = getattr(model, "language_model", None)
    backbone = get_language_backbone(language_model)
    if backbone is None:
        print("[compile] language backbone is unavailable; using eager LLM backbone.", flush=True)
        return
    try:
        import torch._dynamo as torch_dynamo

        torch_dynamo.config.suppress_errors = True
    except Exception:
        pass
    print(f"[compile] compiling language backbone mode={mode}", flush=True)
    compiled_backbone = torch.compile(backbone, mode=mode, fullgraph=False, dynamic=False)
    if hasattr(language_model, "model"):
        language_model.model = compiled_backbone
    elif hasattr(language_model, "transformer"):
        language_model.transformer = compiled_backbone
    print("[compile] language backbone compile wrapper installed", flush=True)


def refresh_direct_quant_report_sizes(model: Any) -> None:
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


def patch_local_transformers_config(model: Any) -> None:
    """Fill optional config fields expected by locally patched transformers builds."""

    configs = [getattr(model, "config", None)]
    if hasattr(model, "language_model"):
        configs.append(getattr(model.language_model, "config", None))
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    configs.append(text_config)

    for config in configs:
        if config is not None and not hasattr(config, "proportion_attn_var"):
            setattr(config, "proportion_attn_var", None)


def register_openvla_auto_classes() -> None:
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def maybe_load_local_norm_stats(model: Any, model_path: str) -> None:
    stats_path = Path(model_path) / "dataset_statistics.json"
    if stats_path.is_file():
        with stats_path.open("r") as f:
            model.norm_stats = json.load(f)


def load_processor(model_path: str) -> Any:
    register_openvla_auto_classes()
    print(f"[load] loading processor from {model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("[load] processor ready", flush=True)
    return processor


def get_openvla_prompt(instruction: str, model_path: str) -> str:
    if "v01" in str(model_path):
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if torch.is_floating_point(value):
                value = value.to(dtype=dtype)
        moved[key] = value
    return moved


def resolve_dtype(mode: str, default_dtype: str) -> torch.dtype:
    normalized = mode.lower().replace("-", "_")
    if normalized == "fp16":
        return torch.float16
    if normalized == "bf16":
        return torch.bfloat16
    if default_dtype.lower() in {"fp16", "float16"}:
        return torch.float16
    if default_dtype.lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {default_dtype}")


def profile_one_mode(args: argparse.Namespace, mode: str, device: torch.device) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dtype = resolve_dtype(mode, args.dtype)
    attn_implementation = resolve_attn_implementation(args.attn_implementation, device)

    processor = load_processor(args.model_path)
    model = load_openvla_model(
        model_path=args.model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        quant_mode=mode,
        group_size=args.quant_group_size,
        min_linear_weight_numel=args.min_linear_weight_numel,
        bnb_int8_threshold=args.bnb_int8_threshold,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        fp8_activation_scale=args.fp8_activation_scale,
        llm_max_layers=args.llm_max_layers,
        llm_layer_strategy=args.llm_layer_strategy,
        compile_llm=args.compile_llm,
        compile_mode=args.compile_mode,
    )
    adapter = OpenVLAInferenceAdapter(
        model=model,
        processor=processor,
        model_path=args.model_path,
        device=str(device),
        dtype=dtype,
        instruction=args.instruction,
        image_path=args.image_path,
        unnorm_key=args.unnorm_key,
        quant_label=mode,
        fast_action_head=args.fast_action_head,
        last_token_logits=args.last_token_logits,
        max_vision_tokens=args.max_vision_tokens,
        vision_token_strategy=args.vision_token_strategy,
        drop_full_attention_mask=args.drop_full_attention_mask,
    )
    if args.fast_action_head and not adapter.fast_action_head:
        print("[fast-action] action-only lm_head path is unavailable for this model; using full vocab logits.", flush=True)
    if args.last_token_logits and not adapter.last_token_logits:
        print("[last-logits] backbone-only prefill path is unavailable; using full language_model logits.", flush=True)

    profiler = VLALatencyProfiler(
        adapter=adapter,
        timer_config=TimerConfig(
            device=device,
            suppress_stage_output=args.suppress_stage_output,
        ),
        profile_memory=args.profile_memory,
        compressed_model_size_mb=args.compressed_model_size_mb,
        print_raw_measurements=args.print_raw_measurements,
    )
    raw_df, summary = profiler.profile(warmup_steps=args.warmup_steps, repeat_steps=args.repeat_steps)

    del model
    del processor
    del adapter
    gc.collect()
    clear_cuda_memory(device)
    return raw_df, summary


def parse_quant_modes(value: str) -> List[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    if not modes:
        raise ValueError("--quant_modes must contain at least one mode")
    return modes


def normalize_mode_for_profile(mode: str) -> str:
    normalized = mode.lower().replace("-", "_")
    if normalized in {"none", "bf16", "bfloat16", "fp16", "float16"}:
        return "bf16" if normalized == "bfloat16" else "fp16" if normalized == "float16" else normalized
    return DirectQuantConfig(mode=normalized).normalized_mode()


def display_mode_for_profile(mode: str) -> str:
    return FP8_REPORT_LABEL if normalize_mode_for_profile(mode) == "fp8" else mode


def print_latency_definition() -> None:
    print(
        "\nLatency definition (VLA-Pruner methodology, max LLM scope):\n"
        "  data_ms: observation/image/instruction read and handoff to preprocessing.\n"
        "  preprocess_ms: resize/normalization/processor/tokenizer/prompt/batch/CPU-to-GPU transfer.\n"
        "  vision_ms: vision backbone kernel-launch (no sync; GPU exec counted in LLM).\n"
        "  llm_ms:   model_latency_ms - vision_ms - action_ms\n"
        "            (backbone-GPU+projector+embed+prefill+decode+D2H+overhead).\n"
        "  action_ms: numpy action decode only (GPU->CPU transfer counted in LLM).\n"
        "  model_latency_ms = wall-clock from before data-load to action end (data+preprocess+model).\n"
        "  end_to_end_latency_ms = model_latency_ms.\n",
        flush=True,
    )


def update_baseline_relative_fields(summaries: List[Dict[str, Any]], baseline_model_name: str) -> None:
    if not summaries:
        return

    baseline = None
    for row in summaries:
        if row["model_name"] == baseline_model_name:
            baseline = row
            break
    if baseline is None:
        baseline = summaries[0]

    for row in summaries:
        row["speedup_vs_baseline"] = _safe_ratio(
            baseline["model_latency_ms_mean"], row["model_latency_ms_mean"]
        )
        row["memory_reduction_vs_baseline"] = _safe_reduction(
            baseline.get("peak_cuda_memory_mb"), row.get("peak_cuda_memory_mb")
        )
        row["model_size_reduction_vs_baseline"] = _safe_reduction(
            baseline.get("model_size_mb"), row.get("model_size_mb")
        )


def cuda_memory_info(device: torch.device) -> Dict[str, Optional[float]]:
    if device.type != "cuda":
        return empty_memory_info()
    return {
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "allocated_cuda_memory_mb": torch.cuda.memory_allocated(device) / (1024**2),
        "reserved_cuda_memory_mb": torch.cuda.memory_reserved(device) / (1024**2),
    }


def empty_memory_info() -> Dict[str, Optional[float]]:
    return {
        "peak_cuda_memory_mb": None,
        "allocated_cuda_memory_mb": None,
        "reserved_cuda_memory_mb": None,
    }


def clear_cuda_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _safe_ratio(baseline: Optional[float], current: Optional[float], inverse: bool = False) -> Optional[float]:
    if baseline in (None, 0) or current in (None, 0):
        return None
    return current / baseline if inverse else baseline / current


def _safe_reduction(baseline: Optional[float], current: Optional[float]) -> Optional[float]:
    if baseline in (None, 0) or current is None:
        return None
    return (baseline - current) / baseline * 100.0


def _percent_reduction(baseline: float, current: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - current) / baseline * 100.0


def resolve_attn_implementation(attn_implementation: str, device: torch.device) -> str:
    normalized = str(attn_implementation or "auto").lower()
    if normalized == "auto":
        normalized = "flash_attention_2" if device.type == "cuda" else "eager"
    if normalized == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        print("[attn] flash_attention_2 requested but flash_attn is not installed; falling back to sdpa.", flush=True)
        return "sdpa"
    return normalized


def enable_line_buffered_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile OpenVLA latency, memory, size, and direct quantization.")
    parser.add_argument("--model_path", type=str, default="openvla/checkpoints/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--instruction", type=str, default="put the spoon on the towel")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--unnorm_key", type=str, default=None)
    parser.add_argument("--quant_modes", type=str, default="w4a8")
    parser.add_argument("--dtype", type=str, default="bf16", choices=("bf16", "bfloat16", "fp16", "float16"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attn_implementation", type=str, default="auto")
    parser.add_argument("--quant_group_size", type=int, default=128)
    parser.add_argument("--min_linear_weight_numel", type=int, default=0)
    parser.add_argument("--bnb_int8_threshold", type=float, default=0.0)
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        default="auto",
        choices=("auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"),
    )
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=("nf4", "fp4"))
    parser.add_argument("--bnb_4bit_use_double_quant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp8_activation_scale", type=float, default=1.0)
    parser.add_argument("--fast_action_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--last_token_logits", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop_full_attention_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_vision_tokens", type=int, default=0)
    parser.add_argument("--vision_token_strategy", type=str, default="uniform", choices=("uniform", "pool", "first"))
    parser.add_argument("--llm_max_layers", type=int, default=0)
    parser.add_argument("--llm_layer_strategy", type=str, default="first", choices=("first", "uniform", "last"))
    parser.add_argument("--compile_llm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead")
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--repeat_steps", type=int, default=100)
    parser.add_argument("--save_csv", type=str, default="openvla/out/openvla_profile_summary.csv")
    parser.add_argument("--save_json", type=str, default="openvla/out/openvla_profile_raw.json")
    parser.add_argument("--baseline_model_name", type=str, default="")
    parser.add_argument("--profile_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print_raw_measurements", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print_latency_definition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suppress_stage_output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compressed_model_size_mb", type=float, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    enable_line_buffered_output()
    args = build_arg_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable; falling back to CPU.", flush=True)
        args.device = "cpu"
    device = torch.device(args.device)
    if args.print_latency_definition:
        print_latency_definition()

    raw_dfs: List[pd.DataFrame] = []
    summaries: List[Dict[str, Any]] = []
    for mode in parse_quant_modes(args.quant_modes):
        print(f"\n=== Profiling OpenVLA mode: {display_mode_for_profile(mode)} ===", flush=True)
        raw_df, summary = profile_one_mode(args, mode, device)
        raw_dfs.append(raw_df)
        summaries.append(summary)

    baseline_model_name = args.baseline_model_name or summaries[0]["model_name"]
    update_baseline_relative_fields(summaries, baseline_model_name)
    summary_df = pd.DataFrame(summaries)
    comparison_df = compare_models(summaries, baseline_model_name)

    raw_all_df = pd.concat(raw_dfs, ignore_index=True) if raw_dfs else pd.DataFrame()
    save_outputs(args.save_csv, args.save_json, summary_df, comparison_df, raw_all_df, summaries)

    print("\nPer-model latency summary:", flush=True)
    summary_columns = [
        "model_name",
        "quant_method",
        "weight_bits",
        "activation_bits",
        "data_ms_mean",
        "preprocess_ms_mean",
        "vision_ms_mean",
        "llm_ms_mean",
        "action_ms_mean",
        "model_latency_ms_mean",
        "end_to_end_latency_ms_mean",
        "peak_cuda_memory_mb",
        "model_size_mb",
    ]
    print(summary_df[_existing_columns(summary_df, summary_columns)].to_string(index=False), flush=True)

    print("\nComparison vs baseline:", flush=True)
    comparison_columns = [
        "model_name",
        "quant_method",
        "model_latency_speedup",
        "end_to_end_latency_speedup",
        "llm_latency_speedup",
        "peak_memory_reduction_percent",
        "model_size_reduction_percent",
    ]
    print(comparison_df[_existing_columns(comparison_df, comparison_columns)].to_string(index=False), flush=True)


def save_outputs(
    save_csv: str,
    save_json: str,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    summaries: Sequence[Dict[str, Any]],
) -> None:
    if save_csv:
        csv_path = Path(save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(csv_path, index=False)
        comparison_df.to_csv(csv_path.with_name(csv_path.stem + "_comparison.csv"), index=False)
        raw_df.to_csv(csv_path.with_name(csv_path.stem + "_raw.csv"), index=False)

    if save_json:
        json_path = Path(save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summaries": summaries,
            "comparison": comparison_df.to_dict(orient="records"),
            "raw_measurements": raw_df.to_dict(orient="records"),
        }
        with json_path.open("w") as f:
            json.dump(payload, f, indent=2)


def _existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [column for column in columns if column in df.columns]


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
