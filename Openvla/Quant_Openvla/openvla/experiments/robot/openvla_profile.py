"""VLA quantization latency, memory, and size profiling CLI.

Timing methodology follows VLA-Pruner:
  - All GPU stages use wall-clock timing: torch.cuda.synchronize() + time.perf_counter()
  - vision_ms = vision_backbone_ms + projector_ms (direct measurement)
  - action_ms = action_decode_ms (direct measurement)
  - model_latency_ms = total wall-clock from vision start to action end
  - llm_ms = max(0.0, model_latency_ms - vision_ms - action_ms)  (subtraction, captures LLM + inter-module overhead)

Latency breakdown:
  End-to-End Latency = Data + Preprocess + Model
  Model Latency      = Vision + LLM + Action
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
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


SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
LATENCY_FIELDS = ("data_ms", "preprocess_ms", "vision_ms", "llm_ms", "action_ms")
SUMMARY_STATS = ("mean", "std", "p50", "p90", "p95", "min", "max")


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
    ) -> None:
        super().__init__(model=model, processor=processor, device=device)
        self.model_path = model_path
        self.dtype = dtype
        self.instruction = instruction
        self.image_path = image_path
        self.unnorm_key = unnorm_key
        self.quant_label = quant_label
        self._synthetic_image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).convert("RGB")

    def model_name(self) -> str:
        base = Path(str(self.model_path)).name or str(self.model_path)
        return f"{base}:{self.quant_label}"

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

    def preprocess(self, raw_sample: OpenVLASample) -> Dict[str, Any]:
        prompt = get_openvla_prompt(raw_sample.instruction, self.model_path)
        batch = self.processor(prompt, raw_sample.image)
        batch = move_batch_to_device(batch, self.device, self.dtype)
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
        batch["unnorm_key"] = raw_sample.unnorm_key
        return batch

    def vision_forward(self, batch: Dict[str, Any]) -> VisionOutputs:
        pixel_values = batch["pixel_values"]

        patch_features = self.model.vision_backbone(pixel_values)
        projected_patch_embeddings = self.model.projector(patch_features)

        return VisionOutputs(projected_patch_embeddings=projected_patch_embeddings)

    def _build_multimodal_inputs(
        self,
        vision_outputs: VisionOutputs,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        input_embeddings = self.model.get_input_embeddings()(input_ids)
        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], vision_outputs.projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
        )

        multimodal_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.ones(
                (
                    vision_outputs.projected_patch_embeddings.shape[0],
                    vision_outputs.projected_patch_embeddings.shape[1],
                ),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            multimodal_attention_mask = torch.cat(
                [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
            )
        return multimodal_embeddings, multimodal_attention_mask

    def llm_forward(self, vision_outputs: VisionOutputs, batch: Dict[str, Any]) -> LLMOutputs:
        action_dim = self.model.get_action_dim(batch.get("unnorm_key"))
        generated_tokens: List[torch.Tensor] = []
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_inputs(vision_outputs, batch)

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
        predicted_action_token_ids = llm_outputs.generated_action_token_ids[0].detach().cpu().numpy()
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

    def derive_token_counts(self) -> Tuple[int, int]:
        raw_sample = self.load_sample(0)
        batch = self.preprocess(raw_sample)
        text_tokens = int(batch["input_ids"].shape[1])
        vision_tokens = int(getattr(self.model.vision_backbone.featurizer.patch_embed, "num_patches", 0))
        if vision_tokens <= 0:
            vision_outputs = self.vision_forward(batch)
            vision_tokens = int(vision_outputs.projected_patch_embeddings.shape[1])
        _sync_if_cuda(self.device)
        return text_tokens, vision_tokens


class VLALatencyProfiler:
    def __init__(
        self,
        adapter: VLAInferenceAdapter,
        timer_config: TimerConfig,
        theoretical_tflops: float,
        profile_memory: bool = True,
        compressed_model_size_mb: Optional[float] = None,
        print_raw_measurements: bool = True,
    ) -> None:
        self.adapter = adapter
        self.timer_config = timer_config
        self.theoretical_tflops = theoretical_tflops
        self.profile_memory = profile_memory
        self.compressed_model_size_mb = compressed_model_size_mb
        self.print_raw_measurements = print_raw_measurements

    @torch.inference_mode()
    def profile(self, warmup_steps: int, repeat_steps: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        self.adapter.model.eval()
        clear_cuda_memory(self.timer_config.device)

        for idx in range(warmup_steps):
            self._run_one(idx, collect_timing=False)

        clear_cuda_memory(self.timer_config.device)
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
            theoretical_tflops=self.theoretical_tflops,
            device=self.timer_config.device,
            profile_memory=self.profile_memory,
            compressed_model_size_mb=self.compressed_model_size_mb,
        )
        return raw_df, summary

    def _run_one(self, idx: int, collect_timing: bool) -> Dict[str, Any]:
        timer = StageTimer(self.timer_config)

        # Data & Preprocess (CPU-heavy stages, sync after preprocess for GPU transfer)
        raw_sample, data_ms = timer.time_cpu(lambda: self.adapter.load_sample(idx), sync_after=False)
        batch, preprocess_ms = timer.time_cpu(lambda: self.adapter.preprocess(raw_sample), sync_after=True)

        # ── VLA-Pruner style timing ──
        # model_latency_ms: total wall-clock from vision start to action end
        # vision_ms:        direct wall-clock measurement (backbone + projector)
        # action_ms:        direct wall-clock measurement (action decode)
        # llm_ms:           subtraction: model_latency_ms - vision_ms - action_ms

        # Total model forward start
        _sync_if_cuda(self.timer_config.device)
        t_model_start = time.perf_counter()

        # Vision (backbone + projector)
        t_vision_start = time.perf_counter()
        vision_outputs = self.adapter.vision_forward(batch)
        _sync_if_cuda(self.timer_config.device)
        vision_ms = (time.perf_counter() - t_vision_start) * 1000.0

        # LLM (prefill + decode, no separate sync before — already synced from vision)
        llm_outputs = self.adapter.llm_forward(vision_outputs, batch)
        _sync_if_cuda(self.timer_config.device)
        t_after_llm = time.perf_counter()

        # Action decode
        action = self.adapter.action_decode(llm_outputs, batch)
        _sync_if_cuda(self.timer_config.device)
        action_ms = (time.perf_counter() - t_after_llm) * 1000.0

        # Total model latency
        model_latency_ms = (time.perf_counter() - t_model_start) * 1000.0

        # VLA-Pruner subtraction formula
        llm_ms = max(0.0, model_latency_ms - vision_ms - action_ms)

        if not collect_timing:
            return {}

        # VLA-Pruner formula: model_latency_ms is directly measured wall-clock total;
        # llm_ms is derived by subtraction. Keep both in the record.
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


def estimate_transformer_tflops(T: int, K: int, N: int, M: int, rho: float, d: int, ffn_dim: int) -> Dict[str, float]:
    mu = N + M
    mu_tilde = N + rho * M

    def c(n: float) -> float:
        return 4 * n * d**2 + 2 * n**2 * d + 2 * n * d * ffn_dim

    flops_full = T * c(mu)
    flops_prune = (K - 1) * c(mu) + (T - K + 1) * c(mu_tilde)
    flops_ratio = flops_prune / flops_full if flops_full else 0.0
    return {
        "mu": mu,
        "mu_tilde": mu_tilde,
        "flops_full": flops_full,
        "flops_prune": flops_prune,
        "tflops_full": flops_full / 1.0e12,
        "tflops_prune": flops_prune / 1.0e12,
        "flops_ratio": flops_ratio,
        "flops_reduction_percent": (1.0 - flops_ratio) * 100.0,
    }


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
                "effective_tflops_per_second": row["effective_tflops_per_second"],
                "effective_tflops_per_second_improvement": _safe_ratio(
                    baseline["effective_tflops_per_second"], row["effective_tflops_per_second"], inverse=True
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_raw_records(
    raw_df: pd.DataFrame,
    adapter: VLAInferenceAdapter,
    model: torch.nn.Module,
    theoretical_tflops: float,
    device: torch.device,
    profile_memory: bool,
    compressed_model_size_mb: Optional[float],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "model_name": adapter.model_name(),
        **adapter.quant_info(),
    }

    for field in LATENCY_FIELDS + ("model_latency_ms", "end_to_end_latency_ms"):
        values = raw_df[field].to_numpy(dtype=np.float64)
        stats = latency_stats(values)
        for stat_name, stat_value in stats.items():
            summary[f"{field}_{stat_name}"] = stat_value

    memory_info = cuda_memory_info(device) if profile_memory else empty_memory_info()
    summary.update(memory_info)

    model_size_mb = compressed_model_size_mb if compressed_model_size_mb is not None else estimate_model_size_mb(model)
    model_latency_ms = summary["model_latency_ms_mean"]
    summary["model_size_mb"] = model_size_mb
    summary["theoretical_tflops"] = theoretical_tflops
    summary["effective_tflops_per_second"] = theoretical_tflops / (model_latency_ms / 1000.0) if model_latency_ms else 0.0
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
) -> Any:
    register_openvla_auto_classes()
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    maybe_load_local_norm_stats(model, model_path)
    patch_local_transformers_config(model)
    normalized_mode = quant_mode.lower().replace("-", "_")
    if normalized_mode not in {"none", "fp16", "bf16"}:
        report = quantize_openvla_language_model(
            model,
            DirectQuantConfig(
                mode=normalized_mode,
                group_size=group_size,
                min_linear_weight_numel=min_linear_weight_numel,
            ),
        )
        print(
            "[direct-quant] "
            f"mode={report.mode} backend={report.backend} replaced={report.replaced_linear_layers} "
            f"target_size={report.original_target_size_mb:.2f}->{report.quantized_target_size_mb:.2f} MB"
        )
    else:
        model._direct_quant_report = {"mode": "none", "backend": "none", "target": "none"}

    return model.to(device)


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
    return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


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


def derive_transformer_shape(
    model: Any,
    adapter: OpenVLAInferenceAdapter,
    args: argparse.Namespace,
) -> Dict[str, int]:
    text_tokens, vision_tokens = adapter.derive_token_counts()
    text_config = model.config.text_config
    return {
        "T": args.T or int(getattr(text_config, "num_hidden_layers")),
        "K": args.K or 1,
        "N": args.N or text_tokens,
        "M": args.M or vision_tokens,
        "d": args.d or int(getattr(text_config, "hidden_size")),
        "ffn_dim": args.ffn_dim or int(getattr(text_config, "intermediate_size")),
    }


def profile_one_mode(args: argparse.Namespace, mode: str, device: torch.device) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dtype = resolve_dtype(mode, args.dtype)
    attn_implementation = args.attn_implementation
    if attn_implementation == "auto":
        attn_implementation = "eager" if device.type == "cuda" else "eager"

    processor = load_processor(args.model_path)
    model = load_openvla_model(
        model_path=args.model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        quant_mode=mode,
        group_size=args.quant_group_size,
        min_linear_weight_numel=args.min_linear_weight_numel,
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
    )

    shape = derive_transformer_shape(model, adapter, args)
    tflops_info = estimate_transformer_tflops(
        T=shape["T"],
        K=shape["K"],
        N=shape["N"],
        M=shape["M"],
        rho=args.rho,
        d=shape["d"],
        ffn_dim=shape["ffn_dim"],
    )
    print(
        "[tflops] "
        f"mode={mode} T={shape['T']} K={shape['K']} N={shape['N']} M={shape['M']} "
        f"rho={args.rho} d={shape['d']} ffn_dim={shape['ffn_dim']} "
        f"tflops={tflops_info['tflops_prune']:.6f}"
    )

    profiler = VLALatencyProfiler(
        adapter=adapter,
        timer_config=TimerConfig(
            device=device,
            suppress_stage_output=args.suppress_stage_output,
        ),
        theoretical_tflops=tflops_info["tflops_prune"],
        profile_memory=args.profile_memory,
        compressed_model_size_mb=args.compressed_model_size_mb,
        print_raw_measurements=args.print_raw_measurements,
    )
    raw_df, summary = profiler.profile(warmup_steps=args.warmup_steps, repeat_steps=args.repeat_steps)
    summary.update({f"tflops_{key}": value for key, value in tflops_info.items()})

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


def print_latency_definition() -> None:
    print(
        "\nLatency definition (VLA-Pruner methodology):\n"
        "  data_ms: observation/image/instruction read and handoff to preprocessing.\n"
        "  preprocess_ms: resize/normalization/processor/tokenizer/prompt/batch/CPU-to-GPU transfer.\n"
        "  vision_ms: vision backbone + multimodal projector (wall-clock, sync before/after).\n"
        "  llm_ms:   max(0, model_latency_ms - vision_ms - action_ms) — subtraction, captures LLM + overhead.\n"
        "  action_ms: action-token decode, detokenization, unnormalization (wall-clock, sync before/after).\n"
        "  model_latency_ms = directly measured wall-clock total (vision start → action end).\n"
        "  end_to_end_latency_ms = data_ms + preprocess_ms + model_latency_ms.\n",
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile OpenVLA latency, memory, size, and direct quantization.")
    parser.add_argument("--model_path", type=str, default="openvla/checkpoints/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--instruction", type=str, default="put the spoon on the towel")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--unnorm_key", type=str, default=None)
    parser.add_argument("--quant_modes", type=str, default="w8a8,w8a16,w4a16")
    parser.add_argument("--dtype", type=str, default="bf16", choices=("bf16", "bfloat16", "fp16", "float16"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--quant_group_size", type=int, default=128)
    parser.add_argument("--min_linear_weight_numel", type=int, default=0)
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

    parser.add_argument("--T", type=int, default=0)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--N", type=int, default=0)
    parser.add_argument("--M", type=int, default=0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--d", type=int, default=0)
    parser.add_argument("--ffn_dim", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    if args.print_latency_definition:
        print_latency_definition()

    raw_dfs: List[pd.DataFrame] = []
    summaries: List[Dict[str, Any]] = []
    for mode in parse_quant_modes(args.quant_modes):
        print(f"\n=== Profiling OpenVLA mode: {mode} ===")
        raw_df, summary = profile_one_mode(args, mode, device)
        raw_dfs.append(raw_df)
        summaries.append(summary)

    baseline_model_name = args.baseline_model_name or summaries[0]["model_name"]
    update_baseline_relative_fields(summaries, baseline_model_name)
    summary_df = pd.DataFrame(summaries)
    comparison_df = compare_models(summaries, baseline_model_name)

    raw_all_df = pd.concat(raw_dfs, ignore_index=True) if raw_dfs else pd.DataFrame()
    save_outputs(args.save_csv, args.save_json, summary_df, comparison_df, raw_all_df, summaries)

    print("\nPer-model latency summary:")
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
        "theoretical_tflops",
        "effective_tflops_per_second",
    ]
    print(summary_df[_existing_columns(summary_df, summary_columns)].to_string(index=False))

    print("\nComparison vs baseline:")
    comparison_columns = [
        "model_name",
        "quant_method",
        "model_latency_speedup",
        "end_to_end_latency_speedup",
        "llm_latency_speedup",
        "peak_memory_reduction_percent",
        "model_size_reduction_percent",
        "effective_tflops_per_second",
    ]
    print(comparison_df[_existing_columns(comparison_df, comparison_columns)].to_string(index=False))


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
