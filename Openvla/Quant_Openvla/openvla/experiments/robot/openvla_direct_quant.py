"""Direct, calibration-free quantization utilities for OpenVLA inference.

This module intentionally does not reuse the repository's existing QVLA code.
It replaces Linear layers inside ``model.language_model`` only, leaving the
vision backbone, multimodal projector, embeddings, norms, and LM head in their
loaded dtype.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_SUPPORTED_MODES = {"none", "fp16", "bf16", "w8a16", "int8", "w8a8", "int8_dynamic", "w4a16", "int4"}


@dataclass
class DirectQuantConfig:
    """Configuration for calibration-free OpenVLA LLM quantization."""

    mode: str = "none"
    group_size: int = 128
    min_linear_weight_numel: int = 0
    target: str = "llm"
    skip_module_name_substrings: Tuple[str, ...] = field(
        default_factory=lambda: ("lm_head", "embed", "norm", "rotary")
    )

    def normalized_mode(self) -> str:
        mode = self.mode.lower().replace("-", "_")
        aliases = {
            "int8": "w8a16",
            "int8_dynamic": "w8a8",
            "int4": "w4a16",
            "w8a8_dynamic": "w8a8",
        }
        mode = aliases.get(mode, mode)
        if mode not in _SUPPORTED_MODES:
            raise ValueError(f"Unsupported direct quantization mode: {self.mode}")
        if mode in {"fp16", "bf16"}:
            return "none"
        return mode


@dataclass
class QuantizationReport:
    """Summary emitted after replacing OpenVLA LLM Linear layers."""

    mode: str
    backend: str
    target: str
    replaced_linear_layers: int
    skipped_linear_layers: int
    original_target_size_mb: float
    quantized_target_size_mb: float
    original_model_size_mb: float
    quantized_model_size_mb: float
    skipped_modules: List[str]

    @property
    def target_size_reduction_percent(self) -> float:
        return _reduction_percent(self.original_target_size_mb, self.quantized_target_size_mb)

    @property
    def model_size_reduction_percent(self) -> float:
        return _reduction_percent(self.original_model_size_mb, self.quantized_model_size_mb)

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["target_size_reduction_percent"] = self.target_size_reduction_percent
        data["model_size_reduction_percent"] = self.model_size_reduction_percent
        return data


class WeightOnlyInt8Linear(nn.Module):
    """Per-output-channel INT8 weight-only Linear.

    Forward uses dequantize-then-matmul for broad compatibility. This reliably
    reduces persistent model memory; use ``DynamicInt8Linear`` when CUDA int8
    matmul latency is the priority.
    """

    quant_backend = "direct-w8a16-dequant"
    weight_bits = 8
    activation_bits = 16

    def __init__(
        self,
        qweight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("weight_scale", weight_scale.contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.contiguous())
        else:
            self.bias = None

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear) -> "WeightOnlyInt8Linear":
        weight = linear.weight.detach()
        scale = weight.float().abs().amax(dim=1).clamp(min=1.0e-8) / 127.0
        qweight = torch.round(weight.float() / scale[:, None]).clamp(-127, 127).to(torch.int8)
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        return cls(qweight, scale.to(dtype=weight.dtype, device=weight.device), bias, linear.in_features, linear.out_features)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        compute_dtype = input_tensor.dtype if input_tensor.is_cuda else torch.float32
        weight = self.qweight.to(dtype=compute_dtype) * self.weight_scale.to(dtype=compute_dtype)[:, None]
        bias = self.bias
        if bias is not None:
            bias = bias.to(dtype=compute_dtype)

        linear_input = input_tensor if input_tensor.dtype == compute_dtype else input_tensor.to(compute_dtype)
        output = F.linear(linear_input, weight, bias)
        return output if output.dtype == input_tensor.dtype else output.to(input_tensor.dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class DynamicInt8Linear(nn.Module):
    """Dynamic W8A8 Linear using ``torch._int_mm`` on CUDA when available.

    Activations are quantized per flattened input row at inference time. This
    path is calibration-free and can reduce LLM matmul latency on GPUs with
    efficient int8 tensor-core support. CPU or unsupported CUDA paths fall back
    to dequantized weight-only matmul.
    """

    quant_backend = "direct-w8a8-dynamic-int-mm"
    weight_bits = 8
    activation_bits = 8

    def __init__(
        self,
        qweight_t: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight_t", qweight_t.contiguous())
        self.register_buffer("weight_scale", weight_scale.contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.contiguous())
        else:
            self.bias = None

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear) -> "DynamicInt8Linear":
        weight = linear.weight.detach()
        scale = weight.float().abs().amax(dim=1).clamp(min=1.0e-8) / 127.0
        qweight = torch.round(weight.float() / scale[:, None]).clamp(-127, 127).to(torch.int8)
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        return cls(qweight.t().contiguous(), scale.to(device=weight.device), bias, linear.in_features, linear.out_features)

    def _dequantized_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        compute_dtype = input_tensor.dtype if input_tensor.is_cuda else torch.float32
        weight = self.qweight_t.t().to(dtype=compute_dtype) * self.weight_scale.to(dtype=compute_dtype)[:, None]
        bias = self.bias
        if bias is not None:
            bias = bias.to(dtype=compute_dtype)
        linear_input = input_tensor if input_tensor.dtype == compute_dtype else input_tensor.to(compute_dtype)
        output = F.linear(linear_input, weight, bias)
        return output if output.dtype == input_tensor.dtype else output.to(input_tensor.dtype)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if not (input_tensor.is_cuda and hasattr(torch, "_int_mm")):
            return self._dequantized_forward(input_tensor)

        original_shape = input_tensor.shape[:-1]
        x_2d = input_tensor.reshape(-1, self.in_features)
        x_float = x_2d.float()
        x_scale = x_float.abs().amax(dim=1, keepdim=True).clamp(min=1.0e-8) / 127.0
        x_q = torch.round(x_float / x_scale).clamp(-127, 127).to(torch.int8).contiguous()

        # torch._int_mm prefers tensor-core-friendly row counts; decode usually has one row.
        rows = x_q.shape[0]
        padded_rows = _ceil_to_multiple(max(rows, 32), 32)
        if padded_rows != rows:
            pad_rows = padded_rows - rows
            x_q_for_mm = F.pad(x_q, (0, 0, 0, pad_rows))
            x_scale_for_mm = F.pad(x_scale, (0, 0, 0, pad_rows))
        else:
            x_q_for_mm = x_q
            x_scale_for_mm = x_scale

        out_i32 = torch._int_mm(x_q_for_mm, self.qweight_t)
        out = out_i32.float() * x_scale_for_mm * self.weight_scale.float()[None, :]
        out = out[:rows]
        if self.bias is not None:
            out = out + self.bias.float()[None, :]
        return out.reshape(*original_shape, self.out_features).to(dtype=input_tensor.dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class WeightOnlyInt4Linear(nn.Module):
    """Group-wise signed INT4 weight-only Linear with uint8 packing."""

    quant_backend = "direct-w4a16-dequant"
    weight_bits = 4
    activation_bits = 16

    def __init__(
        self,
        packed_qweight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor],
        in_features: int,
        out_features: int,
        group_size: int,
        padded_in_features: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.padded_in_features = padded_in_features
        self.register_buffer("packed_qweight", packed_qweight.contiguous())
        self.register_buffer("weight_scale", weight_scale.contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.contiguous())
        else:
            self.bias = None

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear, group_size: int) -> "WeightOnlyInt4Linear":
        if group_size <= 0:
            raise ValueError("group_size must be a positive integer for W4A16 quantization")

        weight = linear.weight.detach()
        padded_in_features = _ceil_to_multiple(linear.in_features, group_size)
        pad_width = padded_in_features - linear.in_features
        padded = F.pad(weight.float(), (0, pad_width)) if pad_width else weight.float()
        grouped = padded.reshape(linear.out_features, padded_in_features // group_size, group_size)
        scale = grouped.abs().amax(dim=-1).clamp(min=1.0e-8) / 7.0
        qweight = torch.round(grouped / scale.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
        packed = _pack_int4(qweight.reshape(linear.out_features, padded_in_features))
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        return cls(
            packed,
            scale.to(dtype=weight.dtype, device=weight.device),
            bias,
            linear.in_features,
            linear.out_features,
            group_size,
            padded_in_features,
        )

    def _dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        qweight = _unpack_int4(self.packed_qweight, self.padded_in_features)
        qweight = qweight.reshape(self.out_features, self.padded_in_features // self.group_size, self.group_size)
        weight = qweight.to(dtype=dtype) * self.weight_scale.to(dtype=dtype).unsqueeze(-1)
        return weight.reshape(self.out_features, self.padded_in_features)[:, : self.in_features].contiguous()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        compute_dtype = input_tensor.dtype if input_tensor.is_cuda else torch.float32
        weight = self._dequantize_weight(compute_dtype)
        bias = self.bias
        if bias is not None:
            bias = bias.to(dtype=compute_dtype)
        linear_input = input_tensor if input_tensor.dtype == compute_dtype else input_tensor.to(compute_dtype)
        output = F.linear(linear_input, weight, bias)
        return output if output.dtype == input_tensor.dtype else output.to(input_tensor.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
            f"group_size={self.group_size}"
        )


def quantize_openvla_language_model(model: nn.Module, config: DirectQuantConfig) -> QuantizationReport:
    """Replace Linear layers in ``model.language_model`` with direct quantized modules."""

    mode = config.normalized_mode()
    if mode == "none":
        model_size_mb = estimate_model_size_mb(model)
        report = QuantizationReport(
            mode="none",
            backend="none",
            target=config.target,
            replaced_linear_layers=0,
            skipped_linear_layers=0,
            original_target_size_mb=model_size_mb,
            quantized_target_size_mb=model_size_mb,
            original_model_size_mb=model_size_mb,
            quantized_model_size_mb=model_size_mb,
            skipped_modules=[],
        )
        model._direct_quant_report = report.to_dict()
        return report

    if config.target != "llm":
        raise ValueError("OpenVLA direct quantization currently supports target='llm' only")
    if not hasattr(model, "language_model"):
        raise AttributeError("Expected OpenVLA model to expose a `language_model` module")

    target_module = model.language_model
    original_target_size_mb = estimate_model_size_mb(target_module)
    original_model_size_mb = estimate_model_size_mb(model)

    replaced, skipped, skipped_modules = _replace_linear_children(target_module, "", mode, config)
    quantized_target_size_mb = estimate_model_size_mb(target_module)
    quantized_model_size_mb = estimate_model_size_mb(model)

    backend = _backend_name_for_mode(mode)
    report = QuantizationReport(
        mode=mode,
        backend=backend,
        target=config.target,
        replaced_linear_layers=replaced,
        skipped_linear_layers=skipped,
        original_target_size_mb=original_target_size_mb,
        quantized_target_size_mb=quantized_target_size_mb,
        original_model_size_mb=original_model_size_mb,
        quantized_model_size_mb=quantized_model_size_mb,
        skipped_modules=skipped_modules,
    )
    model._direct_quant_report = report.to_dict()
    return report


def estimate_model_size_mb(model: nn.Module) -> float:
    """Estimate model size from current parameters and buffers."""

    seen_data_ptrs = set()
    total_bytes = 0
    for tensor in _iter_parameters_and_buffers(model):
        data_ptr = tensor.data_ptr()
        if data_ptr in seen_data_ptrs:
            continue
        seen_data_ptrs.add(data_ptr)
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024**2)


def get_direct_quant_info(model: nn.Module) -> Dict[str, object]:
    """Return quantization metadata attached by ``quantize_openvla_language_model``."""

    report = getattr(model, "_direct_quant_report", None)
    if isinstance(report, dict):
        return report
    return {
        "mode": "none",
        "backend": "none",
        "target": "none",
        "replaced_linear_layers": 0,
    }


def _replace_linear_children(
    module: nn.Module,
    prefix: str,
    mode: str,
    config: DirectQuantConfig,
) -> Tuple[int, int, List[str]]:
    replaced = 0
    skipped = 0
    skipped_modules: List[str] = []

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear):
            if _should_skip_linear(full_name, child, config):
                skipped += 1
                skipped_modules.append(full_name)
                continue
            setattr(module, child_name, _make_quantized_linear(child, mode, config))
            replaced += 1
            continue

        child_replaced, child_skipped, child_skipped_modules = _replace_linear_children(child, full_name, mode, config)
        replaced += child_replaced
        skipped += child_skipped
        skipped_modules.extend(child_skipped_modules)

    return replaced, skipped, skipped_modules


def _should_skip_linear(name: str, linear: nn.Linear, config: DirectQuantConfig) -> bool:
    lower_name = name.lower()
    if any(pattern in lower_name for pattern in config.skip_module_name_substrings):
        return True
    if linear.weight.numel() < config.min_linear_weight_numel:
        return True
    return False


def _make_quantized_linear(linear: nn.Linear, mode: str, config: DirectQuantConfig) -> nn.Module:
    if mode == "w8a16":
        return WeightOnlyInt8Linear.from_linear(linear)
    if mode == "w8a8":
        return DynamicInt8Linear.from_linear(linear)
    if mode == "w4a16":
        return WeightOnlyInt4Linear.from_linear(linear, group_size=config.group_size)
    raise ValueError(f"Unsupported normalized quantization mode: {mode}")


def _backend_name_for_mode(mode: str) -> str:
    if mode == "w8a16":
        return WeightOnlyInt8Linear.quant_backend
    if mode == "w8a8":
        return DynamicInt8Linear.quant_backend
    if mode == "w4a16":
        return WeightOnlyInt4Linear.quant_backend
    return "none"


def _iter_parameters_and_buffers(module: nn.Module) -> Iterable[torch.Tensor]:
    yield from module.parameters(recurse=True)
    yield from module.buffers(recurse=True)


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _pack_int4(qweight: torch.Tensor) -> torch.Tensor:
    if qweight.shape[-1] % 2:
        qweight = F.pad(qweight, (0, 1))
    q_unsigned = torch.where(qweight < 0, qweight + 16, qweight).to(torch.uint8)
    low = q_unsigned[..., 0::2]
    high = q_unsigned[..., 1::2] << 4
    return low | high


def _unpack_int4(packed_qweight: torch.Tensor, total_values: int) -> torch.Tensor:
    low = packed_qweight & 0x0F
    high = (packed_qweight >> 4) & 0x0F
    unpacked = torch.stack((low, high), dim=-1).reshape(*packed_qweight.shape[:-1], packed_qweight.shape[-1] * 2)
    unpacked = unpacked[..., :total_values].to(torch.int8)
    return torch.where(unpacked >= 8, unpacked - 16, unpacked)


def _reduction_percent(baseline: float, current: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - current) / baseline * 100.0
