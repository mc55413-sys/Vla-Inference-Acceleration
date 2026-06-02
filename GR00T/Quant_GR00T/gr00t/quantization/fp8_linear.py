"""
Selective FP8 Linear wrapper for Blackwell GPUs.

Only converts large matmuls to FP8 via torch._scaled_mm.
Small matmuls stay in BF16 — the FP8 conversion overhead dominates
for small M*K*N products on RTX 5090.

Benchmark (RTX 5090, batch=1):
  - 1x3584x14336 (gate_proj, 51M): FP8 52us vs BF16 70us → 1.35x
  - 1x3584x4096  (smaller FFN, 14.7M): FP8 42us vs BF16 30us → 0.71x (slower!)
  - 1x2048x8192  (DiT FFN, 16.8M):   FP8 32us vs BF16 12us → 0.38x (much slower!)

Threshold: FP8 only when out_features * in_features >= MIN_FP8_OPS (default 30M).
On GR00T N1.5 + Qwen3 with moderate intermediate_size, this may convert
zero additional layers. torch.compile provides the primary speedup (~1.1-1.3x).
"""

import torch
import torch.nn as nn
from typing import Optional


# Minimum ops to use FP8. Set conservatively — on RTX 5090,
# FP8 only wins for very large matmuls (>30M ops).
_MIN_FP8_OPS = 30_000_000
_FP8_MAX = 448.0  # float8_e4m3fn max representable value

# Layer name patterns to ALWAYS skip (need full precision)
_SKIP_PATTERNS = ("lm_head", "embed_tokens", "embed", "head", "classifier")


class FP8DynamicLinear(nn.Module):
    """Replaces nn.Linear with FP8-accelerated version for very large matmuls.

    Uses @torch.compiler.disable on the FP8 path to prevent torch.compile
    from trying to trace through torch._scaled_mm (which has no autograd
    derivative). The surrounding BF16 operations still benefit from compilation.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.use_fp8 = (linear.out_features * linear.in_features) >= _MIN_FP8_OPS

        if self.use_fp8:
            w_bf16 = linear.weight.data.detach().to(dtype=torch.bfloat16)
            w_t = w_bf16.t()  # (in, out) with stride=(1, in) → column-major
            self.register_buffer("_w_fp8", w_t.to(torch.float8_e4m3fn))
            w_scales = (
                w_t.abs()
                .amax(dim=0, keepdim=True)
                .clamp(min=1e-8)
                .float()
                / _FP8_MAX
            ).contiguous()
            self.register_buffer("_w_scales", w_scales)
            self._w_bf16 = None
        else:
            self._w_fp8 = None
            self._w_scales = None
            self._w_bf16 = linear.weight.data.detach().to(dtype=torch.bfloat16)

        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.detach().to(dtype=torch.bfloat16))
        else:
            self.bias = None

    @torch.compiler.disable
    def _fp8_matmul(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Isolated FP8 matmul — disabled from torch.compile trace."""
        x_bf16 = x_flat.to(dtype=torch.bfloat16)
        x_scales = (
            x_bf16.abs()
            .amax(dim=-1, keepdim=True)
            .clamp(min=1e-8)
            .float()
            / _FP8_MAX
        ).contiguous()
        x_fp8 = x_bf16.to(torch.float8_e4m3fn).contiguous()
        return torch._scaled_mm(
            x_fp8, self._w_fp8, x_scales, self._w_scales,
            out_dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fp8 and x.shape[-1] == self.in_features:
            # Flatten 3D+ inputs to 2D for _scaled_mm
            orig_shape = x.shape
            if x.dim() > 2:
                x_flat = x.reshape(-1, self.in_features)
            else:
                x_flat = x

            result = self._fp8_matmul(x_flat)

            if x.dim() > 2:
                result = result.reshape(*orig_shape[:-1], self.out_features)
        else:
            x_bf16 = x.to(dtype=torch.bfloat16)
            if self.use_fp8:
                w_bf16 = self._w_fp8.to(dtype=torch.bfloat16) * self._w_scales.to(
                    dtype=torch.bfloat16
                )
                result = torch.nn.functional.linear(x_bf16, w_bf16.t(), None)
            else:
                result = torch.nn.functional.linear(x_bf16, self._w_bf16, None)

        if self.bias is not None:
            result = result + self.bias.to(dtype=result.dtype)
        return result.to(dtype=x.dtype)


def convert_to_fp8_linear(model: nn.Module, verbose: bool = True) -> int:
    """Selectively replace very large nn.Linear layers with FP8DynamicLinear.

    Only converts layers where out_features * in_features >= MIN_FP8_OPS (30M).
    Skips lm_head and similar output projection layers.

    Returns:
        Number of layers converted.
    """
    converted = 0
    skipped_small = 0
    skipped_pattern = 0

    def _convert(parent: nn.Module, prefix: str = ""):
        nonlocal converted, skipped_small, skipped_pattern
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                ops = child.out_features * child.in_features

                # Skip output heads
                if any(p in child_name.lower() for p in _SKIP_PATTERNS):
                    skipped_pattern += 1
                    if verbose:
                        print(f"  [SKIP] {full_name}: {child.in_features}x{child.out_features}"
                              f" ({ops/1e6:.1f}M ops) -> kept BF16 (output head)")
                elif ops >= _MIN_FP8_OPS:
                    fp8_linear = FP8DynamicLinear(child)
                    setattr(parent, child_name, fp8_linear)
                    converted += 1
                    if verbose:
                        print(f"  [FP8] {full_name}: {child.in_features}x{child.out_features}"
                              f" ({ops/1e6:.1f}M ops) -> FP8")
                else:
                    skipped_small += 1
                    if verbose and ops > 1_000_000:
                        print(f"  [BF16] {full_name}: {child.in_features}x{child.out_features}"
                              f" ({ops/1e6:.1f}M ops) -> keeping BF16")
            else:
                _convert(child, full_name)

    _convert(model)
    if verbose:
        print(
            f"[FP8] Converted {converted} layers, "
            f"skipped {skipped_pattern} output heads, "
            f"kept {skipped_small} smaller layers in BF16"
        )
    return converted
