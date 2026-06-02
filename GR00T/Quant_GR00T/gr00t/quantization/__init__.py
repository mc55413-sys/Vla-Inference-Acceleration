"""GR00T DuQuant W4A8 quantization module."""

from .duquant_layers import (
    DuQuantConfig,
    DuQuantLinear,
    enable_duquant_if_configured,
    select_targets,
    wrap_duquant,
)
from .fp8_linear import (
    FP8DynamicLinear,
    convert_to_fp8_linear,
)

__all__ = [
    "DuQuantConfig",
    "DuQuantLinear",
    "FP8DynamicLinear",
    "convert_to_fp8_linear",
    "enable_duquant_if_configured",
    "select_targets",
    "wrap_duquant",
]
