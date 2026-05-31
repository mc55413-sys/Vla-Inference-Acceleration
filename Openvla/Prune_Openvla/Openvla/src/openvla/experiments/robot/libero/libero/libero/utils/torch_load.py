"""Compatibility helpers for loading trusted LIBERO torch files."""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch


def _numpy_safe_globals() -> List[Any]:
    from numpy.core.multiarray import _reconstruct, scalar

    dtype_classes = {
        type(np.dtype(dtype))
        for dtype in (
            np.bool_,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.float16,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        )
    }
    return [_reconstruct, scalar, np.ndarray, np.dtype, *dtype_classes]


def torch_load_numpy(path: str) -> Any:
    """Load trusted LIBERO data files that contain numpy arrays/scalars."""
    with torch.serialization.safe_globals(_numpy_safe_globals()):
        return torch.load(path, weights_only=True)
