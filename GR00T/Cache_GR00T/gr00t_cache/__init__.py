# SPDX-FileCopyrightText: Copyright (c) 2025 GR00T-Cache Contributors
# SPDX-License-Identifier: Apache-2.0
"""
GR00T-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching.

This package implements the VLA-Cache method adapted for the NVIDIA GR00T model.
"""

from .config import GR00TCacheConfig, CacheMode
from .token_index_map import TokenIndexMap
from .cache_manager import GR00TCacheManager
from .attention_wrapper import (
    CachedAttentionWrapper,
    apply_cache_to_backbone,
    apply_cache_to_action_head,
    remove_cache_from_model,
)
from .profiling import (
    ProfileResults,
    ProfileTimer,
    profile_gr00t_cache,
    summarize_profile,
    profile_policy_pipeline,
)
from .flops_estimator import (
    estimate_transformer_flops,
    estimate_cache_transformer_flops,
    compute_gr00t_model_flops,
)
from .ablation import (
    ABLATION_PRESETS,
    run_ablation,
    format_ablation_table,
)
from .correctness import (
    check_cache_correctness,
    compute_action_similarity,
    generate_debug_visualization,
)
from .dummy_model import (
    DummyGR00TModel,
    DummyGR00TConfig,
    create_dummy_gr00t_model,
    create_dummy_observation,
)
from .adapter import RealGR00TAdapter

__version__ = "0.1.0"
