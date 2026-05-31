# SPDX-FileCopyrightText: Copyright (c) 2025 GR00T-Cache Contributors
# SPDX-License-Identifier: Apache-2.0
"""Profiling tools for GR00T-Cache latency and memory measurement."""

from __future__ import annotations

import csv
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

from .utils import cuda_sync, elapsed_ms, gpu_timer_end, gpu_timer_start, summarize_statistics


@dataclass
class ProfileResults:
    """Container for profiling results."""

    # Per-step raw measurements
    data_ms: list[float] = field(default_factory=list)
    preprocess_ms: list[float] = field(default_factory=list)
    vision_ms: list[float] = field(default_factory=list)
    backbone_ms: list[float] = field(default_factory=list)
    action_head_ms: list[float] = field(default_factory=list)
    action_postprocess_ms: list[float] = field(default_factory=list)
    model_latency_ms: list[float] = field(default_factory=list)
    end_to_end_ms: list[float] = field(default_factory=list)

    # CUDA memory
    peak_cuda_memory_mb: list[float] = field(default_factory=list)
    allocated_cuda_memory_mb: list[float] = field(default_factory=list)
    reserved_cuda_memory_mb: list[float] = field(default_factory=list)

    # Cache statistics
    reused_token_ratio: list[float] = field(default_factory=list)
    evicted_task_token_ratio: list[float] = field(default_factory=list)
    cache_hit_rate: list[float] = field(default_factory=list)
    cache_reset_count: list[int] = field(default_factory=list)

    # Action quality (if baseline available)
    action_l2_diff: list[float] = field(default_factory=list)
    action_cosine_sim: list[float] = field(default_factory=list)
    max_abs_action_diff: list[float] = field(default_factory=list)

    # Metadata
    config_snapshot: dict = field(default_factory=dict)
    model_name: str = "gr00t"
    cache_mode: str = "none"

    def record_timing(self, name: str, value_ms: float) -> None:
        """Record a timing value."""
        if hasattr(self, f"{name}_ms"):
            getattr(self, f"{name}_ms").append(value_ms)
        elif hasattr(self, name):
            getattr(self, name).append(value_ms)

    def record_cuda_memory(self) -> None:
        """Record current CUDA memory state."""
        if torch.cuda.is_available():
            self.peak_cuda_memory_mb.append(
                torch.cuda.max_memory_allocated() / (1024 * 1024)
            )
            self.allocated_cuda_memory_mb.append(
                torch.cuda.memory_allocated() / (1024 * 1024)
            )
            self.reserved_cuda_memory_mb.append(
                torch.cuda.memory_reserved() / (1024 * 1024)
            )

    def summarize(self) -> dict[str, dict[str, float]]:
        """Compute summary statistics for all measured quantities."""
        fields = [
            "data_ms", "preprocess_ms", "vision_ms", "backbone_ms",
            "action_head_ms", "action_postprocess_ms", "model_latency_ms",
            "end_to_end_ms",
            "peak_cuda_memory_mb", "allocated_cuda_memory_mb",
            "reserved_cuda_memory_mb",
            "reused_token_ratio", "evicted_task_token_ratio",
            "cache_hit_rate",
            "action_l2_diff", "action_cosine_sim", "max_abs_action_diff",
        ]
        summary = {}
        for field_name in fields:
            values = getattr(self, field_name, [])
            if values:
                summary[field_name] = summarize_statistics(values)
        return summary

    def to_csv_rows(self) -> list[dict]:
        """Convert to list of dicts for CSV export."""
        n = max(
            len(self.end_to_end_ms),
            len(self.model_latency_ms),
            1,
        )
        rows = []
        for i in range(n):
            row = {
                "model_name": self.model_name,
                "cache_mode": self.cache_mode,
            }
            row.update(self.config_snapshot)

            for field_name in [
                "data_ms", "preprocess_ms", "vision_ms", "backbone_ms",
                "action_head_ms", "action_postprocess_ms", "model_latency_ms",
                "end_to_end_ms",
                "peak_cuda_memory_mb", "allocated_cuda_memory_mb",
                "reserved_cuda_memory_mb",
                "reused_token_ratio", "evicted_task_token_ratio",
                "cache_hit_rate",
                "action_l2_diff", "action_cosine_sim", "max_abs_action_diff",
            ]:
                values = getattr(self, field_name, [])
                row[field_name] = values[i] if i < len(values) else float("nan")

            # Compute derived fields
            for latency_field in ["data_ms", "preprocess_ms", "vision_ms",
                                   "backbone_ms", "action_head_ms",
                                   "action_postprocess_ms", "model_latency_ms",
                                   "end_to_end_ms"]:
                val = row.get(latency_field, float("nan"))
                if val > 0 and not np.isnan(val):
                    row[latency_field.replace("_ms", "_hz")] = 1000.0 / val
                else:
                    row[latency_field.replace("_ms", "_hz")] = 0.0

            rows.append(row)
        return rows

    def save_csv(self, path: str | Path) -> None:
        """Save raw measurements to CSV."""
        rows = self.to_csv_rows()
        if not rows:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def save_summary(self, path: str | Path) -> None:
        """Save summary statistics to JSON."""
        summary = {
            "model_name": self.model_name,
            "cache_mode": self.cache_mode,
            "config": self.config_snapshot,
            "statistics": self.summarize(),
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)


class ProfileTimer:
    """Context manager for profiling with CUDA synchronization."""

    def __init__(
        self,
        results: ProfileResults,
        key: str,
        use_cuda_events: bool = True,
    ):
        self.results = results
        self.key = key
        self.use_cuda_events = use_cuda_events and torch.cuda.is_available()
        self.start_time: float = 0.0
        self.start_ev: Optional[torch.cuda.Event] = None
        self.end_ev: Optional[torch.cuda.Event] = None

    def __enter__(self):
        if self.use_cuda_events:
            cuda_sync()
            self.start_time = time.perf_counter()
            self.start_ev = torch.cuda.Event(enable_timing=True)
            self.end_ev = torch.cuda.Event(enable_timing=True)
            self.start_ev.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.use_cuda_events and self.start_ev is not None and self.end_ev is not None:
            self.end_ev.record()
            cuda_sync()
            elapsed = self.start_ev.elapsed_time(self.end_ev)
        else:
            elapsed = elapsed_ms(self.start_time)
        self.results.record_timing(self.key, elapsed)


@contextmanager
def profile_section(
    results: ProfileResults,
    key: str,
    use_cuda_events: bool = True,
):
    """Context manager for timing a code section."""
    timer = ProfileTimer(results, key, use_cuda_events)
    timer.__enter__()
    try:
        yield
    finally:
        timer.__exit__()


def profile_policy_pipeline(
    policy_fn: Callable[[dict], dict],
    observations: list[dict],
    warmup_steps: int = 10,
    repeat_steps: int = 100,
    use_cuda_events: bool = True,
    profile_memory: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> ProfileResults:
    """Profile the full policy pipeline.

    Measures: data, preprocess, vision, backbone, action_head,
    action_postprocess, model, end-to-end latency.

    Args:
        policy_fn: Function that takes observation dict and returns
            (action_dict, timings_dict).
        observations: List of observation dicts.
        warmup_steps: Number of warmup iterations.
        repeat_steps: Number of measurement iterations.
        use_cuda_events: Use CUDA events for GPU timing.
        profile_memory: Record CUDA memory usage.
        progress_callback: Optional callback for progress reporting.

    Returns:
        ProfileResults with all measurements.
    """
    results = ProfileResults(use_cuda_events=use_cuda_events)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    for i in range(warmup_steps):
        obs = observations[i % len(observations)]
        policy_fn(obs)
        if progress_callback:
            progress_callback(i + 1, warmup_steps + repeat_steps)

    # Measurement
    for i in range(repeat_steps):
        obs = observations[i % len(observations)]
        cuda_sync()
        total_start = time.perf_counter()

        # Data fetch latency (already done — measure from observation receipt)
        data_start = time.perf_counter()

        # Run the policy
        with torch.inference_mode():
            action, timings = policy_fn(obs)

        cuda_sync()
        total_elapsed = elapsed_ms(total_start)

        # Extract timings from returned dict
        results.record_timing("preprocess_ms", timings.get("preprocess_ms", timings.get("server_preprocess_transform_ms", 0)))
        results.record_timing("vision_ms", timings.get("vision_ms", timings.get("server_system2_vision_ms", 0)))
        results.record_timing("backbone_ms", timings.get("backbone_ms", timings.get("server_system2_backbone_ms", 0)))
        results.record_timing("action_head_ms", timings.get("action_head_ms", timings.get("server_system1_action_head_ms", 0)))
        results.record_timing("action_postprocess_ms", timings.get("action_postprocess_ms", timings.get("server_postprocess_untransform_ms", 0)))
        results.record_timing("model_latency_ms", timings.get("model_total_ms", timings.get("server_model_total_ms", total_elapsed)))
        results.record_timing("end_to_end_ms", total_elapsed)

        if profile_memory:
            results.record_cuda_memory()

        if progress_callback:
            progress_callback(warmup_steps + i + 1, warmup_steps + repeat_steps)

    return results


def summarize_profile(results: ProfileResults) -> dict[str, Any]:
    """Generate a human-readable summary from profile results."""
    stats = results.summarize()

    lines = []
    lines.append("=" * 70)
    lines.append(f"  GR00T-Cache Profiling Results: {results.cache_mode}")
    lines.append("=" * 70)

    timing_fields = [
        ("Data", "data_ms"),
        ("Preprocess", "preprocess_ms"),
        ("Vision", "vision_ms"),
        ("Backbone", "backbone_ms"),
        ("Action Head", "action_head_ms"),
        ("Action Postprocess", "action_postprocess_ms"),
        ("Model Total", "model_latency_ms"),
        ("End-to-End", "end_to_end_ms"),
    ]

    for label, key in timing_fields:
        if key in stats:
            s = stats[key]
            lines.append(
                f"  {label:20s}: mean={s['mean']:7.2f}ms  "
                f"p50={s['p50']:7.2f}ms  p95={s['p95']:7.2f}ms  "
                f"min={s['min']:7.2f}ms  max={s['max']:7.2f}ms"
            )

    if "end_to_end_ms" in stats:
        e2e = stats["end_to_end_ms"]["mean"]
        hz = 1000.0 / e2e if e2e > 0 else 0
        lines.append(f"  {'Control Frequency':20s}: {hz:.1f} Hz")

    if "peak_cuda_memory_mb" in stats:
        mem = stats["peak_cuda_memory_mb"]
        lines.append(f"  {'Peak CUDA Memory':20s}: {mem['mean']:.0f} MB")

    if "reused_token_ratio" in stats:
        r = stats["reused_token_ratio"]
        lines.append(f"  {'Reused Token Ratio':20s}: {r['mean']:.3f}")

    lines.append("=" * 70)
    return "\n".join(lines), stats


def profile_gr00t_cache(
    policy_fn: Callable,
    observations: list[dict],
    warmup_steps: int = 10,
    repeat_steps: int = 100,
    use_cuda_events: bool = True,
    profile_memory: bool = True,
    output_dir: Optional[str | Path] = None,
) -> ProfileResults:
    """Convenience function to profile GR00T with caching.

    Args:
        policy_fn: Policy get_action function.
        observations: List of observation dicts.
        warmup_steps: Warmup iterations.
        repeat_steps: Measurement iterations.
        use_cuda_events: Use CUDA events for GPU timing.
        profile_memory: Record CUDA memory.
        output_dir: Directory to save output files.

    Returns:
        ProfileResults object.
    """
    results = profile_policy_pipeline(
        policy_fn=policy_fn,
        observations=observations,
        warmup_steps=warmup_steps,
        repeat_steps=repeat_steps,
        use_cuda_events=use_cuda_events,
        profile_memory=profile_memory,
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results.save_csv(output_dir / "raw_measurements.csv")
        results.save_summary(output_dir / "summary.json")
        summary_text, _ = summarize_profile(results)
        (output_dir / "summary.txt").write_text(summary_text)
        print(f"Results saved to {output_dir}")

    return results
