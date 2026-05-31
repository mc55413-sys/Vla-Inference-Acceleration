#!/usr/bin/env python3
"""Summarize QuantVLA benchmark result directories into CSV/Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def get_nested(data: dict[str, Any], keys: list[str], default=None):
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def bytes_to_gib(value):
    if value is None:
        return None
    return float(value) / (1024**3)


def row_for_dir(path: Path) -> dict[str, Any]:
    latency = load_json(path / "latency_local.json")
    client_latency = load_json(path / "latency_client.json")
    memory = load_json(path / "memory.json")
    flops = load_json(path / "flops.json")
    success = load_json(path / "success.json")
    real_metrics = load_json(path / "real_libero_metrics.json")
    drift = load_json(path / "action_drift_vs_baseline.json")

    variant = get_nested(latency, ["metadata", "variant"], path.parent.name)
    task = path.name
    policy_p50 = get_nested(latency, ["summary", "policy_total_profiled_ms", "median_ms"])
    preprocess_p50 = get_nested(latency, ["summary", "preprocess_transform_ms", "median_ms"])
    backbone_p50 = get_nested(latency, ["summary", "backbone_eagle_ms", "median_ms"])
    action_p50 = get_nested(latency, ["summary", "action_head_total_ms", "median_ms"])
    if action_p50 is None:
        action_p50 = get_nested(latency, ["summary", "action_head_total_profiled_ms", "median_ms"])
    dit_per_step = get_nested(latency, ["summary", "action_dit_per_step_ms", "median_ms"])
    client_p95 = get_nested(client_latency, ["summary", "client_zmq_end_to_end_ms", "p95_ms"])
    if client_p95 is None:
        client_p95 = get_nested(
            success, ["latency_summary", "client_zmq_get_action_ms", "p95_ms"]
        )
    real_policy_p50 = get_nested(
        success, ["latency_summary", "client_policy_total_ms", "median_ms"]
    )
    real_system2_p50 = get_nested(
        success, ["latency_summary", "server_system2_backbone_ms", "median_ms"]
    )
    real_system1_p50 = get_nested(
        success, ["latency_summary", "server_system1_action_head_ms", "median_ms"]
    )
    real_gr00t_p50 = get_nested(
        success, ["latency_summary", "server_gr00t_latency_ms", "median_ms"]
    )
    real_stage_sum_p50 = get_nested(
        success, ["latency_summary", "dual_stage_sum_ms", "median_ms"]
    )
    real_model_latency_p50 = get_nested(
        success, ["latency_summary", "dual_model_ms", "median_ms"]
    )
    real_data_p50 = get_nested(
        success, ["latency_summary", "dual_data_ms", "median_ms"]
    )
    real_preprocess_p50 = get_nested(
        success, ["latency_summary", "dual_preprocess_ms", "median_ms"]
    )
    real_s2_vision_p50 = get_nested(
        success, ["latency_summary", "server_system2_vision_ms", "median_ms"]
    )
    real_s2_reasoning_p50 = get_nested(
        success, ["latency_summary", "server_system2_reasoning_ms", "median_ms"]
    )
    real_bridge_p50 = get_nested(
        success, ["latency_summary", "server_system2_to_system1_bridge_ms", "median_ms"]
    )
    real_s1_vision_p50 = get_nested(
        success, ["latency_summary", "server_system1_vision_ms", "median_ms"]
    )
    real_s1_action_p50 = get_nested(
        success, ["latency_summary", "server_system1_action_ms", "median_ms"]
    )
    real_post_p50 = get_nested(
        success, ["latency_summary", "dual_post_other_ms", "median_ms"]
    )
    real_server_policy_p50 = get_nested(
        success, ["latency_summary", "server_policy_total_ms", "median_ms"]
    )
    real_model_p50 = get_nested(
        success, ["latency_summary", "server_model_total_ms", "median_ms"]
    )
    real_control_loop_p50 = get_nested(
        success, ["latency_summary", "control_loop_step_ms", "median_ms"]
    )

    component_total = get_nested(memory, ["model_component_memory", "total_bytes"])
    component_llm = get_nested(memory, ["model_component_memory", "llm_bytes"])
    component_dit = get_nested(memory, ["model_component_memory", "dit_bytes"])
    component_llm_plus_dit = get_nested(memory, ["model_component_memory", "llm_plus_dit_bytes"])
    component_vision = get_nested(memory, ["model_component_memory", "components", "vision", "bytes"])
    return {
        "variant": variant,
        "task": task,
        "success_rate": success.get("success_rate"),
        "synthetic_local_policy_p50_ms": policy_p50,
        "real_libero_policy_p50_ms": real_policy_p50,
        "real_libero_server_policy_p50_ms": real_server_policy_p50,
        "real_libero_gr00t_p50_ms": real_gr00t_p50,
        "real_libero_end_to_end_latency_p50_ms": real_stage_sum_p50,
        "real_libero_model_latency_p50_ms": real_model_latency_p50,
        "real_libero_server_model_p50_ms": real_model_p50,
        "real_libero_control_loop_p50_ms": real_control_loop_p50,
        "latency_client_p95_ms": client_p95,
        "system2_preprocess_plus_backbone_p50_ms": (
            preprocess_p50 + backbone_p50
            if preprocess_p50 is not None and backbone_p50 is not None
            else None
        ),
        "system2_backbone_p50_ms": backbone_p50,
        "system1_action_head_p50_ms": action_p50,
        "real_libero_system2_backbone_p50_ms": real_system2_p50,
        "real_libero_system1_action_head_p50_ms": real_system1_p50,
        "real_libero_data_p50_ms": real_data_p50,
        "real_libero_preprocess_p50_ms": real_preprocess_p50,
        "real_libero_s2_vision_p50_ms": real_s2_vision_p50,
        "real_libero_s2_reasoning_p50_ms": real_s2_reasoning_p50,
        "real_libero_bridge_p50_ms": real_bridge_p50,
        "real_libero_s1_vision_p50_ms": real_s1_vision_p50,
        "real_libero_s1_action_p50_ms": real_s1_action_p50,
        "real_libero_post_other_p50_ms": real_post_p50,
        "dit_per_step_p50_ms": dit_per_step,
        "model_component_llm_gib": bytes_to_gib(component_llm),
        "model_component_dit_gib": bytes_to_gib(component_dit),
        "model_component_llm_plus_dit_gib": bytes_to_gib(component_llm_plus_dit),
        "model_component_total_gib": bytes_to_gib(component_total),
        "model_component_vision_gib": bytes_to_gib(component_vision),
        "duquant_layers": get_nested(memory, ["model", "duquant_linear_modules"]),
        "duquant_storage": get_nested(memory, ["model", "duquant_storage"]),
        "dense_equiv_tflops_per_call": flops.get("total_tflops"),
        "action_relative_l2_mean": drift.get("relative_l2_mean"),
        "action_mae": drift.get("mae"),
        "gripper_mismatch_rate": drift.get("gripper_mismatch_rate"),
    }


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    headers = list(rows[0].keys()) if rows else []
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header)) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/benchmarks")
    parser.add_argument("--task", default="libero_10")
    parser.add_argument("--output-csv", default="results/benchmarks/summary.csv")
    parser.add_argument("--output-md", default="results/benchmarks/summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = []
    for variant_dir in sorted(root.iterdir()) if root.exists() else []:
        task_dir = variant_dir / args.task
        if task_dir.is_dir():
            rows.append(row_for_dir(task_dir))

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["variant"])
        writer.writeheader()
        writer.writerows(rows)

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(rows, out_md)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
