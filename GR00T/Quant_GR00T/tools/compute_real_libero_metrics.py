#!/usr/bin/env python3
"""Compute real LIBERO latency, dense-equivalent FLOPs, and component memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LATENCY_KEYS = (
    "client_policy_total_ms",
    "client_zmq_get_action_ms",
    "server_policy_total_ms",
    "server_gr00t_latency_ms",
    "server_dual_data_ms",
    "dual_data_ms",
    "dual_preprocess_ms",
    "server_model_total_ms",
    "server_system2_backbone_ms",
    "server_system2_vision_ms",
    "server_system2_reasoning_ms",
    "server_system2_to_system1_bridge_ms",
    "server_system2_other_ms",
    "server_system1_action_head_ms",
    "server_system1_vision_ms",
    "server_system1_action_ms",
    "dual_s2_vision_ms",
    "dual_s2_reasoning_ms",
    "dual_bridge_ms",
    "dual_s1_vision_ms",
    "dual_s1_action_ms",
    "dual_model_ms",
    "dual_model_compute_ms",
    "dual_post_ms",
    "dual_post_other_ms",
    "dual_stage_sum_ms",
    "libero_env_step_ms",
    "control_loop_step_ms",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path:
        return {}
    with path.open("r") as f:
        return json.load(f)


def get_nested(data: dict[str, Any], keys: tuple[str, ...], default=None):
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def metric_block(summary: dict[str, Any], key: str, total_flops: int | None) -> dict[str, Any] | None:
    stats = summary.get(key)
    if not isinstance(stats, dict):
        return None
    return dict(stats)


def flops_for_latency_key(flops: dict[str, Any], key: str, total_flops: int) -> int:
    if key in (
        "server_dual_data_ms",
        "dual_data_ms",
        "dual_preprocess_ms",
        "dual_post_ms",
        "dual_post_other_ms",
    ):
        return 0
    if key == "dual_stage_sum_ms":
        return total_flops
    if key == "server_system2_backbone_ms":
        value = sum(
            int(get_nested(flops, ("by_category", category, "flops"), 0) or 0)
            for category in (
                "system2_vision",
                "system2_reasoning",
                "system2_to_system1_bridge",
                "system2_other",
            )
        )
        return int(value) if value > 0 else total_flops
    if key in ("server_system2_vision_ms", "dual_s2_vision_ms"):
        value = get_nested(flops, ("by_category", "system2_vision", "flops"))
        return int(value) if value is not None else total_flops
    if key in ("server_system2_reasoning_ms", "dual_s2_reasoning_ms"):
        value = get_nested(flops, ("by_category", "system2_reasoning", "flops"))
        return int(value) if value is not None else total_flops
    if key in ("server_system2_to_system1_bridge_ms", "dual_bridge_ms"):
        value = get_nested(flops, ("by_category", "system2_to_system1_bridge", "flops"))
        return int(value) if value is not None else total_flops
    if key == "server_system2_other_ms":
        value = get_nested(flops, ("by_category", "system2_other", "flops"))
        return int(value) if value is not None else total_flops
    if key == "server_system1_action_head_ms":
        s1_vision = get_nested(flops, ("by_category", "system1_vision", "flops"), 0) or 0
        s1_action = get_nested(flops, ("by_category", "system1_action", "flops"), 0) or 0
        total = int(s1_vision) + int(s1_action)
        return total if total > 0 else total_flops
    if key in ("server_system1_vision_ms", "dual_s1_vision_ms"):
        value = get_nested(flops, ("by_category", "system1_vision", "flops"))
        return int(value) if value is not None else total_flops
    if key in ("server_system1_action_ms", "dual_s1_action_ms"):
        value = get_nested(flops, ("by_category", "system1_action", "flops"))
        return int(value) if value is not None else total_flops
    if key == "dual_model_ms":
        return total_flops
    if key == "dual_model_compute_ms":
        value = sum(
            int(get_nested(flops, ("by_category", category, "flops"), 0) or 0)
            for category in (
                "system2_vision",
                "system2_reasoning",
                "system2_to_system1_bridge",
                "system1_vision",
                "system1_action",
            )
        )
        return int(value) if value > 0 else total_flops
    return total_flops


def run(args: argparse.Namespace) -> dict[str, Any]:
    success = load_json(Path(args.success_json))
    flops = load_json(Path(args.flops_json))
    memory = load_json(Path(args.memory_json)) if args.memory_json else {}
    latency_summary = success.get("latency_summary", {})
    if not isinstance(latency_summary, dict):
        raise ValueError(f"{args.success_json} does not contain a latency_summary object")

    total_flops = flops.get("total_flops")
    if total_flops is None:
        raise ValueError(f"{args.flops_json} does not contain total_flops")

    latency = {}
    for key in LATENCY_KEYS:
        block = metric_block(latency_summary, key, flops_for_latency_key(flops, key, int(total_flops)))
        if block is not None:
            latency[key] = block

    primary_key = args.primary_latency_key
    if primary_key not in latency:
        available = ", ".join(sorted(latency))
        raise ValueError(f"Primary latency key {primary_key!r} not found. Available: {available}")

    end_to_end_keys = {
        "client_policy_total_ms": latency.get("client_policy_total_ms"),
        "server_policy_total_ms": latency.get("server_policy_total_ms"),
        "libero_env_step_ms": latency.get("libero_env_step_ms"),
        "control_loop_step_ms": latency.get("control_loop_step_ms"),
    }
    model_only_keys = {
        "server_gr00t_latency_ms": latency.get("server_gr00t_latency_ms"),
        "server_dual_data_ms": latency.get("server_dual_data_ms"),
        "dual_preprocess_ms": latency.get("dual_preprocess_ms"),
        "server_model_total_ms": latency.get("server_model_total_ms"),
        "server_system2_vision_ms": latency.get("server_system2_vision_ms"),
        "server_system2_reasoning_ms": latency.get("server_system2_reasoning_ms"),
        "server_system2_to_system1_bridge_ms": latency.get(
            "server_system2_to_system1_bridge_ms"
        ),
        "server_system1_vision_ms": latency.get("server_system1_vision_ms"),
        "server_system1_action_ms": latency.get("server_system1_action_ms"),
        "dual_model_ms": latency.get("dual_model_ms"),
        "dual_model_compute_ms": latency.get("dual_model_compute_ms"),
        "dual_post_ms": latency.get("dual_post_ms"),
        "dual_post_other_ms": latency.get("dual_post_other_ms"),
        "dual_stage_sum_ms": latency.get("dual_stage_sum_ms"),
    }
    component_memory = memory.get("model_component_memory", {})
    model_memory = {
        "parameter_and_buffer_bytes": get_nested(memory, ("model", "parameter_and_buffer_bytes")),
        "parameter_and_buffer_gib": bytes_to_gib(
            get_nested(memory, ("model", "parameter_and_buffer_bytes"))
        ),
        "duquant_linear_modules": get_nested(memory, ("model", "duquant_linear_modules")),
        "duquant_storage": get_nested(memory, ("model", "duquant_storage"), {}),
        "component_memory": component_memory,
        "component_total_bytes": get_nested(component_memory, ("total_bytes",)),
        "component_total_gib": bytes_to_gib(get_nested(component_memory, ("total_bytes",))),
        "component_llm_bytes": get_nested(component_memory, ("llm_bytes",)),
        "component_llm_gib": bytes_to_gib(get_nested(component_memory, ("llm_bytes",))),
        "component_dit_bytes": get_nested(component_memory, ("dit_bytes",)),
        "component_dit_gib": bytes_to_gib(get_nested(component_memory, ("dit_bytes",))),
        "component_llm_plus_dit_bytes": get_nested(component_memory, ("llm_plus_dit_bytes",)),
        "component_llm_plus_dit_gib": bytes_to_gib(
            get_nested(component_memory, ("llm_plus_dit_bytes",))
        ),
        "real_server_memory_summary": success.get("memory_summary", {}),
    }

    return {
        "metadata": {
            "variant": args.variant,
            "task_suite_name": success.get("task_suite_name"),
            "success_json": str(args.success_json),
            "flops_json": str(args.flops_json),
            "memory_json": str(args.memory_json) if args.memory_json else "",
            "primary_latency_key": primary_key,
            "tflops_definition": "dense_equiv_tflops_per_get_action = dense_equivalent_flops_per_get_action / 1e12",
            "latency_source": (
                "Real LIBERO rollout from run_libero_eval.py with --profile-server; "
                "no synthetic observation or smoke-test latency is used."
            ),
            "flops_source": get_nested(flops, ("notes",), []),
        },
        "success": {
            "total_episodes": success.get("total_episodes"),
            "total_successes": success.get("total_successes"),
            "success_rate": success.get("success_rate"),
            "elapsed_sec": success.get("elapsed_sec"),
        },
        "flops": {
            "total_flops_per_get_action": int(total_flops),
            "total_dense_equiv_tflops_per_get_action": float(total_flops) / 1e12,
            "by_category": flops.get("by_category", {}),
        },
        "end_to_end_latency": end_to_end_keys,
        "model_only_latency": model_only_keys,
        "model_memory": model_memory,
        "primary": {
            "latency_key": primary_key,
            **latency[primary_key],
        },
        "latency": latency,
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def bytes_to_gib(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) / (1024**3)


def fmt_gib_from_bytes(value: Any) -> str:
    gib = bytes_to_gib(value)
    if gib is None:
        return "n/a"
    return f"{gib:.3f}GiB"


def fmt_gib(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}GiB"


def fmt_stats_gib(stats: Any, key: str = "median_gib") -> str:
    if not isinstance(stats, dict):
        return "n/a"
    return fmt_gib(stats.get(key))


def component_entry_line(component: str, entry: dict[str, Any]) -> str:
    return (
        f"component={component} "
        f"size={fmt_gib_from_bytes(entry.get('bytes'))}, "
        f"numel={fmt(entry.get('numel'), 0)}, "
        f"tensors={fmt(entry.get('tensors'), 0)}"
    )


def summary_lines(result: dict[str, Any]) -> list[str]:
    lines = []
    success = result["success"]
    primary = result["primary"]
    flops = result["flops"]
    memory = result.get("model_memory", {})
    lines.append("[real metrics] Real LIBERO benchmark summary")
    lines.append(
        "[real metrics] "
        f"success_rate={fmt(success.get('success_rate'))}, "
        f"episodes={fmt(success.get('total_episodes'))}, "
        f"successes={fmt(success.get('total_successes'))}"
    )
    lines.append("[real metrics][latency]")
    for label, key in (
        ("Data", "dual_data_ms"),
        ("Preprocess", "dual_preprocess_ms"),
        ("System-2 Vision", "dual_s2_vision_ms"),
        ("System-2 Reasoning", "dual_s2_reasoning_ms"),
        ("Bridge", "dual_bridge_ms"),
        ("System-1 Vision", "dual_s1_vision_ms"),
        ("System-1 Action", "dual_s1_action_ms"),
        ("Post/Other", "dual_post_other_ms"),
        ("End to End Latency", "dual_stage_sum_ms"),
        ("Model Latency", "dual_model_ms"),
    ):
        stats = result["latency"].get(key)
        if not stats:
            continue
        lines.append(
            "[real metrics][latency] "
            f"{label}: p50={fmt(stats.get('median_ms'), 2)}ms, "
            f"p95={fmt(stats.get('p95_ms'), 2)}ms"
        )
    e2e_stats = result["latency"].get("dual_stage_sum_ms", {})
    lines.append(
        "[real metrics][tflops] "
        f"dense_equiv={fmt(flops.get('total_dense_equiv_tflops_per_get_action'))}TFLOPs/call"
    )
    if memory:
        server_memory = memory.get("real_server_memory_summary", {})
        if isinstance(server_memory, dict) and server_memory:
            total = server_memory.get("server_model_component_total_bytes", {})
            llm = server_memory.get("server_model_component_llm_bytes", {})
            dit = server_memory.get("server_model_component_dit_bytes", {})
            llm_plus_dit = server_memory.get("server_model_component_llm_plus_dit_bytes", {})
            vision = server_memory.get("server_model_component_vision_bytes", {})
            duquant_layers = server_memory.get("server_duquant_linear_modules", {})
            packed_layers = server_memory.get("server_duquant_packed_modules", {})
            fake_layers = server_memory.get("server_duquant_fake_modules", {})
            atm_env = server_memory.get("server_atm_env_enabled", {})
            ohb_env = server_memory.get("server_ohb_env_enabled", {})
            has_component_stats = any(
                isinstance(stats, dict) and stats.get("median_gib") is not None
                for stats in (total, llm, dit, llm_plus_dit)
            )
            if has_component_stats:
                lines.append(
                    "[real metrics][model component memory] real_server: "
                    f"llm={fmt_stats_gib(llm)}, "
                    f"dit={fmt_stats_gib(dit)}, "
                    f"llm+dit={fmt_stats_gib(llm_plus_dit)}, "
                    f"total={fmt_stats_gib(total)}, "
                    f"vision={fmt_stats_gib(vision)}, "
                    f"duquant_layers={fmt(duquant_layers.get('median_bytes'), 0)}, "
                    f"packed_layers={fmt(packed_layers.get('median_bytes'), 0)}, "
                    f"fake_layers={fmt(fake_layers.get('median_bytes'), 0)}, "
                    f"atm_env={fmt(atm_env.get('median_bytes'), 0)}, "
                    f"ohb_env={fmt(ohb_env.get('median_bytes'), 0)}"
                )
        component_memory = memory.get("component_memory", {})
        if memory.get("component_total_gib") is not None:
            lines.append(
                "[real metrics][model component memory] offline: "
                f"llm={fmt_gib(memory.get('component_llm_gib'))}, "
                f"dit={fmt_gib(memory.get('component_dit_gib'))}, "
                f"llm+dit={fmt_gib(memory.get('component_llm_plus_dit_gib'))}, "
                f"total={fmt_gib(memory.get('component_total_gib'))}, "
                f"duquant_layers={fmt(memory.get('duquant_linear_modules'))}, "
                f"duquant_storage={memory.get('duquant_storage', {})}"
            )
        components = component_memory.get("components", {}) if isinstance(component_memory, dict) else {}
        for component in ("llm", "dit", "vision", "backbone_other", "action_head_other", "other"):
            entry = components.get(component)
            if isinstance(entry, dict):
                lines.append(
                    "[real metrics][model component memory] "
                    + component_entry_line(component, entry)
                )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--success-json", required=True)
    parser.add_argument("--flops-json", required=True)
    parser.add_argument("--memory-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--append-log", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument(
        "--primary-latency-key",
        default="client_policy_total_ms",
        choices=LATENCY_KEYS,
        help=(
            "Main real LIBERO latency key stored in the primary summary block. "
            "Use control_loop_step_ms if you want to include simulator env.step."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"Wrote real LIBERO metrics to {output_path}")
    lines = summary_lines(result)
    for line in lines:
        print(line)
    if args.append_log:
        log_path = Path(args.append_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            for line in lines:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
