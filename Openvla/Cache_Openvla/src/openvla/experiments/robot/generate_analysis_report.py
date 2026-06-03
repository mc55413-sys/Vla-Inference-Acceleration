"""Generate Markdown reports from OpenVLA/VLA-Cache profiling JSON files."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


STAGES = [
    ("E2E", "e2e_observation_to_action_ready"),
    ("Data", "data_observation_to_model_tensors_prompt"),
    ("Vision", "vision_image_tensor_to_projector_output"),
    ("LLM", "llm_multimodal_prefix_forward"),
    ("Action", "action_decode_or_denoise_to_continuous"),
]


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(data: Dict[str, Any], stage: str, group: str, key: str = "mean") -> Optional[float]:
    try:
        value = data[stage][group][key]
    except KeyError:
        return None
    return float(value) if value is not None else None


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _delta(with_value: Optional[float], without_value: Optional[float]) -> Tuple[str, str]:
    if with_value is None or without_value is None or without_value == 0:
        return "N/A", "N/A"
    ratio = without_value / with_value if with_value > 0 else 0.0
    change = (with_value - without_value) / without_value * 100.0
    return f"{change:+.1f}%", f"{ratio:.2f}x"


def _stage_table(with_cache: Dict[str, Any], without_cache: Dict[str, Any]) -> Iterable[str]:
    yield "| Stage | without_cache latency ms | with_cache latency ms | Change | Speedup | without_cache TFLOPs/sample | with_cache TFLOPs/sample | TFLOPs Change |"
    yield "|------|------:|------:|------:|------:|------:|------:|------:|"
    for label, key in STAGES:
        w_lat = _metric(with_cache, key, "latency_ms")
        wo_lat = _metric(without_cache, key, "latency_ms")
        lat_change, speedup = _delta(w_lat, wo_lat)
        w_tflops = _metric(with_cache, key, "tflops")
        wo_tflops = _metric(without_cache, key, "tflops")
        flop_change, _ = _delta(w_tflops, wo_tflops)
        yield (
            f"| {label} | {_fmt(wo_lat)} | {_fmt(w_lat)} | {lat_change} | {speedup} | "
            f"{_fmt(wo_tflops, 4)} | {_fmt(w_tflops, 4)} | {flop_change} |"
        )


def _cache_summary(data: Dict[str, Any]) -> Dict[str, float]:
    steps = [step for step in data.get("per_step", []) if step.get("recorded")]
    if not steps:
        return {"used_rate": 0.0, "stable_patches": 0.0, "reusable_tokens": 0.0}
    return {
        "used_rate": sum(1 for step in steps if step.get("used_vla_cache")) / len(steps),
        "stable_patches": sum(float(step.get("stable_patch_count", 0.0)) for step in steps) / len(steps),
        "reusable_tokens": sum(float(step.get("reusable_token_count", 0.0)) for step in steps) / len(steps),
    }


def _bottleneck(data: Dict[str, Any]) -> str:
    candidates = []
    for label, key in STAGES:
        if label == "E2E":
            continue
        lat = _metric(data, key, "latency_ms")
        if lat is not None:
            candidates.append((lat, label))
    return max(candidates)[1] if candidates else "N/A"


def _effectiveness(with_cache: Dict[str, Any], without_cache: Dict[str, Any]) -> str:
    w_llm = _metric(with_cache, "llm_multimodal_prefix_forward", "latency_ms")
    wo_llm = _metric(without_cache, "llm_multimodal_prefix_forward", "latency_ms")
    w_e2e = _metric(with_cache, "e2e_observation_to_action_ready", "latency_ms")
    wo_e2e = _metric(without_cache, "e2e_observation_to_action_ready", "latency_ms")
    w_tf = _metric(with_cache, "llm_multimodal_prefix_forward", "tflops")
    wo_tf = _metric(without_cache, "llm_multimodal_prefix_forward", "tflops")

    signals = []
    if w_llm is not None and wo_llm is not None and wo_llm > 0:
        signals.append(f"LLM latency changes by {(w_llm - wo_llm) / wo_llm * 100.0:+.1f}%")
    if w_tf is not None and wo_tf is not None and wo_tf > 0:
        signals.append(f"LLM TFLOPs/sample changes by {(w_tf - wo_tf) / wo_tf * 100.0:+.1f}%")
    if w_e2e is not None and wo_e2e is not None and wo_e2e > 0:
        signals.append(f"E2E latency changes by {(w_e2e - wo_e2e) / wo_e2e * 100.0:+.1f}%")
    if not signals:
        return "VLA-Cache effectiveness cannot be determined from the available metrics."
    return "; ".join(signals) + "."


def generate(with_cache_path: Path, without_cache_path: Path, output_path: Path) -> None:
    with_cache = _load(with_cache_path)
    without_cache = _load(without_cache_path)
    with_cfg = with_cache.get("config", {})
    without_cfg = without_cache.get("config", {})
    cache_diag = _cache_summary(with_cache)

    lines = [
        "# OpenVLA / VLA-Cache Profiling Analysis",
        "",
        "## Summary",
        "",
        f"- With-cache result: `{with_cache_path}`",
        f"- Without-cache result: `{without_cache_path}`",
        f"- Recorded action steps: with_cache={with_cfg.get('num_recorded_action_steps', 'N/A')}, without_cache={without_cfg.get('num_recorded_action_steps', 'N/A')}",
        f"- Success rate: with_cache={_fmt(float(with_cfg.get('success_rate', 0.0)) * 100, 1)}%, without_cache={_fmt(float(without_cfg.get('success_rate', 0.0)) * 100, 1)}%",
        f"- Warmup action steps excluded per episode: {with_cfg.get('warmup_action_steps_per_episode', 'N/A')}",
        "",
        "## Stage Comparison",
        "",
        *_stage_table(with_cache, without_cache),
        "",
        "## Bottleneck",
        "",
        f"- without_cache bottleneck: {_bottleneck(without_cache)}",
        f"- with_cache bottleneck: {_bottleneck(with_cache)}",
        "",
        "## Cache Diagnostics",
        "",
        f"- Recorded-step cache-use rate: {_fmt(cache_diag['used_rate'] * 100, 1)}%",
        f"- Average stable patches: {_fmt(cache_diag['stable_patches'], 1)}",
        f"- Average reusable tokens after task filtering: {_fmt(cache_diag['reusable_tokens'], 1)}",
        f"- Static patch top-k: {with_cfg.get('static_patch_top_k', 'N/A')}",
        f"- Attention top-k: {with_cfg.get('attention_top_k', 'N/A')}",
        "",
        "## Interpretation",
        "",
        _effectiveness(with_cache, without_cache),
        "",
        "The `TFLOPs/sample` values are theoretical Llama decoder FLOPs following the VLA-Cache formula",
        "`4*n*d^2 + 2*n^2*d + 3*n*d*m`. They are not Nsight hardware throughput and should be compared stage-by-stage rather than summed across the full pipeline.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with_cache", type=Path, required=True)
    parser.add_argument("--without_cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.with_cache, args.without_cache, args.output)


if __name__ == "__main__":
    main()
