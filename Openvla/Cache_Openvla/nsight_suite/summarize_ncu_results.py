#!/usr/bin/env python3
"""Summarize Nsight Compute CSV exports for VLA-Cache experiments."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


VALUE_KEYS = ("Metric Value", "Value", "Avg", "Average", "Sum")
NAME_KEYS = ("Metric Name", "Name")
UNIT_KEYS = ("Metric Unit", "Unit")
KERNEL_KEYS = ("Kernel Name", "Kernel", "Demangled Name")

PRIMARY_METRIC_PATTERNS = (
    "pct_of_peak_sustained_elapsed",
    "throughput",
    "duration",
    "tensor",
    "hmma",
    "mma",
    "fadd",
    "fmul",
    "ffma",
    "hadd",
    "hmul",
    "hfma",
    "dadd",
    "dmul",
    "dfma",
)


def _clean_number(value: str) -> Optional[float]:
    if value is None:
        return None
    text = value.strip().replace(",", "")
    if not text or text.upper() in {"N/A", "NAN", "INF", "-INF"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _pick(row: Dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        if key in row and row[key]:
            return row[key]
    return ""


def _duration_to_seconds(value: float, unit: str) -> Optional[float]:
    unit = unit.strip().lower()
    if unit in {"s", "sec", "second", "seconds"}:
        return value
    if unit in {"ms", "msecond", "millisecond", "milliseconds"}:
        return value / 1e3
    if unit in {"us", "usecond", "microsecond", "microseconds"}:
        return value / 1e6
    if unit in {"ns", "nsecond", "nanosecond", "nanoseconds"}:
        return value / 1e9
    return None


def _is_duration_metric(metric: str, unit: str) -> bool:
    lower = metric.lower()
    return "duration" in lower or unit.strip().lower() in {"s", "ms", "us", "ns"}


def _flop_weight(metric: str) -> int:
    lower = metric.lower()
    if "ffma" in lower or "hfma" in lower or "dfma" in lower:
        return 2
    if any(token in lower for token in ("fadd", "fmul", "hadd", "hmul", "dadd", "dmul")):
        return 1
    return 0


def _fmt_float(value: object, precision: int = 4, scientific: bool = False) -> str:
    if not isinstance(value, float):
        return "-"
    if scientific:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def load_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path.exists() or path.stat().st_size == 0:
        return rows

    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("=="))
        for row in reader:
            if row:
                rows.append({str(k).strip(): str(v).strip() for k, v in row.items() if k is not None})
    return rows


def summarize_csv(path: Path) -> Dict[str, object]:
    rows = load_rows(path)
    metrics: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    kernels = set()

    for row in rows:
        metric = _pick(row, NAME_KEYS)
        unit = _pick(row, UNIT_KEYS)
        value = _clean_number(_pick(row, VALUE_KEYS))
        kernel = _pick(row, KERNEL_KEYS)
        if kernel:
            kernels.add(kernel)
        if metric and value is not None:
            metrics[metric].append((value, unit))

    duration_seconds = 0.0
    flop_count = 0.0
    selected_metrics = []
    for metric, values in metrics.items():
        metric_lower = metric.lower()
        numeric_values = [value for value, _ in values]
        units = [unit for _, unit in values if unit]
        unit = units[0] if units else ""
        total = sum(numeric_values)
        mean = total / len(numeric_values) if numeric_values else 0.0

        if _is_duration_metric(metric, unit):
            converted = [_duration_to_seconds(value, unit) for value, unit in values]
            duration_seconds += sum(value for value in converted if value is not None)

        weight = _flop_weight(metric)
        if weight:
            flop_count += total * weight

        if any(pattern in metric_lower for pattern in PRIMARY_METRIC_PATTERNS):
            selected_metrics.append((metric, len(values), mean, total, unit))

    tflops = flop_count / duration_seconds / 1e12 if duration_seconds > 0 and flop_count > 0 else None
    selected_metrics.sort(key=lambda item: item[0].lower())

    return {
        "path": path,
        "row_count": len(rows),
        "kernel_count": len(kernels),
        "metric_count": len(metrics),
        "duration_seconds": duration_seconds if duration_seconds > 0 else None,
        "flop_count": flop_count if flop_count > 0 else None,
        "estimated_tflops": tflops,
        "selected_metrics": selected_metrics,
    }


def _latest_run(root: Path) -> Path:
    runs = sorted((p for p in root.glob("run_*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No run_* directory under {root}")
    return runs[-1]


def _find_raw_csv(run_dir: Path, tag: str) -> Optional[Path]:
    candidates = sorted((run_dir / tag).glob("ncu_*_raw.csv"))
    return candidates[0] if candidates else None


def render_markdown(run_dir: Path, output: Path) -> None:
    entries = []
    for tag in ("with_cache", "without_cache"):
        path = _find_raw_csv(run_dir, tag)
        entries.append((tag, path, summarize_csv(path) if path is not None else None))

    lines: List[str] = []
    lines.append("# Nsight Compute 指标分析\n\n")
    lines.append(f"- 实验目录: `{run_dir}`\n")
    lines.append("- 数据来源: `ncu_*_raw.csv`，由 `.ncu-rep` 导出。\n")
    lines.append("- 只有当 NCU 成功采集到 FLOP 指令计数和 duration 指标时，本文才给出 derived TFLOPS。\n\n")

    lines.append("## 总览\n")
    lines.append("| 模式 | raw CSV | kernel 数 | metric 数 | duration(s) | FLOP 计数 | derived TFLOPS |\n")
    lines.append("|------|---------|-----------|-----------|-------------|-----------|----------------|\n")
    for title, path, summary in entries:
        if summary is None:
            lines.append(f"| {title} | 未找到 | - | - | - | - | - |\n")
            continue
        rel_path = path.relative_to(run_dir) if path is not None else "-"
        lines.append(
            f"| {title} | `{rel_path}` | {summary['kernel_count']} | {summary['metric_count']} | "
            f"{_fmt_float(summary['duration_seconds'], 6)} | "
            f"{_fmt_float(summary['flop_count'], 4, scientific=True)} | "
            f"{_fmt_float(summary['estimated_tflops'], 4)} |\n"
        )

    lines.append("\n## 关键 NCU 指标\n")
    for title, _path, summary in entries:
        lines.append(f"### {title}\n")
        if summary is None:
            lines.append("未找到 NCU raw CSV。\n\n")
            continue
        selected = summary["selected_metrics"][:80]
        if not selected:
            lines.append("raw CSV 中没有匹配 throughput/duration/FLOP/tensor 关键词的指标。\n\n")
            continue
        lines.append("| Metric Name | 样本数 | mean | sum | unit |\n")
        lines.append("|-------------|--------|------|-----|------|\n")
        for metric, count, mean, total, unit in selected:
            lines.append(f"| `{metric}` | {count} | {mean:.6g} | {total:.6g} | {unit} |\n")
        lines.append("\n")

    lines.append("## 解读\n")
    lines.append("- NCU 的原始计数器比 Python hook 估算更可靠，适合回答 kernel 级别的 SM/Tensor/Memory 吞吐。\n")
    lines.append("- `pct_of_peak_sustained_elapsed` 是峰值占比，不是直接 TFLOPS；要换算绝对 TFLOPS，需要已知 GPU 峰值或采集 FLOP 指令计数。\n")
    lines.append("- 如果报告里没有 derived TFLOPS，通常是所选 `NCU_SET/NCU_METRICS` 没有包含 FLOP 指令计数，或系统未开放 performance counter 权限。\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize NCU CSV exports")
    parser.add_argument("--run-dir", type=Path, default=None, help="Nsight run directory. Defaults to latest run.")
    default_root = Path(__file__).resolve().parents[1] / "nsight_results"
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir is not None else _latest_run(args.root)
    output = args.output if args.output is not None else run_dir / "FINAL_NCU_ANALYSIS.md"
    render_markdown(run_dir, output)


if __name__ == "__main__":
    main()
