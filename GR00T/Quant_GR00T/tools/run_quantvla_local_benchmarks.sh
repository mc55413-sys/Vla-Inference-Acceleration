#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-baseline}"
TASK="${2:-libero_10}"
OUT_ROOT="${3:-results/benchmarks}"
WARMUP="${QVLA_WARMUP:-5}"
ITERS="${QVLA_ITERS:-20}"
DENOISING_STEPS="${QVLA_DENOISING_STEPS:-8}"
ACTION_SAMPLES="${QVLA_ACTION_SAMPLES:-32}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

source "$SCRIPT_DIR/quantvla_env.sh"
case "$VARIANT" in
  baseline|baseline_bf16_sdpa)
    quantvla_use_baseline "$TASK"
    ;;
  duquant|duquant_w4a8)
    quantvla_use_duquant "$TASK"
    ;;
  full|quantvla_full)
    quantvla_use_full "$TASK"
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    echo "Use one of: baseline, duquant, full" >&2
    exit 1
    ;;
esac

OUT_DIR="$OUT_ROOT/$QVLA_VARIANT/$QVLA_TASK"
mkdir -p "$OUT_DIR"

echo "[QuantVLA] Writing local benchmark outputs to $OUT_DIR"

python tools/benchmark_gr00t_latency.py \
  --mode local \
  --variant "$QVLA_VARIANT" \
  --model-path "$QVLA_MODEL_PATH" \
  --data-config "$QVLA_DATA_CONFIG" \
  --embodiment-tag new_embodiment \
  --denoising-steps "$DENOISING_STEPS" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --fine-grained \
  --output-json "$OUT_DIR/latency_local.json"

python tools/benchmark_gr00t_memory.py \
  --variant "$QVLA_VARIANT" \
  --model-path "$QVLA_MODEL_PATH" \
  --data-config "$QVLA_DATA_CONFIG" \
  --embodiment-tag new_embodiment \
  --denoising-steps "$DENOISING_STEPS" \
  --warmup 3 \
  --iters 10 \
  --output-json "$OUT_DIR/memory.json"

python tools/benchmark_gr00t_flops.py \
  --variant "$QVLA_VARIANT" \
  --model-path "$QVLA_MODEL_PATH" \
  --data-config "$QVLA_DATA_CONFIG" \
  --embodiment-tag new_embodiment \
  --denoising-steps "$DENOISING_STEPS" \
  --output-json "$OUT_DIR/flops.json"

python tools/collect_gr00t_actions.py \
  --variant "$QVLA_VARIANT" \
  --model-path "$QVLA_MODEL_PATH" \
  --data-config "$QVLA_DATA_CONFIG" \
  --embodiment-tag new_embodiment \
  --denoising-steps "$DENOISING_STEPS" \
  --num-samples "$ACTION_SAMPLES" \
  --output-npz "$OUT_DIR/actions.npz" \
  --output-json "$OUT_DIR/actions.json"

echo "[QuantVLA] Done: $OUT_DIR"
