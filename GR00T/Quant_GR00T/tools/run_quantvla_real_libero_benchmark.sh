#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-baseline}"
TASK="${2:-libero_10}"
if (($# >= 3)) && [[ "$3" != --* ]]; then
  OUT_ROOT="$3"
  shift 3
else
  OUT_ROOT="results/benchmarks"
  shift 2
fi
EXTRA_EVAL_ARGS=("$@")

PRIMARY_LATENCY_KEY="${QVLA_PRIMARY_LATENCY_KEY:-dual_stage_sum_ms}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

source "$SCRIPT_DIR/quantvla_env.sh"
case "$VARIANT" in
  baseline|baseline_bf16_sdpa)
    quantvla_use_baseline "$TASK"
    ;;
  duquant|duquant_w4a8|duquant_w4_packed)
    quantvla_use_duquant "$TASK"
    ;;
  full|quantvla_full|quantvla_full_w4_packed)
    quantvla_use_full "$TASK"
    ;;
  fp8|fp8_selective)
    quantvla_use_fp8 "$TASK"
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    echo "Use one of: baseline, duquant, full, fp8" >&2
    exit 1
    ;;
esac

DENOISING_STEPS="$(quantvla_resolve_denoising_steps)"
OUT_DIR="$OUT_ROOT/$QVLA_VARIANT/$QVLA_TASK"
mkdir -p "$OUT_DIR"
METRICS_LOG="/tmp/logs/libero_eval_${QVLA_VARIANT}_${QVLA_TASK}.log"
mkdir -p "$(dirname "$METRICS_LOG")"

echo "[QuantVLA] Real LIBERO benchmark output: $OUT_DIR"
echo "[QuantVLA] This script does not run synthetic latency. It expects a matching server on port ${QVLA_PORT:-5556}."
echo "[QuantVLA] Start the server in another terminal with:"
echo "  QVLA_DENOISING_STEPS=$DENOISING_STEPS tools/run_quantvla_inference_server.sh $VARIANT $TASK"

# FLOPs loads the GR00T policy, so run it in the same environment as the
# inference server. run_libero_eval.sh below will switch to libero_test itself
# on local conda installs; Docker images use the current Python environment.
if [[ "${QUANTVLA_SKIP_CONDA:-0}" != "1" && -f "$HOME/miniconda3/etc/profile.d/conda.sh" && -d "$HOME/miniconda3/envs/groot_test" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate groot_test
else
  echo "[QuantVLA] Conda env groot_test not found; using current Python environment"
fi
python tools/check_cuda_available.py --context "QuantVLA FLOPs/memory pre-benchmark"

python tools/benchmark_gr00t_flops.py \
  --variant "$QVLA_VARIANT" \
  --model-path "$QVLA_MODEL_PATH" \
  --data-config "$QVLA_DATA_CONFIG" \
  --embodiment-tag new_embodiment \
  --denoising-steps "$DENOISING_STEPS" \
  --output-json "$OUT_DIR/flops.json"

DENSE_EQUIV_TFLOPS="$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['total_tflops'])" "$OUT_DIR/flops.json")"
TFLOPS_HEADER="[tflops] dense_equiv=$DENSE_EQUIV_TFLOPS TFLOPs/call"
echo "$TFLOPS_HEADER"
printf '%s\n' "$TFLOPS_HEADER" >> "$METRICS_LOG"

python tools/benchmark_gr00t_memory.py \
  --variant "$QVLA_VARIANT" \
  --model-path "$QVLA_MODEL_PATH" \
  --data-config "$QVLA_DATA_CONFIG" \
  --embodiment-tag new_embodiment \
  --denoising-steps "$DENOISING_STEPS" \
  --append-log "$METRICS_LOG" \
  --output-json "$OUT_DIR/memory.json"

./run_libero_eval.sh "$QVLA_TASK" \
  --headless \
  --profile-server \
  --print_step_latency \
  --dense_equiv_tflops_per_get_action "$DENSE_EQUIV_TFLOPS" \
  --log_file "$METRICS_LOG" \
  --output-json "$OUT_DIR/success.json" \
  "${EXTRA_EVAL_ARGS[@]}"

python tools/compute_real_libero_metrics.py \
  --variant "$QVLA_VARIANT" \
  --success-json "$OUT_DIR/success.json" \
  --flops-json "$OUT_DIR/flops.json" \
  --memory-json "$OUT_DIR/memory.json" \
  --primary-latency-key "$PRIMARY_LATENCY_KEY" \
  --append-log "$METRICS_LOG" \
  --output-json "$OUT_DIR/real_libero_metrics.json"

python tools/summarize_quantvla_results.py \
  --root "$OUT_ROOT" \
  --task "$QVLA_TASK" \
  --output-csv "$OUT_ROOT/summary.csv" \
  --output-md "$OUT_ROOT/summary.md"

echo "[QuantVLA] Done: $OUT_DIR"
