#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-baseline}"
TASK="${2:-libero_10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

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

export GR00T_DENOISING_STEPS="$(quantvla_resolve_denoising_steps)"
python tools/check_cuda_available.py --context "QuantVLA inference server"

echo "[QuantVLA] Starting real inference server"
echo "  variant: $QVLA_VARIANT"
echo "  task: $QVLA_TASK"
echo "  denoising steps: $GR00T_DENOISING_STEPS"
echo "  port: ${QVLA_PORT:-5556}"

exec ./run_inference_server.sh "$QVLA_TASK"
