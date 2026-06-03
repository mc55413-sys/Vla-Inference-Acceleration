#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"

if [[ -f "${PROJECT_DIR}/local_config.sh" ]]; then
  # Optional machine-local overrides. This file is git-ignored.
  # shellcheck source=/dev/null
  source "${PROJECT_DIR}/local_config.sh"
fi

OPENVLA_DIR="${PROJECT_DIR}/src/openvla"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_DIR}/checkpoints/openvla-7b-finetuned-libero-spatial}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/profiling_results}"
REPORT_PATH="${REPORT_PATH:-${RESULTS_DIR}/PERFORMANCE_ANALYSIS.md}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TRIALS="${TRIALS:-1}"
MAX_TASKS="${MAX_TASKS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
WARMUP_ACTION_STEPS="${WARMUP_ACTION_STEPS:-1}"
STATIC_PATCH_TOP_K="${STATIC_PATCH_TOP_K:-150}"
STATIC_PATCH_SIM_THRESHOLD="${STATIC_PATCH_SIM_THRESHOLD:-0.996}"
ATTENTION_TOP_K="${ATTENTION_TOP_K:-120}"
CHECK_ONLY="${CHECK_ONLY:-0}"

export PYTHONPATH="${PROJECT_DIR}/src/LIBERO:${OPENVLA_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="false"

cd "${OPENVLA_DIR}"

COMMON_ARGS=(
  --pretrained_checkpoint "${CHECKPOINT_PATH}"
  --task_suite_name "${TASK_SUITE}"
  --num_trials_per_task "${TRIALS}"
  --max_tasks "${MAX_TASKS}"
  --warmup_action_steps "${WARMUP_ACTION_STEPS}"
  --static_patch_top_k "${STATIC_PATCH_TOP_K}"
  --static_patch_sim_threshold "${STATIC_PATCH_SIM_THRESHOLD}"
  --attention_top_k "${ATTENTION_TOP_K}"
  --output_dir "${RESULTS_DIR}"
)

if [[ -n "${MAX_STEPS}" ]]; then
  COMMON_ARGS+=(--max_steps "${MAX_STEPS}")
fi

"${PYTHON_BIN}" experiments/robot/run_profiling_experiment.py \
  "${COMMON_ARGS[@]}" \
  --check_only True

if [[ "${CHECK_ONLY}" == "1" ]]; then
  exit 0
fi

"${PYTHON_BIN}" experiments/robot/run_profiling_experiment.py \
  "${COMMON_ARGS[@]}" \
  --use_vla_cache True

"${PYTHON_BIN}" experiments/robot/run_profiling_experiment.py \
  "${COMMON_ARGS[@]}" \
  --use_vla_cache False

"${PYTHON_BIN}" experiments/robot/generate_analysis_report.py \
  --with_cache "${RESULTS_DIR}/profiling_with_cache.json" \
  --without_cache "${RESULTS_DIR}/profiling_without_cache.json" \
  --output "${REPORT_PATH}"

echo "Done. Report: ${REPORT_PATH}"
