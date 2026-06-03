#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ -f "${PROJECT_DIR}/local_config.sh" ]]; then
  # Optional machine-local overrides. This file is git-ignored.
  # shellcheck source=/dev/null
  source "${PROJECT_DIR}/local_config.sh"
fi

OPENVLA_DIR="${PROJECT_DIR}/src/openvla"
PYTHON_BIN="${PYTHON_BIN:-python}"
NSYS_BIN="${NSYS_BIN:-/opt/nvidia/nsight-systems/2026.2.1/target-linux-x64/nsys}"
NCU_BIN="${NCU_BIN:-/opt/nvidia/nsight-compute/2025.3.1/ncu}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_DIR}/checkpoints/openvla-7b-finetuned-libero-spatial}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TRIALS="${TRIALS:-1}"
MAX_TASKS="${MAX_TASKS:-1}"
MAX_STEPS="${MAX_STEPS:-2}"
WARMUP_ACTION_STEPS="${WARMUP_ACTION_STEPS:-1}"
STATIC_PATCH_TOP_K="${STATIC_PATCH_TOP_K:-150}"
STATIC_PATCH_SIM_THRESHOLD="${STATIC_PATCH_SIM_THRESHOLD:-0.996}"
ATTENTION_TOP_K="${ATTENTION_TOP_K:-120}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/nsight_results/run_${RUN_ID}}"
PROFILE_MODE="${PROFILE_MODE:-nsys}"
CACHE_MODE="${CACHE_MODE:-both}"
CHECK_ONLY="${CHECK_ONLY:-0}"

NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cudnn,cublas}"
NSYS_SAMPLE="${NSYS_SAMPLE:-cpu}"
NSYS_STATS_REPORTS="${NSYS_STATS_REPORTS:-cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum}"
NCU_SET="${NCU_SET:-speedOfLight}"
NCU_TARGET_PROCESSES="${NCU_TARGET_PROCESSES:-all}"
NCU_REPLAY_MODE="${NCU_REPLAY_MODE:-kernel}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-0}"
NCU_LAUNCH_SKIP_BEFORE_MATCH="${NCU_LAUNCH_SKIP_BEFORE_MATCH:-0}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-50}"
NCU_KERNEL_NAME="${NCU_KERNEL_NAME:-}"
NCU_KERNEL_NAME_BASE="${NCU_KERNEL_NAME_BASE:-demangled}"
NCU_METRICS="${NCU_METRICS:-}"
NCU_SECTIONS="${NCU_SECTIONS:-}"
NCU_NVTX="${NCU_NVTX:-0}"
NCU_NVTX_INCLUDE="${NCU_NVTX_INCLUDE:-}"
NCU_CLOCK_CONTROL="${NCU_CLOCK_CONTROL:-base}"
NCU_CACHE_CONTROL="${NCU_CACHE_CONTROL:-all}"

export PYTHONPATH="${PROJECT_DIR}/src/LIBERO:${OPENVLA_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="false"

mkdir -p "${RESULT_ROOT}"
cd "${OPENVLA_DIR}"

COMMON_ARGS=(
  --pretrained_checkpoint "${CHECKPOINT_PATH}"
  --task_suite_name "${TASK_SUITE}"
  --num_trials_per_task "${TRIALS}"
  --max_tasks "${MAX_TASKS}"
  --max_steps "${MAX_STEPS}"
  --warmup_action_steps "${WARMUP_ACTION_STEPS}"
  --static_patch_top_k "${STATIC_PATCH_TOP_K}"
  --static_patch_sim_threshold "${STATIC_PATCH_SIM_THRESHOLD}"
  --attention_top_k "${ATTENTION_TOP_K}"
)

run_check_only() {
  "${PYTHON_BIN}" experiments/robot/run_profiling_experiment.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "${RESULT_ROOT}/check_only" \
    --check_only True
}

run_plain() {
  local tag="$1"
  local cache_flag="$2"
  local out_dir="${RESULT_ROOT}/${tag}"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" experiments/robot/run_profiling_experiment.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "${out_dir}" \
    --use_vla_cache "${cache_flag}"
}

run_nsys() {
  local tag="$1"
  local cache_flag="$2"
  local out_dir="${RESULT_ROOT}/${tag}"
  local report_prefix="${out_dir}/nsys_${tag}"
  mkdir -p "${out_dir}"

  "${NSYS_BIN}" profile \
    --force-overwrite=true \
    --trace="${NSYS_TRACE}" \
    --sample="${NSYS_SAMPLE}" \
    --cuda-memory-usage=true \
    --output="${report_prefix}" \
    "${PYTHON_BIN}" experiments/robot/run_profiling_experiment.py \
      "${COMMON_ARGS[@]}" \
      --output_dir "${out_dir}" \
      --use_vla_cache "${cache_flag}"

  if [[ -f "${report_prefix}.nsys-rep" ]]; then
    "${NSYS_BIN}" stats \
      --force-overwrite=true \
      --report "${NSYS_STATS_REPORTS}" \
      --format csv \
      --output "${out_dir}/nsys_stats_${tag}" \
      "${report_prefix}.nsys-rep" || true
  fi
}

run_ncu() {
  local tag="$1"
  local cache_flag="$2"
  local out_dir="${RESULT_ROOT}/${tag}"
  local report_prefix="${out_dir}/ncu_${tag}"
  local report_file="${report_prefix}.ncu-rep"
  mkdir -p "${out_dir}"

  local ncu_args=(
    --force-overwrite \
    --target-processes "${NCU_TARGET_PROCESSES}" \
    --replay-mode "${NCU_REPLAY_MODE}" \
    --launch-skip "${NCU_LAUNCH_SKIP}" \
    --launch-skip-before-match "${NCU_LAUNCH_SKIP_BEFORE_MATCH}" \
    --launch-count "${NCU_LAUNCH_COUNT}" \
    --kernel-name-base "${NCU_KERNEL_NAME_BASE}" \
    --clock-control "${NCU_CLOCK_CONTROL}" \
    --cache-control "${NCU_CACHE_CONTROL}" \
    --export "${report_prefix}" \
  )

  if [[ -n "${NCU_KERNEL_NAME}" ]]; then
    ncu_args+=(--kernel-name "${NCU_KERNEL_NAME}")
  fi

  if [[ "${NCU_NVTX}" == "1" ]]; then
    ncu_args+=(--nvtx)
    if [[ -n "${NCU_NVTX_INCLUDE}" ]]; then
      ncu_args+=(--nvtx-include "${NCU_NVTX_INCLUDE}")
    fi
  fi

  if [[ -n "${NCU_METRICS}" ]]; then
    ncu_args+=(--metrics "${NCU_METRICS}")
  elif [[ -n "${NCU_SECTIONS}" ]]; then
    IFS=',' read -ra ncu_sections <<< "${NCU_SECTIONS}"
    for section in "${ncu_sections[@]}"; do
      ncu_args+=(--section "${section}")
    done
  else
    ncu_args+=(--set "${NCU_SET}")
  fi

  "${NCU_BIN}" "${ncu_args[@]}" \
    "${PYTHON_BIN}" experiments/robot/run_profiling_experiment.py \
      "${COMMON_ARGS[@]}" \
      --output_dir "${out_dir}" \
      --use_vla_cache "${cache_flag}"

  if [[ -f "${report_file}" ]]; then
    "${NCU_BIN}" --import "${report_file}" --csv --page raw --print-units base --print-fp \
      > "${out_dir}/ncu_${tag}_raw.csv" || true
    "${NCU_BIN}" --import "${report_file}" --csv --page details --print-details all --print-units base --print-fp \
      > "${out_dir}/ncu_${tag}_details.csv" || true
    "${NCU_BIN}" --import "${report_file}" --print-summary per-kernel \
      > "${out_dir}/ncu_${tag}_summary.txt" || true
  fi
}

run_one_mode() {
  local tag="$1"
  local cache_flag="$2"
  case "${PROFILE_MODE}" in
    plain) run_plain "${tag}" "${cache_flag}" ;;
    nsys) run_nsys "${tag}" "${cache_flag}" ;;
    ncu) run_ncu "${tag}" "${cache_flag}" ;;
    both)
      run_nsys "${tag}" "${cache_flag}"
      run_ncu "${tag}" "${cache_flag}"
      ;;
    *)
      echo "Unknown PROFILE_MODE=${PROFILE_MODE}. Use plain, nsys, ncu, or both." >&2
      exit 2
      ;;
  esac
}

run_check_only

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "Check passed. Nsight result root prepared at: ${RESULT_ROOT}"
  exit 0
fi

case "${CACHE_MODE}" in
  with) run_one_mode "with_cache" "True" ;;
  without) run_one_mode "without_cache" "False" ;;
  both)
    run_one_mode "with_cache" "True"
    run_one_mode "without_cache" "False"
    ;;
  *)
    echo "Unknown CACHE_MODE=${CACHE_MODE}. Use with, without, or both." >&2
    exit 2
    ;;
esac

echo "Nsight experiment done: ${RESULT_ROOT}"
