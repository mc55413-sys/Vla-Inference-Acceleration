#!/usr/bin/env bash
# Source this file to switch QuantVLA benchmark variants.
#
# Example:
#   source tools/quantvla_env.sh
#   quantvla_use_baseline libero_10
#   quantvla_use_full libero_10

export QUANTVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

quantvla_clear_quant_env() {
  local key
  while IFS= read -r key; do unset "$key"; done < <(compgen -e GR00T_DUQUANT_)
  while IFS= read -r key; do unset "$key"; done < <(compgen -e GR00T_ATM_)
  while IFS= read -r key; do unset "$key"; done < <(compgen -e GR00T_OHB_)
}

quantvla_select_task() {
  local task="${1:-libero_10}"
  export QVLA_TASK="$task"
  export QVLA_PORT="${QVLA_PORT:-5556}"
  export QVLA_DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"

  case "$task" in
    libero_spatial)
      export QVLA_MODEL_PATH="youliangtan/gr00t-n1.5-libero-spatial-posttrain"
      export QVLA_ATM_ALPHA_PATH="$QUANTVLA_ROOT/atm_alpha_beta_spatial.json"
      ;;
    libero_goal)
      export QVLA_MODEL_PATH="youliangtan/gr00t-n1.5-libero-goal-posttrain"
      export QVLA_DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfigMeanStd"
      export QVLA_ATM_ALPHA_PATH="$QUANTVLA_ROOT/atm_alpha_beta_goal.json"
      ;;
    libero_object)
      export QVLA_MODEL_PATH="youliangtan/gr00t-n1.5-libero-object-posttrain"
      export QVLA_ATM_ALPHA_PATH="$QUANTVLA_ROOT/atm_alpha_beta_object.json"
      ;;
    libero_90)
      export QVLA_MODEL_PATH="youliangtan/gr00t-n1.5-libero-90-posttrain"
      export QVLA_ATM_ALPHA_PATH="$QUANTVLA_ROOT/atm_alpha_beta_long.json"
      ;;
    libero_10)
      export QVLA_MODEL_PATH="youliangtan/gr00t-n1.5-libero-long-posttrain"
      export QVLA_ATM_ALPHA_PATH="$QUANTVLA_ROOT/atm_alpha_beta_long.json"
      ;;
    *)
      echo "Unknown task: $task" >&2
      return 1
      ;;
  esac
}

quantvla_use_baseline() {
  local task="${1:-${QVLA_TASK:-libero_10}}"
  quantvla_clear_quant_env
  quantvla_select_task "$task" || return 1
  export QVLA_VARIANT="baseline_bf16_sdpa"
  export GR00T_ATTN_IMPLEMENTATION="${GR00T_ATTN_IMPLEMENTATION:-sdpa}"
  export NO_ALBUMENTATIONS_UPDATE=1
  echo "[QuantVLA] variant=$QVLA_VARIANT task=$QVLA_TASK model=$QVLA_MODEL_PATH"
}

quantvla_use_duquant() {
  local task="${1:-${QVLA_TASK:-libero_10}}"
  quantvla_use_baseline "$task" || return 1
  export QVLA_VARIANT="duquant_w4_packed"
  export GR00T_DUQUANT_SCOPE=""
  export GR00T_DUQUANT_INCLUDE='.*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*'
  export GR00T_DUQUANT_EXCLUDE='(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)'
  export GR00T_DUQUANT_WBITS_DEFAULT=4
  export GR00T_DUQUANT_ABITS=0
  export GR00T_DUQUANT_BLOCK=64
  export GR00T_DUQUANT_PERMUTE=0
  export GR00T_DUQUANT_ROW_ROT=restore
  export GR00T_DUQUANT_ACT_PCT=99.9
  export GR00T_DUQUANT_CALIB_STEPS=32
  export GR00T_DUQUANT_LS=0.15
  export GR00T_DUQUANT_STORAGE=packed
  export GR00T_DUQUANT_ACT_MODE=off
  export GR00T_DUQUANT_PACKDIR="$QUANTVLA_ROOT/results/duquant_pack/${QVLA_TASK}"
  echo "[QuantVLA] variant=$QVLA_VARIANT task=$QVLA_TASK packdir=$GR00T_DUQUANT_PACKDIR"
}

quantvla_use_full() {
  local task="${1:-${QVLA_TASK:-libero_10}}"
  quantvla_use_duquant "$task" || return 1
  export QVLA_VARIANT="quantvla_full_w4_packed"
  export GR00T_ATM_ALPHA_PATH="$QVLA_ATM_ALPHA_PATH"
  export GR00T_ATM_ENABLE=1
  export GR00T_ATM_SCOPE="${GR00T_ATM_SCOPE:-dit}"
  export GR00T_OHB_ENABLE=1
  export GR00T_OHB_SCOPE="${GR00T_OHB_SCOPE:-dit}"
  export GR00T_OHB_FALLBACK="${GR00T_OHB_FALLBACK:-1.0}"
  echo "[QuantVLA] variant=$QVLA_VARIANT task=$QVLA_TASK alpha=$GR00T_ATM_ALPHA_PATH"
}
