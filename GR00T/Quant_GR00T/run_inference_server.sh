#!/bin/bash
# Script to run GR00T inference server for Libero evaluation
# Usage: ./run_inference_server.sh [task_suite_name]
# task_suite_name: libero_spatial (default), libero_goal, libero_object, libero_90, libero_10

TASK=${1:-libero_10}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate groot_test when running on the local conda setup. Docker images run
# directly in the image Python environment and do not contain conda.
if [[ "${QUANTVLA_SKIP_CONDA:-0}" != "1" && -f "$HOME/miniconda3/etc/profile.d/conda.sh" && -d "$HOME/miniconda3/envs/groot_test" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate groot_test
else
    echo "[QuantVLA] Conda env groot_test not found; using current Python environment"
fi
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
python "$PROJECT_ROOT/tools/check_cuda_available.py" --context "GR00T inference server in groot_test"

# Set model path and data config based on task
case $TASK in
    libero_spatial)
        MODEL_PATH="youliangtan/gr00t-n1.5-libero-spatial-posttrain"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
        ;;
    libero_goal)
        MODEL_PATH="youliangtan/gr00t-n1.5-libero-goal-posttrain"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfigMeanStd"
        ;;
    libero_object)
        MODEL_PATH="youliangtan/gr00t-n1.5-libero-object-posttrain"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
        ;;
    libero_90)
        MODEL_PATH="youliangtan/gr00t-n1.5-libero-90-posttrain"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
        ;;
    libero_10)
        MODEL_PATH="youliangtan/gr00t-n1.5-libero-long-posttrain"
        DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
        ;;
    *)
        echo "Unknown task: $TASK"
        echo "Available tasks: libero_spatial, libero_goal, libero_object, libero_90, libero_10"
        exit 1
        ;;
esac

# Allow env preset overrides from tools/quantvla_env.sh
MODEL_PATH=${QVLA_MODEL_PATH:-$MODEL_PATH}
DATA_CONFIG=${QVLA_DATA_CONFIG:-$DATA_CONFIG}
VARIANT=${QVLA_VARIANT:-manual}

# Allow override of denoising steps via environment variable
DENOISING_STEPS=${GR00T_DENOISING_STEPS:-8}

echo "=========================================="
echo "Starting GR00T inference server for $TASK"
echo "Variant: $VARIANT"
echo "Model: $MODEL_PATH"
echo "Data Config: $DATA_CONFIG"
echo "Port: 5556"
echo "Denoising Steps: $DENOISING_STEPS"
echo "Attention: ${GR00T_ATTN_IMPLEMENTATION:-sdpa}"
echo "DuQuant enabled: $(env | grep -q '^GR00T_DUQUANT_' && echo yes || echo no)"
echo "ATM enabled: ${GR00T_ATM_ENABLE:-0}"
echo "OHB enabled: ${GR00T_OHB_ENABLE:-0}"
echo "=========================================="

cd "$PROJECT_ROOT"

python scripts/inference_service.py \
    --model_path $MODEL_PATH \
    --server \
    --data_config $DATA_CONFIG \
    --denoising-steps "$DENOISING_STEPS" \
    --port 5556 \
    --embodiment-tag new_embodiment
