#!/bin/bash
# Script to run Libero evaluation
# Usage: ./run_libero_eval.sh [task_suite_name] [extra args...]
# task_suite_name: libero_spatial (default), libero_goal, libero_object, libero_90, libero_10

TASK=${1:-libero_10}
shift || true
EXTRA_ARGS=("$@")
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBERO_ROOT="$PROJECT_ROOT/LIBERO"

# Keep user-provided relative output paths rooted at the project directory even
# though the eval entrypoint is launched from examples/Libero/eval.
for i in "${!EXTRA_ARGS[@]}"; do
    if [[ "${EXTRA_ARGS[$i]}" == "--output-json" ]]; then
        next_idx=$((i + 1))
        if [[ $next_idx -lt ${#EXTRA_ARGS[@]} && "${EXTRA_ARGS[$next_idx]}" != /* ]]; then
            EXTRA_ARGS[$next_idx]="$PROJECT_ROOT/${EXTRA_ARGS[$next_idx]}"
        fi
    elif [[ "${EXTRA_ARGS[$i]}" == --output-json=* ]]; then
        output_path="${EXTRA_ARGS[$i]#--output-json=}"
        if [[ "$output_path" != /* ]]; then
            EXTRA_ARGS[$i]="--output-json=$PROJECT_ROOT/$output_path"
        fi
    fi
done

HEADLESS_FLAG="no"
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "--headless" ]]; then
        HEADLESS_FLAG="yes"
        break
    fi
done

# Activate libero_test environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate libero_test

# Add this pruning project and LIBERO to Python path
export PYTHONPATH="$PROJECT_ROOT:$LIBERO_ROOT:$PYTHONPATH"
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-cache}

echo "=========================================="
echo "Running Libero evaluation for $TASK"
echo "Headless mode: $HEADLESS_FLAG"
echo "Port: 5556 (GR00T)"
echo "=========================================="
echo ""
echo "Make sure the inference server is running in another terminal!"
echo "Run: ./run_inference_server.sh $TASK"
echo ""
echo "Results will be saved to:"
echo "  - Log: /tmp/logs/libero_eval_${TASK}.log"
echo "  - Videos: /tmp/logs/rollout_*.mp4"
for i in "${!EXTRA_ARGS[@]}"; do
    if [[ "${EXTRA_ARGS[$i]}" == "--output-json" ]]; then
        next_idx=$((i + 1))
        if [[ $next_idx -lt ${#EXTRA_ARGS[@]} ]]; then
            echo "  - JSON: ${EXTRA_ARGS[$next_idx]}"
        fi
    elif [[ "${EXTRA_ARGS[$i]}" == --output-json=* ]]; then
        echo "  - JSON: ${EXTRA_ARGS[$i]#--output-json=}"
    fi
done
echo "=========================================="
echo ""

cd "$PROJECT_ROOT/examples/Libero/eval"

python run_libero_eval.py --task_suite_name "$TASK" --port 5556 "${EXTRA_ARGS[@]}"
