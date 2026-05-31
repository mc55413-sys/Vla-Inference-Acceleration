#!/usr/bin/env bash
# ============================================================================
# GR00T-Cache: Run scripts for testing, benchmarking, and ablation.
#
# Usage:
#   ./run_gr00t_cache.sh test         # Run end-to-end tests
#   ./run_gr00t_cache.sh benchmark    # Run benchmark with dummy model
#   ./run_gr00t_cache.sh ablation     # Run full ablation study
#   ./run_gr00t_cache.sh compare      # Run baseline vs cached comparison
#   ./run_gr00t_cache.sh profile      # Profile with detailed timing
#   ./run_gr00t_cache.sh all          # Run everything
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# ── Configuration ────────────────────────────────────────────────────────

DEVICE="${GR00T_DEVICE:-cuda}"
WARMUP="${GR00T_WARMUP:-5}"
ITERS="${GR00T_ITERS:-50}"
OUTPUT_DIR="${GR00T_OUTPUT_DIR:-./gr00t_cache_results}"
CACHE_MODE="${GR00T_CACHE_MODE:-full_cache}"
MAX_REUSE="${GR00T_MAX_REUSE:-0.5}"
TASK_TOPK="${GR00T_TASK_TOPK:-}"
ENTROPY_SCALE="${GR00T_ENTROPY_SCALE:-1.0}"
DENOISING_STEPS="${GR00T_DENOISING_STEPS:-4}"

# ── Color output ──────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Environment check ─────────────────────────────────────────────────────

check_env() {
    info "Checking environment..."
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || {
        error "PyTorch not available. Install: pip install torch"
        exit 1
    }
    python -c "import numpy; print(f'NumPy {numpy.__version__}')" || {
        error "NumPy not available. Install: pip install numpy"
        exit 1
    }
    ok "Environment OK"

    mkdir -p "$OUTPUT_DIR"
    ok "Output directory: $OUTPUT_DIR"
}

# ── Build common args ─────────────────────────────────────────────────────

build_args() {
    local args="--device $DEVICE --warmup $WARMUP --iters $ITERS --output $OUTPUT_DIR/comparison.json"
    args="$args --cache-mode $CACHE_MODE --max-reuse $MAX_REUSE --entropy-scale $ENTROPY_SCALE"
    args="$args --denoising-steps $DENOISING_STEPS"
    if [ -n "$TASK_TOPK" ]; then
        args="$args --task-topk $TASK_TOPK"
    fi
    echo "$args"
}

# ── Commands ──────────────────────────────────────────────────────────────

cmd_test() {
    info "Running end-to-end tests..."
    python run_gr00t_cache_e2e.py --device "$DEVICE" "$@"
    ok "Tests complete"
}

cmd_benchmark() {
    info "Running benchmark (dummy model)..."
    python run_gr00t_cache_example.py $(build_args) "$@"
    ok "Benchmark complete"
}

cmd_ablation() {
    info "Running full ablation study..."
    info "This will test 7 cache configurations:"
    info "  1. baseline (no cache)"
    info "  2. static_only"
    info "  3. static_plus_task_eviction"
    info "  4. static_plus_layer_adaptive"
    info "  5. full_gr00t_cache"
    info "  6. action_head_condition_cache_only"
    info "  7. backbone_visual_cache_only"
    python run_gr00t_cache_example.py --ablation $(build_args) "$@"
    ok "Ablation complete"
    info "Results saved to $OUTPUT_DIR/"
}

cmd_compare() {
    info "Running baseline vs cached comparison..."
    info "Cache mode: $CACHE_MODE, max_reuse: $MAX_REUSE, iters: $ITERS"
    python run_gr00t_cache_example.py --strict $(build_args) "$@"
    ok "Comparison complete"
}

cmd_profile() {
    info "Running detailed profiling..."
    python -c "
import sys; sys.path.insert(0, '.')
from gr00t_cache.dummy_model import create_dummy_gr00t_model, create_dummy_observation
from gr00t_cache.profiling import profile_gr00t_cache, summarize_profile
import torch

device = '$DEVICE'
model = create_dummy_gr00t_model(device=device)
obs = [create_dummy_observation(seed=i) for i in range($((WARMUP + ITERS)))]

def policy_fn(o):
    import time
    start = time.perf_counter()
    result = model.get_action(o)
    elapsed = (time.perf_counter() - start) * 1000
    return result, {'model_total_ms': elapsed}

results = profile_gr00t_cache(
    policy_fn, obs,
    warmup_steps=$WARMUP, repeat_steps=$ITERS,
    profile_memory=True,
    output_dir='$OUTPUT_DIR/profile',
)
summary_text, stats = summarize_profile(results)
print(summary_text)
"
    ok "Profile complete"
}

cmd_all() {
    info "Running full GR00T-Cache test suite..."
    echo ""
    cmd_test
    echo ""
    cmd_compare
    echo ""
    cmd_profile
    echo ""
    ok "All tests complete!"
}

cmd_help() {
    echo "GR00T-Cache Runner"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  test         Run end-to-end tests (dummy model)"
    echo "  benchmark    Run benchmark with dummy model"
    echo "  ablation     Run full 7-config ablation study"
    echo "  compare      Run baseline vs cached comparison with correctness check"
    echo "  profile      Run detailed profiling"
    echo "  all          Run everything (test + compare + profile)"
    echo "  help         Show this help"
    echo ""
    echo "Environment variables:"
    echo "  GR00T_DEVICE          Device (cuda|cpu), default: cuda"
    echo "  GR00T_WARMUP          Warmup steps, default: 5"
    echo "  GR00T_ITERS           Measurement steps, default: 50"
    echo "  GR00T_OUTPUT_DIR      Output directory, default: ./gr00t_cache_results"
    echo "  GR00T_CACHE_MODE      Cache mode, default: full_cache"
    echo "  GR00T_MAX_REUSE       Max reuse ratio, default: 0.5"
    echo "  GR00T_TASK_TOPK       Task-relevant top-k, default: (off)"
    echo "  GR00T_ENTROPY_SCALE   Entropy scale, default: 1.0"
    echo "  GR00T_DENOISING_STEPS Denoising steps, default: 4"
    echo ""
    echo "Examples:"
    echo "  $0 test"
    echo "  $0 benchmark"
    echo "  GR00T_MAX_REUSE=0.7 GR00T_TASK_TOPK=5 $0 compare"
    echo "  GR00T_DEVICE=cpu GR00T_ITERS=20 $0 test"
}

# ── Main ──────────────────────────────────────────────────────────────────

case "${1:-help}" in
    test)      check_env; cmd_test "${@:2}" ;;
    benchmark) check_env; cmd_benchmark "${@:2}" ;;
    ablation)  check_env; cmd_ablation "${@:2}" ;;
    compare)   check_env; cmd_compare "${@:2}" ;;
    profile)   check_env; cmd_profile "${@:2}" ;;
    all)       check_env; cmd_all "${@:2}" ;;
    help|--help|-h) cmd_help ;;
    *)         error "Unknown command: ${1:-}"; cmd_help; exit 1 ;;
esac
