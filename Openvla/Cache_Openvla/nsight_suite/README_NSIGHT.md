# Nsight Profiling Suite

This is a separate profiling entrypoint. It does not replace or modify the existing `run_latency_experiment.sh` flow.

## Detected Tools

- Nsight Systems: `/opt/nvidia/nsight-systems/2026.2.1/target-linux-x64/nsys`
- Nsight Systems UI: `/opt/nvidia/nsight-systems/2026.2.1/host-linux-x64/nsys-ui`
- Nsight Compute: `/opt/nvidia/nsight-compute/2025.3.1/ncu`

CUDA also exposes `/usr/local/cuda-13.0/bin/nsys` and `/usr/local/cuda-13.0/bin/ncu`, but this suite defaults to the explicit `/opt/nvidia/...` installs.

## Quick Environment Check

```bash
CHECK_ONLY=1 bash nsight_suite/run_nsight_experiment.sh
```

## Fast Nsight Systems Smoke

```bash
PROFILE_MODE=nsys TRIALS=1 MAX_TASKS=1 MAX_STEPS=2 \
  bash nsight_suite/run_nsight_experiment.sh
```

Outputs go to `nsight_results/run_<timestamp>/`.

## Formal Nsight Systems Run

```bash
PROFILE_MODE=nsys CACHE_MODE=both TRIALS=10 MAX_TASKS=1 MAX_STEPS=220 \
  WARMUP_ACTION_STEPS=1 STATIC_PATCH_TOP_K=150 ATTENTION_TOP_K=120 \
  bash nsight_suite/run_nsight_experiment.sh
```

## Profile Only One Mode

```bash
CACHE_MODE=with PROFILE_MODE=nsys TRIALS=3 MAX_TASKS=1 bash nsight_suite/run_nsight_experiment.sh
CACHE_MODE=without PROFILE_MODE=nsys TRIALS=3 MAX_TASKS=1 bash nsight_suite/run_nsight_experiment.sh
```

## Nsight Compute

Nsight Compute can be much slower because it replays kernels and collects hardware counters in detail. This is the path to use when you want more accurate kernel-level throughput / FLOP-related indicators than the Python hook estimate.

```bash
PROFILE_MODE=ncu CACHE_MODE=with TRIALS=1 MAX_TASKS=1 MAX_STEPS=2 \
  bash nsight_suite/run_nsight_experiment.sh
```

Recommended paired comparison:

```bash
PROFILE_MODE=ncu CACHE_MODE=both TRIALS=1 MAX_TASKS=1 MAX_STEPS=2 \
  WARMUP_ACTION_STEPS=1 NCU_LAUNCH_SKIP=0 NCU_LAUNCH_COUNT=50 \
  bash nsight_suite/run_nsight_experiment.sh
```

If the run is too slow, reduce `NCU_LAUNCH_COUNT`. If you know early launches are mostly setup kernels, increase `NCU_LAUNCH_SKIP` or `NCU_LAUNCH_SKIP_BEFORE_MATCH`.

For LLM-heavy kernels only, add a regex filter:

```bash
PROFILE_MODE=ncu CACHE_MODE=both TRIALS=1 MAX_TASKS=1 MAX_STEPS=2 \
  NCU_KERNEL_NAME='regex:(gemm|gemv|matmul|attention|flash|cutlass|cublas)' \
  NCU_LAUNCH_COUNT=50 \
  bash nsight_suite/run_nsight_experiment.sh
```

For a custom hardware-counter metric set, pass `NCU_METRICS`. Example for scalar FLOP instruction counters, if supported by the GPU and NCU version:

```bash
PROFILE_MODE=ncu CACHE_MODE=both TRIALS=1 MAX_TASKS=1 MAX_STEPS=2 \
  NCU_METRICS='regex:smsp__sass_thread_inst_executed_op_(fadd|fmul|ffma|hadd|hmul|hfma|dadd|dmul|dfma).*' \
  NCU_LAUNCH_COUNT=50 \
  bash nsight_suite/run_nsight_experiment.sh
```

After a run, generate a compact Markdown summary from exported NCU CSV files:

```bash
python nsight_suite/summarize_ncu_results.py
```

The summary is written to `nsight_results/run_<timestamp>/FINAL_NCU_ANALYSIS.md` by default.

Important: NCU hardware counters require NVIDIA Performance Counter permission. On this machine, `ncu --query-metrics` currently reports `ERR_NVGPUCTRPERM` for the RTX 5090 when run as the normal user. If you see the same error during profiling, enable GPU performance counters in the NVIDIA driver settings or run NCU with sufficient admin privileges. Without this, NCU cannot provide accurate FLOP / SM / Tensor Core counter values.

Use `PROFILE_MODE=both` only for very small runs.

## Output Layout

```text
nsight_results/run_<timestamp>/
  check_only/
  with_cache/
    profiling_with_cache.json
    nsys_with_cache.nsys-rep
    nsys_stats_with_cache_*.csv
    ncu_with_cache.ncu-rep
    ncu_with_cache_raw.csv
    ncu_with_cache_details.csv
    ncu_with_cache_summary.txt
  without_cache/
    profiling_without_cache.json
    nsys_without_cache.nsys-rep
    nsys_stats_without_cache_*.csv
    ncu_without_cache.ncu-rep
    ncu_without_cache_raw.csv
    ncu_without_cache_details.csv
    ncu_without_cache_summary.txt
```

The JSON files are still produced by the existing E2E profiler, while `.nsys-rep` / `.ncu-rep` are external profiler reports.

## NCU Environment Variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `NCU_SET` | `speedOfLight` | Built-in NCU section set when `NCU_METRICS` and `NCU_SECTIONS` are empty. |
| `NCU_METRICS` | empty | Comma-separated metrics or `regex:<expr>`; highest priority. |
| `NCU_SECTIONS` | empty | Comma-separated section IDs; used when `NCU_METRICS` is empty. |
| `NCU_LAUNCH_COUNT` | `50` | Number of matching kernel launches to profile. |
| `NCU_LAUNCH_SKIP` | `0` | Skip matching launches before profiling. |
| `NCU_LAUNCH_SKIP_BEFORE_MATCH` | `0` | Skip all launches before matching filters. |
| `NCU_KERNEL_NAME` | empty | Optional exact or `regex:<expr>` kernel filter. |
| `NCU_REPLAY_MODE` | `kernel` | NCU replay mode. |
| `NCU_CLOCK_CONTROL` | `base` | Locks clocks for repeatable counter measurement. |
| `NCU_CACHE_CONTROL` | `all` | Flushes GPU caches between replay passes. |
