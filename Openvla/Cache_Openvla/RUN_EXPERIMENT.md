# Experiment Guide

This guide contains copy-paste commands for running OpenVLA/VLA-Cache profiling experiments.

## Local Configuration

Create a local configuration file:

```bash
cp local_config.example.sh local_config.sh
```

Edit it for your machine:

```bash
PYTHON_BIN="/path/to/env/bin/python"
CHECKPOINT_PATH="/path/to/openvla-7b-finetuned-libero-spatial"
```

The scripts also accept these values as environment variables.

## Environment Check

```bash
CHECK_ONLY=1 bash run_latency_experiment.sh
```

This verifies the checkpoint path, Python environment, and VLA-Cache Llama hooks.

## Quick Smoke Test

```bash
WARMUP_ACTION_STEPS=0 TRIALS=1 MAX_TASKS=1 MAX_STEPS=1 \
  bash run_latency_experiment.sh
```

## Formal End-to-End Run

```bash
RESULTS_DIR=profiling_results/run_$(date +%Y%m%d_%H%M%S) \
  TRIALS=10 MAX_TASKS=1 MAX_STEPS=220 \
  WARMUP_ACTION_STEPS=1 STATIC_PATCH_TOP_K=150 ATTENTION_TOP_K=120 \
  bash run_latency_experiment.sh
```

The script runs both modes:

1. `--use_vla_cache True`
2. `--use_vla_cache False`
3. Markdown report generation

Output layout:

```text
profiling_results/run_<timestamp>/
├── profiling_with_cache.json
├── profiling_without_cache.json
└── PERFORMANCE_ANALYSIS.md
```

## Nsight Systems Run

```bash
PROFILE_MODE=nsys CACHE_MODE=both TRIALS=10 MAX_TASKS=1 MAX_STEPS=220 \
  WARMUP_ACTION_STEPS=1 STATIC_PATCH_TOP_K=150 ATTENTION_TOP_K=120 \
  bash nsight_suite/run_nsight_experiment.sh
```

Output layout:

```text
nsight_results/run_<timestamp>/
├── with_cache/
│   ├── profiling_with_cache.json
│   ├── nsys_with_cache.nsys-rep
│   ├── nsys_with_cache.sqlite
│   └── nsys_stats_with_cache_*.csv
└── without_cache/
    ├── profiling_without_cache.json
    ├── nsys_without_cache.nsys-rep
    ├── nsys_without_cache.sqlite
    └── nsys_stats_without_cache_*.csv
```

## Nsight Compute Run

Use Nsight Compute for kernel-level hardware counter details:

```bash
PROFILE_MODE=ncu CACHE_MODE=both TRIALS=1 MAX_TASKS=1 MAX_STEPS=2 \
  WARMUP_ACTION_STEPS=1 STATIC_PATCH_TOP_K=150 ATTENTION_TOP_K=120 \
  NCU_LAUNCH_COUNT=50 \
  bash nsight_suite/run_nsight_experiment.sh
```

If NVIDIA performance counter permissions are disabled, run with sufficient privileges or enable unrestricted GPU performance counters in the driver configuration.

## Docker Run

Build the image:

```bash
docker build -t cache-openvla .
```

Run the environment check. Mount the directory containing `openvla-7b-finetuned-libero-spatial` at `/workspace/checkpoints`:

```bash
docker run --rm --gpus all --ipc host --network host \
  -v "$PWD:/workspace/Cache_Openvla" \
  -v "/path/to/checkpoints:/workspace/checkpoints:ro" \
  -e CHECKPOINT_PATH=/workspace/checkpoints/openvla-7b-finetuned-libero-spatial \
  cache-openvla \
  "CHECK_ONLY=1 bash run_latency_experiment.sh"
```

Run a full E2E comparison:

```bash
docker run --rm --gpus all --ipc host --network host \
  -v "$PWD:/workspace/Cache_Openvla" \
  -v "/path/to/checkpoints:/workspace/checkpoints:ro" \
  -v "$PWD/docker_results:/workspace/Cache_Openvla/docker_results" \
  -e CHECKPOINT_PATH=/workspace/checkpoints/openvla-7b-finetuned-libero-spatial \
  cache-openvla \
  "RESULTS_DIR=docker_results/run_$(date +%Y%m%d_%H%M%S) TRIALS=10 MAX_TASKS=1 MAX_STEPS=220 bash run_latency_experiment.sh"
```

Docker Compose shortcut:

```bash
CHECKPOINT_DIR=/path/to/checkpoints docker compose run --rm cache-openvla \
  "CHECK_ONLY=1 bash run_latency_experiment.sh"
```

If your Docker installation uses the legacy Compose binary, replace `docker compose` with `docker-compose`.

## Metrics

| Stage | JSON key |
|------|------|
| E2E | `e2e_observation_to_action_ready` |
| Data | `data_observation_to_model_tensors_prompt` |
| Vision | `vision_image_tensor_to_projector_output` |
| LLM | `llm_multimodal_prefix_forward` |
| Action | `action_decode_or_denoise_to_continuous` |

Per-step cache diagnostics are stored under `per_step`:

| Field | Meaning |
|------|------|
| `recorded` | Whether the step is included in summary metrics |
| `used_vla_cache` | Whether previous KV cache was used |
| `stable_patch_count` | Stable visual patches before task filtering |
| `reusable_token_count` | Final reusable visual tokens |
| `llm_prefix_tflops_sample` | Per-step LLM prefix theoretical TFLOPs/sample |

## Result Analysis

Use `src/openvla/experiments/robot/generate_analysis_report.py` for profiling JSON summaries. For Nsight CSV summaries, use `nsight_suite/summarize_ncu_results.py` or inspect the generated `nsys_stats_*` files directly.
