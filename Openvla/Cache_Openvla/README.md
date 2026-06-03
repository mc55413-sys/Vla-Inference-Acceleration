# Cache_Openvla: VLA-Cache Profiling for OpenVLA

Cache_Openvla is an OpenVLA 7B profiling workspace for evaluating **VLA-Cache** on LIBERO. It extends the VLA-Cache/OpenVLA evaluation path with reproducible end-to-end measurements, stage-level latency breakdowns, and theoretical FLOPs estimates for cache-enabled and cache-disabled inference.

## Overview

VLA-Cache accelerates repeated robot-control inference by identifying stable visual patches across frames and reusing selected visual token computations. This project focuses on OpenVLA and reports performance across the full inference pipeline rather than only CUDA-layer latency.

| Stage | Meaning |
|------|---------|
| E2E | Observation to continuous action ready |
| Data | LIBERO observation to model tensors and prompt |
| Vision | Image tensor to vision backbone/projector output |
| LLM | Multimodal prefix / language backbone forward |
| Action | Action token decoding and token-to-continuous postprocess |

`TFLOPs/sample` follows the original VLA-Cache convention for multi-token Llama decoder forwards:

```text
4*n*d^2 + 2*n^2*d + 3*n*d*m
```

This is a theoretical per-sample FLOPs estimate, not hardware throughput, and should not be summed across stages.

## Environment Setup

Create or activate a Python environment with OpenVLA, PyTorch, LIBERO, robosuite, and the VLA-Cache-compatible Transformers package installed.

Configure local paths by copying the example config:

```bash
cp local_config.example.sh local_config.sh
```

Then edit `local_config.sh`:

```bash
PYTHON_BIN="/path/to/env/bin/python"
CHECKPOINT_PATH="/path/to/openvla-7b-finetuned-libero-spatial"
```

`local_config.sh` is ignored by git. You can also pass `PYTHON_BIN` and `CHECKPOINT_PATH` as environment variables for a single run.

## Usage

### 1. Check Environment

```bash
CHECK_ONLY=1 bash run_latency_experiment.sh
```

This verifies that the checkpoint exists and that the installed Llama implementation includes the VLA-Cache hooks.

### 2. End-to-End Profiling

```bash
RESULTS_DIR=profiling_results/run_$(date +%Y%m%d_%H%M%S) \
  TRIALS=10 MAX_TASKS=1 MAX_STEPS=220 \
  WARMUP_ACTION_STEPS=1 STATIC_PATCH_TOP_K=150 ATTENTION_TOP_K=120 \
  bash run_latency_experiment.sh
```

Outputs:

```text
profiling_results/run_<timestamp>/
├── profiling_with_cache.json
├── profiling_without_cache.json
└── PERFORMANCE_ANALYSIS.md
```

### 3. Nsight Systems Profiling

```bash
PROFILE_MODE=nsys CACHE_MODE=both TRIALS=10 MAX_TASKS=1 MAX_STEPS=220 \
  WARMUP_ACTION_STEPS=1 STATIC_PATCH_TOP_K=150 ATTENTION_TOP_K=120 \
  bash nsight_suite/run_nsight_experiment.sh
```

See [nsight_suite/README_NSIGHT.md](nsight_suite/README_NSIGHT.md) for Nsight Systems and Nsight Compute options.

## Docker Usage

The Docker image is intended for reproducible profiling on machines with NVIDIA Container Toolkit installed. Model checkpoints are not baked into the image; mount them at runtime.

Build the image:

```bash
docker build -t cache-openvla .
```

Run the environment check:

```bash
docker run --rm --gpus all --ipc host --network host \
  -v "$PWD:/workspace/Cache_Openvla" \
  -v "/path/to/checkpoints:/workspace/checkpoints:ro" \
  -e CHECKPOINT_PATH=/workspace/checkpoints/openvla-7b-finetuned-libero-spatial \
  cache-openvla \
  "CHECK_ONLY=1 bash run_latency_experiment.sh"
```

Run an end-to-end comparison:

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

For Nsight profiling in Docker, install Nsight tools in the image or mount host Nsight binaries and pass `NSYS_BIN` / `NCU_BIN` as environment variables.

## Key Parameters

| Variable | Description | Default |
|------|------|------|
| `CHECKPOINT_PATH` | Local OpenVLA checkpoint path | `checkpoints/openvla-7b-finetuned-libero-spatial` |
| `PYTHON_BIN` | Python executable used by scripts | `python` |
| `TRIALS` | Trials per LIBERO task | `1` |
| `MAX_TASKS` | Number of LIBERO tasks | `1` |
| `MAX_STEPS` | Max action steps per episode | suite default for E2E, `2` for Nsight wrapper |
| `WARMUP_ACTION_STEPS` | Real action steps excluded from metrics | `1` |
| `STATIC_PATCH_TOP_K` | Stable patches considered for reuse | `150` |
| `ATTENTION_TOP_K` | Task-relevant visual tokens protected from reuse | `120` |
| `CACHE_MODE` | Nsight mode: `with`, `without`, `both` | `both` |
| `PROFILE_MODE` | `plain`, `nsys`, `ncu`, `both` | `nsys` |

## Project Structure

```text
Cache_Openvla/
├── README.md
├── RUN_EXPERIMENT.md
├── PROJECT_STRUCTURE.md
├── local_config.example.sh
├── Dockerfile
├── docker-compose.yml
├── run_latency_experiment.sh
├── nsight_suite/
│   ├── README_NSIGHT.md
│   ├── run_nsight_experiment.sh
│   └── summarize_ncu_results.py
├── docs/
└── src/
    ├── openvla/
    ├── openvla-oft/
    └── LIBERO/
```

Generated artifacts are written to `profiling_results/`, `nsight_results/`, or `docker_results/`. These directories are ignored by git.

## Citation

```bibtex
@article{xu2025vla,
  title={VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching},
  author={Xu, Siyu and Wang, Yunke and Xia, Chenghao and Zhu, Dihao and Huang, Tao and Xu, Chang},
  journal={arXiv preprint arXiv:2502.02175},
  year={2025}
}
```
