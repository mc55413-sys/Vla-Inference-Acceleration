# Project Structure

`Cache_Openvla` is an OpenVLA cache-acceleration and profiling sub-project.

## Root Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview and quick start |
| `RUN_EXPERIMENT.md` | Copy-paste commands for E2E, Nsight Systems, and NCU runs |
| `PROJECT_STRUCTURE.md` | Directory map |
| `local_config.example.sh` | Template for machine-local paths |
| `Dockerfile` | CUDA/PyTorch container definition |
| `docker-compose.yml` | Docker Compose service for GPU profiling runs |
| `run_latency_experiment.sh` | Main E2E with/without cache launcher |

## Source Code

```text
src/
├── openvla/       # OpenVLA source used by profiling
├── openvla-oft/   # OpenVLA-OFT source kept from VLA-Cache
└── LIBERO/        # LIBERO benchmark source
```

Important profiling files:

| File | Purpose |
|------|---------|
| `src/openvla/experiments/robot/run_profiling_experiment.py` | Stage profiler and VLA-Cache comparison runner |
| `src/openvla/experiments/robot/generate_analysis_report.py` | Markdown report generator for profiling JSON files |
| `src/openvla/experiments/robot/openvla_utils.py` | OpenVLA loading and action helper path |
| `src/openvla/experiments/robot/vla_cache_utils.py` | Static patch detection and reuse policy utilities |

## Nsight Suite

```text
nsight_suite/
├── README_NSIGHT.md
├── run_nsight_experiment.sh
└── summarize_ncu_results.py
```

The Nsight wrapper supports:

| Mode | Purpose |
|------|---------|
| `PROFILE_MODE=plain` | Run the Python profiler without external tools |
| `PROFILE_MODE=nsys` | Capture Nsight Systems traces and CSV summaries |
| `PROFILE_MODE=ncu` | Capture Nsight Compute reports and exported CSVs |
| `PROFILE_MODE=both` | Run Nsys and NCU sequentially for small tests |

## Generated Results

Normal E2E profiling:

```text
profiling_results/run_<timestamp>/
```

Nsight profiling:

```text
nsight_results/run_<timestamp>/
```

These directories are generated artifacts and are ignored by git by default.

## Documents

```text
docs/
└── archive/       # Optional archived reports or previous analyses
```
