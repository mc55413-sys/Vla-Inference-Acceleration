# Adaptive Token Caching, Pruning & Quantization for VLA Models

A collection of optimization techniques for Vision-Language-Action (VLA) models, targeting faster inference, reduced memory footprint, and efficient deployment. The project spans two model families — **NVIDIA GR00T N1.5** and **OpenVLA 7B** — with three optimization axes: caching, pruning, and quantization.

## Project Structure

```
ATC/
├── GR00T/                          # GR00T N1.5 model optimizations
│   ├── Cache_GR00T/                # Token-level KV cache & DiT block output cache
│   ├── Prune_GR00T/                # FastV-style visual token pruning
│   └── Quant_GR00T/                # QuantVLA: DuQuant W4 + ATM + OHB quantization
│
└── Openvla/                        # OpenVLA 7B model optimizations
    ├── Prune_Openvla/              # VLA-Pruner: FastV + prefill attention + temporal pruning
    └── Quant_Openvla/              # Calibration-free W8A16/W8A8/W4A16 quantization
```

## Sub-Project Overview

### GR00T — NVIDIA GR00T N1.5

| Variant | Technique | Key Optimization |
|---------|-----------|-----------------|
| [Cache_GR00T](GR00T/Cache_GR00T/) | Adaptive Token Caching | KV cache for backbone visual tokens + DiT block-level output cache + condition K/V cache |
| [Prune_GR00T](GR00T/Prune_GR00T/) | FastV Token Pruning | Prune ~50% visual tokens after layer 2, reducing LLM computation |
| [Quant_GR00T](GR00T/Quant_GR00T/) | QuantVLA Full Quantization | DuQuant W4 + Activation Temperature Modifier (ATM) + Output Head Bias (OHB) |

**Common workflow**: Two-terminal setup — one terminal runs the inference server, the other runs the LIBERO simulation evaluation. All variants support Docker-based execution.

### OpenVLA — OpenVLA 7B

| Variant | Technique | Key Optimization |
|---------|-----------|-----------------|
| [Prune_Openvla](Openvla/Prune_Openvla/) | VLA-Pruner | FastV + prefill attention + temporal pruning for visual token reduction |
| [Quant_Openvla](Openvla/Quant_Openvla/) | Calibration-Free Quantization | Direct W8A16/W8A8/W4A16 quantization of LLM layers without calibration data |

## Prerequisites

All sub-projects require:

- **Docker** with NVIDIA Container Toolkit (recommended) **or** a local Conda installation
- **NVIDIA GPU** with CUDA support:
  - GR00T N1.5 (3B): RTX 4090 / A40 / A6000 / H100 / B200
  - GR00T N1.5 on Jetson: Orin (JetPack 6.2) / Thor (JetPack 7.0)
  - OpenVLA 7B: GPU with ≥16 GB VRAM
- **Hugging Face** access (models are downloaded on first run)

Verify your Docker + GPU setup:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### 1. GR00T-Cache 

```bash
cd GR00T/Cache_GR00T

# Build
docker build -t gr00t-cache .

# Terminal 1 — Cache-enabled server
docker run --gpus all -p 5557:5557 \
    -v $(pwd)/models:/workspace/models \
    -e REUSE=0.5 -e TOPK=5 \
    gr00t-cache run-server-cache

# Terminal 2 — Evaluation
docker run --gpus all --network host \
    -v $(pwd)/results:/workspace/results \
    -e HOST=localhost -e PORT=5557 \
    gr00t-cache run-eval
```

→ See [GR00T/Cache_GR00T/README.md](GR00T/Cache_GR00T/README.md) for full details.

### 2. GR00T-Pruning 

```bash
cd GR00T/Prune_GR00T

# Build
docker build -t prune-gr00t-fastv:latest .

# Terminal 1 — Server with 50% visual token pruning
docker run --gpus all -it --rm --network host \
    -v "$PWD/results:/workspace/Prune_GR00T/results" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    prune-gr00t-fastv:latest \
    bash -lc "GR00T_FASTV_ENABLE=1 GR00T_FASTV_K=2 GR00T_FASTV_R=0.5 ./run_inference_server.sh libero_10"

# Terminal 2 — Evaluation
docker run --gpus all -it --rm --network host \
    -v "$PWD/results:/workspace/Prune_GR00T/results" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    prune-gr00t-fastv:latest \
    bash -lc "./run_libero_eval.sh libero_10 --headless --profile-server --print_step_latency --output-json results/libero_fastv50.json"
```

→ See [GR00T/Prune_GR00T/README.md](GR00T/Prune_GR00T/README.md) for full details.

### 3. GR00T-Quantization 

```bash
cd GR00T/Quant_GR00T

# Build
docker build -t quantvla:cuda13 .

# Terminal 1 — Server with full quantization
docker run --rm -it --gpus all --network host --ipc host \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "$PWD/results:/workspace/QuantVLA/results" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    quantvla:cuda13 \
    tools/run_quantvla_inference_server.sh full libero_10

# Terminal 2 — Evaluation
docker run --rm -it --gpus all --network host --ipc host \
    -e QVLA_HOST=localhost \
    -v "$PWD/results:/workspace/QuantVLA/results" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    quantvla:cuda13 \
    tools/run_quantvla_real_libero_benchmark.sh full libero_10
```

→ See [GR00T/Quant_GR00T/README.md](GR00T/Quant_GR00T/README.md) for full details.

### 4. OpenVLA-Pruning 

```bash
cd Openvla/Prune_Openvla/Openvla

# Build
docker-compose build

# Run evaluation (LIBERO-Spatial, VLA-Pruner + prefill attention)
docker-compose run --rm vlapruner \
    python experiments/robot/libero/run_libero_eval.py \
        --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial \
        --use_fastv True \
        --use_prefil_attention True \
        --use_temporal True \
        --fastv_r 0.75 \
        --seed 7 \
        --run_id_note vlapruner_prefill_25% \
        --num_trials_per_task 50
```

→ See [Openvla/Prune_Openvla/Openvla/README.md](Openvla/Prune_Openvla/Openvla/README.md) for full details.

### 5. OpenVLA-Quantization 

```bash
cd Openvla/Quant_Openvla

# One-shot profiling comparing all modes
python experiments/robot/openvla_profile.py \
    --model_path openvla/openvla-7b \
    --quant_modes w8a16,w8a8,w4a16 \
    --instruction "put the spoon on the towel" \
    --repeat_steps 100 \
    --warmup_steps 10 \
    --attn_implementation eager

# Or via Docker
docker build -t qvla-quant .
docker run --gpus all --rm qvla-quant \
    python experiments/robot/openvla_profile.py \
        --model_path openvla/openvla-7b \
        --quant_modes bf16,w8a16,w8a8,w4a16 \
        --attn_implementation eager
```

→ See [Openvla/Quant_Openvla/README.md](Openvla/Quant_Openvla/README.md) for full details.

## Benchmark Tasks

All sub-projects use the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) manipulation benchmark with five task suites:

| Task Suite | Description | # Tasks |
|-----------|-------------|---------|
| `libero_spatial` | Spatial relationship transfer | 10 |
| `libero_object` | Object-centric knowledge transfer | 10 |
| `libero_goal` | Goal-directed behavior transfer | 10 |
| `libero_90` | Pretraining (90 tasks) | 90 |
| `libero_10` | Lifelong learning evaluation (10 tasks) | 10 |

## Latency Metrics

All GR00T variants output per-step latency breakdowns:

```
[latency] Data=... | Preprocess=... | System-2 Vision=... | System-2 Reasoning=... | 
          Bridge=... | System-1 Vision=... | System-1 Action=... | 
          End to End Latency=... | Model Latency=...
```

| Stage | Description |
|-------|-------------|
| Data | Observation ingestion |
| Preprocess | Image resize + tokenization + CPU→GPU transfer |
| System-2 Vision | Vision encoder + projector |
| System-2 Reasoning | LLM backbone forward pass |
| System-1 Action | DiT denoising loop (action head) |
| End to End | Total wall-clock latency |
| Model Latency | Vision + Reasoning + Action (GPU-only) |




