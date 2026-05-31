# OpenVLA Calibration-Free Quantization

A calibration-free direct quantization inference tool for OpenVLA. Quantize the LLM component of OpenVLA to W8A16 / W8A8 / W4A16 without needing any calibration dataset, significantly reducing GPU memory usage and inference latency.

## Supported Quantization Modes

| Mode | Weight Precision | Activation Precision | Implementation |
|------|---------|---------|---------|
| `w8a16` | INT8 per-channel | BF16/FP16 | Dequantize → FP matmul |
| `w8a8` | INT8 per-channel | INT8 dynamic per-row | `torch._int_mm` hardware-accelerated |
| `w4a16` | INT4 group-wise (g=128) | BF16/FP16 | Group dequantize → FP matmul |

Quantization scope: only `nn.Linear` layers in `model.language_model`. Skips `lm_head`, embedding, norm, rotary, as well as the vision backbone, projector, and action head.

## Environment Setup

```bash
# Create conda environment
conda create -n qvla-quant python=3.10 -y
conda activate qvla-quant

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install in editable mode
cd openvla
pip install -e .
pip install -r requirements-min.txt pillow tqdm

# Optional: Flash Attention 2 for acceleration
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

## Usage

All commands are run from the `openvla/` directory.

### 1. Profiling (One-Shot Comparison)

```bash
python experiments/robot/openvla_profile.py \
  --model_path openvla/openvla-7b \
  --quant_modes w8a16,w8a8,w4a16 \
  --instruction "put the spoon on the towel" \
  --repeat_steps 100 \
  --warmup_steps 10
```

**Key Parameters:**

| Parameter | Description | Default |
|------|------|--------|
| `--model_path` | HF Hub ID or local path | `openvla/openvla-7b` |
| `--quant_modes` | Comma-separated quantization modes | `w8a8,w8a16,w4a16` |
| `--attn_implementation` | `eager` / `sdpa` / `flash_attention_2` | `eager` |
| `--quant_group_size` | W4A16 group size | `128` |
| `--repeat_steps` | Number of test iterations per mode | `100` |
| `--device` | `cuda` / `cpu` | `cuda` |
| `--profile_memory` | Whether to record peak GPU memory | `True` |

**Output:**
- Real-time per-iteration latency printed to terminal, plus a summary comparison table
- CSV/JSON saved to `openvla/out/`

Key output metrics:

| Metric | Meaning |
|------|------|
| `llm_ms_mean` | Average LLM latency |
| `model_latency_ms_mean` | Total Vision + LLM + Action latency |
| `model_size_reduction_percent` | Model size reduction percentage |
| `peak_cuda_memory_mb` | Peak GPU memory usage |

### 2. Deploy Inference Server

```bash
# Install server dependencies
pip install uvicorn fastapi json-numpy

# Start server
python vla-scripts/deploy.py \
  --openvla_path openvla/openvla-7b \
  --direct_quant_mode w8a16
```

Client call:

```python
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={
        "image": np.zeros((256, 256, 3), dtype=np.uint8),
        "instruction": "put the spoon on the towel"
    }
).json()
```

### 3. Direct API Call in Code

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from experiments.robot.openvla_direct_quant import (
    DirectQuantConfig,
    quantize_openvla_language_model,
    estimate_model_size_mb,
)

# Load model
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Quantize
report = quantize_openvla_language_model(
    model,
    DirectQuantConfig(mode="w8a16", group_size=128),
)
print(f"Replaced {report.replaced_linear_layers} layers, "
      f"size: {report.original_model_size_mb:.0f} → "
      f"{report.quantized_model_size_mb:.0f} MB "
      f"(-{report.model_size_reduction_percent:.1f}%)")

model = model.to("cuda")
model.eval()

# Inference
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
inputs = processor("In: What action should the robot take to pick up the spoon?\nOut:", image).to("cuda", dtype=torch.bfloat16)
action = model.predict_action(**inputs, do_sample=False)
```

## Docker Usage

```bash
# Build image
docker build -t qvla-quant .

# Interactive run
docker run --gpus all -it --rm qvla-quant bash

# Run profiling
docker run --gpus all --rm qvla-quant \
  python experiments/robot/openvla_profile.py \
    --model_path openvla/openvla-7b \
    --quant_modes w8a16,w8a8,w4a16

# Start inference server (with port mapping)
docker run --gpus all -p 8000:8000 --rm qvla-quant \
  python vla-scripts/deploy.py \
    --openvla_path openvla/openvla-7b \
    --direct_quant_mode w8a16
```

> **Note:** The Docker image does not include model weight files. Models are downloaded automatically from HuggingFace Hub via `--model_path`, or mounted from a local directory.

## GPU Profiling with Nsight

### Nsight Systems — End-to-End Timeline Analysis

```bash
# Use fewer iterations to keep trace size manageable
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=qvla_quant_nsys \
  --force-overwrite=true \
  python experiments/robot/openvla_profile.py \
    --model_path openvla/openvla-7b \
    --quant_modes w8a16 \
    --repeat_steps 5 \
    --warmup_steps 3 \
    --attn_implementation eager

# Open results in Nsight Systems GUI
nsys-ui qvla_quant_nsys.nsys-rep
```

### Nsight Compute — Single Kernel Analysis

```bash
# Analyze quantized INT8 dequant matmul kernel
# --kernel-name supports regex matching for custom quantized Linear ops
ncu \
  --kernel-name "linear" \
  --launch-count 20 \
  --set full \
  -o qvla_quant_ncu \
  python experiments/robot/openvla_profile.py \
    --model_path openvla/openvla-7b \
    --quant_modes w8a16 \
    --repeat_steps 1 \
    --warmup_steps 0 \
    --attn_implementation eager
```

### Manual NVTX Range Markers (insert in code)

```python
import torch.cuda.nvtx as nvtx

with nvtx.range("quantized_forward"):
    output = quantized_linear(input_tensor)
```

## Project Structure

```
openvla/
├── experiments/robot/
│   ├── openvla_direct_quant.py    # Core quantization module (W8A16/W8A8/W4A16)
│   └── openvla_profile.py         # Profiling CLI
├── prismatic/extern/hf/           # HuggingFace model integration
│   ├── configuration_prismatic.py
│   ├── modeling_prismatic.py
│   └── processing_prismatic.py
├── vla-scripts/
│   └── deploy.py                  # FastAPI inference server
├── pyproject.toml
└── requirements-min.txt
```

## Citation

```bibtex
@misc{xu2026qvlachannelsequalvisionlanguageaction,
      title={QVLA: Not All Channels Are Equal in Vision-Language-Action Model's Quantization},
      author={Yuhao Xu and Yantai Yang and Zhenyang Fan and Yufan Liu and Yuming Li and Bing Li and Zhipeng Zhang},
      year={2026},
      eprint={2602.03782},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.03782},
}
```
