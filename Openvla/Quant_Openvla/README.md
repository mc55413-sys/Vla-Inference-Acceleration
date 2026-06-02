# OpenVLA Calibration-Free Quantization

A calibration-free direct quantization inference tool for OpenVLA. Quantize the LLM component of OpenVLA without needing any calibration dataset. Use `w4a8` (FP8 tensorwise) for latency-oriented quantization; `w8a16` for memory savings with exact accuracy; `bnb_int8` / `bnb_nf4` are useful baselines.

## Supported Quantization Modes

| Mode | Weight Precision | Activation Precision | Implementation |
|------|---------|---------|---------|
| `w4a8` / `fp8` | FP8 tensorwise | FP8 dynamic | `torch._scaled_mm` tensor-core accelerated |
| `bnb_int8` | INT8 + optional FP16 outliers | FP16/FP8 mixed | bitsandbytes `Linear8bitLt` fused backend |
| `bnb_nf4` | NF4 / 4-bit packed | BF16/FP16 compute | bitsandbytes `Linear4bit` fused backend |
| `bnb_fp4` | FP4 / 4-bit packed | BF16/FP16 compute | bitsandbytes `Linear4bit` fused backend |
| `w8a16` | INT8 per-channel | BF16/FP16 | Dequantize → FP matmul |
| `w8a8` | INT8 per-channel | INT8 dynamic per-row | `torch._int_mm` hardware-accelerated |
| `w4a16` | INT4 group-wise (g=128) | BF16/FP16 | Reference-only: group dequantize → FP matmul |

**Note:** `w4a8` is now implemented via tensorwise FP8 quantization (`FP8TensorwiseLinear`) using `torch._scaled_mm` on CUDA. It uses FP8-E4M3 for both weights and activations with tensor-wise scaling. On GPUs without FP8 support it falls back to dequantized BF16 matmul.

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
pip install "bitsandbytes>=0.43.0"  # required for bnb_int8 / bnb_nf4 / bnb_fp4

# Optional: Flash Attention 2 for acceleration
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

## Usage

All local commands are run from the `openvla/` source directory:

```bash
cd /path/to/Openvla/Quant_Openvla/openvla
```

### 1. Profiling (One-Shot Comparison)

```bash
conda run --no-capture-output -n qvla-quant python -u experiments/robot/openvla_profile.py \
  --model_path openvla/openvla-7b \
  --quant_modes fp8,bnb_nf4,bnb_fp4 \
  --repeat_steps 100 \
  --warmup_steps 10
```

Full quantization comparison:

```bash
conda run --no-capture-output -n qvla-quant python -u experiments/robot/openvla_profile.py \
  --model_path openvla/openvla-7b \
  --quant_modes none,w8a16,w8a8,w4a16,fp8,bnb_int8,bnb_nf4,bnb_fp4 \
  --instruction "put the spoon on the towel" \
  --repeat_steps 100 \
  --warmup_steps 10
```

**Key Parameters:**

| Parameter | Description | Default |
|------|------|--------|
| `--model_path` | HF Hub ID or local path | `openvla/checkpoints/openvla-7b-finetuned-libero-spatial` |
| `--quant_modes` | Comma-separated quantization modes | `w4a8` (→ fp8) |
| `--attn_implementation` | `auto` / `sdpa` / `eager` / `flash_attention_2`; `auto` falls back to `sdpa` when `flash_attn` is missing | `auto` |
| `--quant_group_size` | W4A16 group size | `128` |
| `--min_linear_weight_numel` | Minimum weight size to quantize (skip tiny layers) | `0` |
| `--bnb_int8_threshold` | bitsandbytes LLM.int8 outlier threshold; `0.0` is fastest, `6.0` is HF's accuracy-oriented default | `0.0` |
| `--bnb_4bit_compute_dtype` | bitsandbytes 4-bit compute dtype: `auto` / `bf16` / `fp16` / `fp32` | `auto` |
| `--bnb_4bit_quant_type` | Quant type used by `bnb_4bit` alias: `nf4` / `fp4`; explicit modes are `bnb_nf4` and `bnb_fp4` | `nf4` |
| `--bnb_4bit_use_double_quant` | Compress 4-bit quantization statistics with double quantization | `True` |
| `--fp8_activation_scale` | Fixed activation scale used by `fp8` / `w4a8` tensorwise mode | `1.0` |
| `--fast_action_head` | Non-quant optimization: only project action-bin logits instead of full vocab logits | `False` |
| `--last_token_logits` | Exact latency optimization: run backbone directly and apply `lm_head` only to the final hidden state | `False` |
| `--drop_full_attention_mask` | Drop all-ones attention masks; keeps semantics when there is no padding | `True` |
| `--max_vision_tokens` | Non-quant pruning/visual-token reduction; keep `0` for pure quantization comparisons | `0` |
| `--vision_token_strategy` | `uniform` / `pool` / `first` visual-token reduction strategy | `uniform` |
| `--llm_max_layers` | Non-quant layer dropping for aggressive speedups; keep `0` for pure quantization comparisons | `0` |
| `--llm_layer_strategy` | Layer selection strategy when `llm_max_layers > 0`: `first` / `uniform` / `last` | `first` |
| `--compile_llm` | Whether to `torch.compile` the language backbone (experimental) | `False` |
| `--compile_mode` | torch.compile mode: `reduce-overhead` or `default` | `reduce-overhead` |
| `--compressed_model_size_mb` | Override model size estimate in MB | `None` (auto-detected) |
| `--suppress_stage_output` | Suppress model generation output during timing | `True` |
| `--repeat_steps` | Number of test iterations per mode | `100` |
| `--warmup_steps` | Warmup iterations before timing | `10` |
| `--device` | `cuda` / `cpu` | `cuda` |
| `--profile_memory` | Whether to record peak GPU memory | `True` |
| `--print_raw_measurements` | Print per-step raw latency measurements | `True` |
| `--save_csv` | CSV output path for summary | `out/openvla_profile_summary.csv` |
| `--save_json` | JSON output path for raw measurements | `out/openvla_profile_raw.json` |

**Output:**
- Real-time per-iteration latency printed to terminal, plus a summary comparison table
- If you run through `conda run`, keep `--no-capture-output` or `--live-stream`; otherwise conda may buffer stdout until the command exits.
- CSV/JSON saved to `out/`

Latency notes:
- `last_token_logits` is now **disabled by default**. Enable it to skip the unused full-vocabulary prefill projection. When enabled, the LLM still processes the full multimodal context and uses full-vocabulary argmax, but it no longer computes unused prefill logits for every context token.
- `fast_action_head` is more aggressive: it only scores action-bin tokens, bypassing the full `lm_head` projection entirely.

Latency methodology (VLA-Pruner convention, max LLM scope):
- `data_ms`: observation/image/instruction read and handoff to preprocessing.
- `preprocess_ms`: resize/normalization/processor/tokenizer/prompt/batch/CPU-to-GPU transfer.
- `vision_ms`: vision backbone kernel-launch only (no sync for projector; GPU exec falls into LLM window).
- `llm_ms`: `model_latency_ms - vision_ms - action_ms` (covers projector + embedding + prefill + decode + GPU transfer + overhead).
- `action_ms`: numpy action decode only (GPU→CPU transfer counted in LLM).
- `model_latency_ms`: wall-clock from before data-load to after action end.
- `end_to_end_latency_ms`: same as `model_latency_ms`.

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

# Start server (FP8 / w4a8 quantized)
python vla-scripts/deploy.py \
  --openvla_path openvla/openvla-7b \
  --direct_quant_mode fp8 \
  --attn_implementation sdpa
```

**Server Parameters:**

| Parameter | Description | Default |
|------|------|--------|
| `--openvla_path` | HF Hub ID or local model path | `openvla/openvla-7b` |
| `--attn_implementation` | `auto` / `sdpa` / `eager` / `flash_attention_2` | `auto` |
| `--direct_quant_mode` | Quantization mode: `none` / `fp8` / `w8a16` / `w8a8` / `w4a16` / `bnb_*` | `none` |
| `--direct_quant_group_size` | W4A16 group size | `128` |
| `--bnb_int8_threshold` | bitsandbytes outlier threshold | `0.0` |
| `--bnb_4bit_compute_dtype` | bitsandbytes 4-bit compute dtype | `auto` |
| `--bnb_4bit_quant_type` | bitsandbytes 4-bit quant type | `nf4` |
| `--bnb_4bit_use_double_quant` | Double quantization for 4-bit | `True` |
| `--fp8_activation_scale` | FP8 activation scale | `1.0` |
| `--fast_action_head` | Action-bin logits only | `False` |
| `--last_token_logits` | Backbone-only prefill projection | `True` |
| `--drop_full_attention_mask` | Drop all-ones attention masks | `True` |
| `--max_vision_tokens` | Vision token reduction limit | `0` |
| `--vision_token_strategy` | `uniform` / `pool` / `first` | `uniform` |
| `--llm_max_layers` | LLM layer limit | `0` |
| `--llm_layer_strategy` | `first` / `uniform` / `last` | `first` |
| `--host` | Server host | `0.0.0.0` |
| `--port` | Server port | `8000` |

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

# Quantize with FP8 tensorwise (previously w4a8)
report = quantize_openvla_language_model(
    model,
    DirectQuantConfig(mode="fp8", fp8_activation_scale=1.0),
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
cd /path/to/Openvla/Quant_Openvla
docker build -t qvla-quant .

# Interactive run
docker run --gpus all -it --rm qvla-quant bash

# Run profiling
docker run --gpus all --rm qvla-quant \
  python experiments/robot/openvla_profile.py \
    --model_path openvla/openvla-7b \
    --quant_modes none,fp8,w8a16

# Start inference server (with port mapping)
docker run --gpus all -p 8000:8000 --rm qvla-quant \
  python vla-scripts/deploy.py \
    --openvla_path openvla/openvla-7b \
    --direct_quant_mode fp8
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
    --quant_modes fp8 \
    --repeat_steps 5 \
    --warmup_steps 3 \
    --attn_implementation sdpa

# Open results in Nsight Systems GUI
nsys-ui qvla_quant_nsys.nsys-rep
```

### Nsight Compute — Single Kernel Analysis

```bash
# Analyze quantized FP8 tensor-core matmul kernel
ncu \
  --kernel-name "scaled_mm" \
  --launch-count 20 \
  --set full \
  -o qvla_quant_ncu \
  python experiments/robot/openvla_profile.py \
    --model_path openvla/openvla-7b \
    --quant_modes fp8 \
    --repeat_steps 1 \
    --warmup_steps 0 \
    --attn_implementation sdpa
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
│   ├── openvla_direct_quant.py    # Core quantization module (W8A16/W8A8/W4A16/FP8/bnb)
│   ├── openvla_fast_action.py     # Fast action generation & utility helpers
│   └── openvla_profile.py         # Profiling CLI with VLA-Pruner timing methodology
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
