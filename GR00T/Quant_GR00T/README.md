# QuantVLA Docker Run Guide

This project runs Baseline BF16, DuQuant W4, and QuantVLA Full quantization evaluations on GR00T N1.5. The repository provides Docker-based entry points — containers do not depend on a local conda installation. The original `groot_test` and `libero_test` dependencies have been merged into a single image for easier reproducibility across machines.

## Image Selection

| Device | Dockerfile | Purpose |
|---|---|---|
| x86_64 NVIDIA GPU, e.g. RTX 5090/H100/B200 | `Dockerfile` | Inference service, real LIBERO evaluation, local latency/memory/flops benchmarks |
| Jetson Orin / JetPack 6 | `orin.Dockerfile` | Edge-side inference service and local benchmarks; running full LIBERO simulation on Orin is not recommended |

## Prerequisites

The host machine needs:

```bash
docker --version
nvidia-smi
```

And confirm Docker can access the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

On first run the model will be downloaded from Hugging Face. It is recommended to mount the Hugging Face cache inside the container to avoid re-downloading each time.

## Build x86 GPU Image

Run from the project root:

```bash
cd /path/to/Quant_GR00T
docker build -t quantvla:cuda13 .
```

If a Hugging Face token is required:

```bash
export HF_TOKEN=your_token
```

## Build Jetson Orin Image

On the Jetson Orin:

```bash
cd /path/to/Quant_GR00T
docker build -t quantvla:orin -f orin.Dockerfile .
```

When running the Orin image, use the NVIDIA runtime:

```bash
docker run --rm -it --runtime nvidia --network host --ipc host \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:orin bash
```

## Tasks and Modes

Supported tasks:

```bash
libero_spatial
libero_goal
libero_object
libero_90
libero_10
```

Supported run modes:

```bash
baseline   # BF16 baseline
duquant    # DuQuant W4 packed
full       # QuantVLA Full: DuQuant W4 + ATM + OHB
```

## Run Baseline BF16

Two terminals are required. The first starts the inference service; the second runs the LIBERO evaluation.

Terminal 1:

```bash
cd /path/to/Quant_GR00T
docker run --rm -it --gpus all --network host --ipc host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_inference_server.sh baseline libero_10
```

Terminal 2:

```bash
cd /path/to/Quant_GR00T
docker run --rm -it --gpus all --network host --ipc host \
  -e QVLA_HOST=localhost \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_real_libero_benchmark.sh baseline libero_10
```

## Run QuantVLA Full Quantization

Terminal 1:

```bash
cd /path/to/Quant_GR00T
docker run --rm -it --gpus all --network host --ipc host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_inference_server.sh full libero_10
```

Terminal 2:

```bash
cd /path/to/Quant_GR00T
docker run --rm -it --gpus all --network host --ipc host \
  -e QVLA_HOST=localhost \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_real_libero_benchmark.sh full libero_10
```

## Run DuQuant Only

Terminal 1:

```bash
docker run --rm -it --gpus all --network host --ipc host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_inference_server.sh duquant libero_10
```

Terminal 2:

```bash
docker run --rm -it --gpus all --network host --ipc host \
  -e QVLA_HOST=localhost \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_real_libero_benchmark.sh duquant libero_10
```

## Real-Time Metric Output

`tools/run_quantvla_real_libero_benchmark.sh` automatically enables:

```bash
--profile-server
--print_step_latency
```

At runtime you will see output like:

```text
[latency] task=0 episode=0 step=10 Data=... | Preprocess=... | System-2 Vision=... | System-2 Reasoning=... | Bridge=... | System-1 Vision=... | System-1 Action=... | End to End Latency=... avg=... | Model Latency=... avg=...
[tflops] task=0 episode=0 step=10 dense_equiv=...TFLOPs/call
```

For more detailed per-step memory and preprocessing breakdown:

```bash
docker run --rm -it --gpus all --network host --ipc host \
  -e QVLA_HOST=localhost \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_real_libero_benchmark.sh full libero_10 \
  --print_detailed_step_latency
```

Detailed mode additionally prints:

```text
[model component memory] llm=...GiB | dit=...GiB | llm+dit=...GiB | total=...GiB | duquant_layers=... | packed_layers=... | atm_env=... | ohb_env=...
[latency detail] policy=...ms | server=...ms | gr00t=...ms | ... | env=...ms | loop=...ms
```

## Local Benchmark (no LIBERO simulation)

This mode only loads the model and measures latency, memory, FLOPs, and action dumps using synthetic LIBERO observations — useful for quickly checking whether a device can run the pipeline.

Baseline:

```bash
docker run --rm -it --gpus all --network host --ipc host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_local_benchmarks.sh baseline libero_10
```

QuantVLA Full:

```bash
docker run --rm -it --gpus all --network host --ipc host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_local_benchmarks.sh full libero_10
```

Environment variables can control local benchmark iteration counts:

```bash
docker run --rm -it --gpus all --network host --ipc host \
  -e QVLA_WARMUP=3 \
  -e QVLA_ITERS=10 \
  -e QVLA_ACTION_SAMPLES=16 \
  -v "$PWD/results:/workspace/QuantVLA/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  quantvla:cuda13 \
  tools/run_quantvla_local_benchmarks.sh full libero_10
```

## Output Files

Real LIBERO evaluation outputs:

```bash
results/benchmarks/<variant>/<task>/flops.json
results/benchmarks/<variant>/<task>/memory.json
results/benchmarks/<variant>/<task>/success.json
results/benchmarks/<variant>/<task>/real_libero_metrics.json
results/benchmarks/summary.csv
results/benchmarks/summary.md
```

Runtime logs:

```bash
/tmp/logs/libero_eval_<variant>_<task>.log
```

Since `/tmp/logs` is inside the container, mount an external directory to persist them:

```bash
-v "$PWD/logs:/tmp/logs"
```

## Common Environment Variables

| Variable | Default | Description |
|---|---:|---|
| `QVLA_DENOISING_STEPS` | `8` | Denoising steps |
| `QVLA_HOST` | `localhost` | Host for the eval client to connect to the inference service |
| `QVLA_WARMUP` | `5` | Local benchmark warmup iterations |
| `QVLA_ITERS` | `20` | Local benchmark timing iterations |
| `QVLA_ACTION_SAMPLES` | `32` | Local benchmark action dump count |
| `HF_TOKEN` | empty | Hugging Face token for private or rate-limited model access |

## Native Conda Run

Docker is the recommended approach. If you already have a local conda environment set up, you can also run directly:

```bash
conda activate groot_test
tools/run_quantvla_inference_server.sh baseline libero_10
```

In another terminal:

```bash
tools/run_quantvla_real_libero_benchmark.sh baseline libero_10
```

The scripts auto-detect conda; inside Docker they use the container Python directly.

## Troubleshooting

If the container cannot see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

If the evaluation client cannot connect to the server, verify both containers include:

```bash
--network host
```

If the first run is very slow, it is likely downloading the Hugging Face model or generating the DuQuant packed cache. Always mount:

```bash
-v "$HOME/.cache/huggingface:/root/.cache/huggingface"
-v "$PWD/results:/workspace/QuantVLA/results"
```
