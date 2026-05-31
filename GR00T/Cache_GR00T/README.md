# GR00T-Cache: Efficient VLA Manipulation via Adaptive Token Caching

**Token-level KV cache for GR00T N1.5 — accelerating robot manipulation inference.**

> ⚠️ **Development status**: Cache infrastructure is implemented and functional.  
> Measurable latency improvement on the benchmark requires further tuning  
> of cache thresholds and dynamic tiling configuration.

## Architecture

```
Observation (256×256 × 2 cams)
    │
    ▼
Eagle VLM Backbone (System 2)
    ├── Vision Encoder (SigLIP):  ~7 ms
    ├── LLM (Qwen2, 12 layers):   ~9 ms  ← KV cache for static visual tokens
    └── Project to condition:     ~1 ms
    │
    ▼
Flow-Matching Action Head (System 1)
    ├── DiT (16 blocks × 8 steps): ~32 ms  ← Block-level output cache
    │   ├── Cross-attn blocks: condition K/V shared across steps
    │   └── Self-attn blocks:  output cached when input stable
    └── Action decoder:            ~1 ms
    │
    ▼
7-DOF Action + Gripper
```

## Cache Methods

### 1. DiT Block Output Cache (DeepCache-style, CVPR 2024)

Reuses entire DiT block outputs across denoising steps when the block input hasn't changed significantly (cosine similarity > 0.99).

- Blocks 0–3: always computed (early layers capture important changes)
- Blocks 4–15: cached after warmup (steps 0–1), reused in steps 2–7
- **Saves ~30% of DiT latency** (10 ms)

### 2. DiT Condition K/V Cache

Cross-attention condition features (backbone output) are **identical across all 8 denoising steps**. K/V projection computed once, reused 7 times.

### 3. Backbone Visual KV Cache

Static visual tokens (unchanged image regions between consecutive policy steps) reuse K/V from the previous step. Dynamic tiling is disabled to keep sequence lengths stable.

## Quick Start (Docker)

### 1. Build

```bash
cd Cache_GR00T
docker build -t gr00t-cache .
```

### 2. Download model

```bash
docker run --gpus all -v $(pwd)/models:/workspace/models \
    gr00t-cache download-model nvidia/GR00T-N1.5-3B
```

### 3. Run baseline (no cache)

```bash
# Terminal 1 — start server
docker run --gpus all -p 5556:5556 \
    -v $(pwd)/models:/workspace/models \
    gr00t-cache run-server-baseline

# Terminal 2 — run evaluation
docker run --gpus all --network host \
    -v $(pwd)/results:/workspace/results \
    -e HOST=localhost -e PORT=5556 \
    gr00t-cache run-eval
```

### 4. Run with cache

```bash
# Terminal 1 — start cache-enabled server
docker run --gpus all -p 5557:5557 \
    -v $(pwd)/models:/workspace/models \
    -e REUSE=0.5 -e TOPK=5 \
    gr00t-cache run-server-cache

# Terminal 2 — run evaluation
docker run --gpus all --network host \
    -v $(pwd)/results:/workspace/results \
    -e HOST=localhost -e PORT=5557 \
    gr00t-cache run-eval
```

### 5. Compare results

```bash
python -c "
import json
for name in ['baseline','cached']:
    with open(f'results/{name}.json') as f:
        d = json.load(f)
    s = d.get('latency_summary', {})
    print(f'\n=== {name.upper()} ===')
    for k,v in s.items():
        print(f'  {k:35s} mean={v[\"mean_ms\"]:7.2f}ms  p95={v[\"p95_ms\"]:7.2f}ms')
"
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/workspace/models/gr00t` | Path to GR00T checkpoint |
| `PORT` | `5556` (baseline) / `5557` (cache) | Server port |
| `STEPS` | `8` | Denoising steps |
| `REUSE` | `0.5` | Max visual token reuse ratio |
| `TOPK` | `5` | Task-relevant top-K eviction |
| `SUITE` | `libero_spatial` | LIBERO task suite |
| `HOST` | `localhost` | Server hostname |

## Manual Run (without Docker)

### Prerequisites

Two conda environments are required due to dependency conflicts:

```bash
# Environment 1: GR00T server (with CUDA)
conda create -n groot_test python=3.10
conda activate groot_test
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.inference.txt
pip install -e .

# Environment 2: LIBERO eval (with MuJoCo/robosuite)
conda create -n libero_test python=3.10
conda activate libero_test
pip install -r requirements.libero.txt
pip install -e LIBERO/
pip install -e .
```

### Step 1: Start Server

**Baseline (no cache):**
```bash
conda activate groot_test
python scripts/inference_service_cache.py --server \
    --model-path /path/to/gr00t-n1.5-libero \
    --embodiment-tag new_embodiment \
    --port 5556 \
    --denoising-steps 8 \
    --cache-mode none
```

**With cache:**
```bash
conda activate groot_test
python scripts/inference_service_cache.py --server \
    --model-path /path/to/gr00t-n1.5-libero \
    --embodiment-tag new_embodiment \
    --port 5556 \
    --denoising-steps 8 \
    --cache-mode full_cache \
    --max-reuse-ratio 0.5 \
    --task-topk 5
```

### Step 2: Run Eval

```bash
conda activate libero_test
export PYTHONPATH="$PWD:$PWD/LIBERO:$PYTHONPATH"
python examples/Libero/eval/run_libero_eval.py \
    --task-suite-name libero_spatial \
    --port 5556 \
    --profile-server \
    --print-step-latency \
    --output-json /tmp/results.json
```

### Step 3: Compare Results

```bash
python -c "
import json
for name, path in [('baseline','/tmp/baseline.json'),('cached','/tmp/cached.json')]:
    with open(path) as f:
        d = json.load(f)
    s = d.get('latency_summary', {})
    print(f'\n=== {name.upper()} ===')
    for k,v in s.items():
        print(f'  {k:35s} mean={v[\"mean_ms\"]:7.2f}ms  p95={v[\"p95_ms\"]:7.2f}ms')
"
```

## CLI Reference

```
--model-path           Path to GR00T model checkpoint
--embodiment-tag       Robot embodiment (default: new_embodiment)
--port                 Server port (default: 5556)
--denoising-steps      Flow-matching denoising steps (default: 8)
--cache-mode           none | full_cache | backbone_visual_kv_cache | action_head_condition_kv_cache
--max-reuse-ratio      Max fraction of visual tokens to reuse (default: 0.5)
--task-topk            Top-K task-relevant tokens to evict from cache
--entropy-scale        Layer-adaptive entropy scale (default: 1.0)
```

## Output Format

The eval script prints per-step latency and a running summary:

```
[latency] task=0 episode=0 step=10 Data=4.5ms | Preprocess=4.2ms | System-2 Vision=6.7ms | System-2 Reasoning=8.3ms | Bridge=0.0ms | System-1 Vision=1.7ms | System-1 Action=22.5ms | End to End Latency=47.9ms | Model Latency=39.2ms
```

| Stage | Description |
|---|---|
| Data | Observation ingestion |
| Preprocess | Image resize + VLM tokenization + CPU→GPU |
| System-2 Vision | Vision encoder + projector |
| System-2 Reasoning | LLM backbone forward |
| System-1 Action | DiT denoising loop |
| Model Latency | Vision + Reasoning + Action |
| End to End | Data + Preprocess + Model |

## Project Structure

```
Cache_GR00T/
├── gr00t_cache/                  # Cache implementation
│   ├── config.py                 # GR00TCacheConfig
│   ├── cache_manager.py          # Cross-timestep cache logic
│   ├── attention_wrapper.py      # Safe attention KV replacement
│   ├── dit_cache.py              # DiT block-level output cache
│   ├── token_index_map.py        # Multimodal token position mapping
│   ├── model_flops.py            # Theoretical FLOPs counter
│   ├── profiling.py              # Latency profiling tools
│   ├── correctness.py            # Action quality verification
│   ├── ablation.py               # 7-preset ablation runner
│   ├── dummy_model.py            # Dummy model for testing
│   └── adapter.py                # RealGR00TAdapter
├── scripts/
│   └── inference_service_cache.py  # Cache-enabled inference server
├── examples/Libero/eval/
│   └── run_libero_eval_local.py  # Local (no-server) eval
├── tools/
│   └── benchmark_gr00t_cache.py  # 5-stage latency benchmark
├── Dockerfile
├── docker-compose.yml
└── run_gr00t_cache.sh
```

## Citation

If you use GR00T-Cache in your research, please cite:

```bibtex
@misc{gr00t-cache,
  title   = {GR00T-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching},
  author  = {},
  year    = {2025},
}
```
