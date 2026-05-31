# Prune GR00T: FastV Visual Token Pruning

This project tests FastV-style visual token pruning on GR00T N1.5 + LIBERO. The pruning logic is located in `gr00t/model/fastv.py` and is controlled via the environment variables `GR00T_FASTV_ENABLE`, `GR00T_FASTV_K`, and `GR00T_FASTV_R`.

The Dockerfile includes two built-in conda environments:

| Environment | Purpose |
|---|---|
| `groot_test` | Load the GR00T model and start the baseline / FastV pruning inference service |
| `libero_test` | Run the LIBERO simulation evaluation client |

## 1. Build the Docker Image

```bash
cd "/home/dell/桌面/STJ/ATC/GR00T/Prune_GR00T"
docker build -t prune-gr00t-fastv:latest .
```

It is recommended to mount the Hugging Face cache to avoid re-downloading the model each time the container is rebuilt or restarted:

```bash
mkdir -p "$HOME/.cache/huggingface" results logs
```

## 2. Run Baseline

Terminal 1 — start the GR00T server without pruning:

```bash
docker run --gpus all -it --rm --network host \
  -v "$PWD/results:/workspace/Prune_GR00T/results" \
  -v "$PWD/logs:/tmp/logs" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  prune-gr00t-fastv:latest \
  bash -lc "GR00T_FASTV_ENABLE=0 ./run_inference_server.sh libero_10"
```

Terminal 2 — run the LIBERO evaluation:

```bash
docker run --gpus all -it --rm --network host \
  -v "$PWD/results:/workspace/Prune_GR00T/results" \
  -v "$PWD/logs:/tmp/logs" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  prune-gr00t-fastv:latest \
  bash -lc "./run_libero_eval.sh libero_10 --headless --profile-server --print_step_latency --output-json results/libero_baseline.json"
```

`--profile-server` is essential: without it you only see the total client latency and cannot observe the System-2 / System-1 per-stage breakdown.

## 3. Run FastV Pruning

Terminal 1 — start the pruning service. The configuration below prunes approximately 50% of visual tokens after layer 2:

```bash
docker run --gpus all -it --rm --network host \
  -v "$PWD/results:/workspace/Prune_GR00T/results" \
  -v "$PWD/logs:/tmp/logs" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  prune-gr00t-fastv:latest \
  bash -lc "GR00T_FASTV_ENABLE=1 GR00T_FASTV_K=2 GR00T_FASTV_R=0.5 ./run_inference_server.sh libero_10"
```

Terminal 2 — run the same evaluation suite:

```bash
docker run --gpus all -it --rm --network host \
  -v "$PWD/results:/workspace/Prune_GR00T/results" \
  -v "$PWD/logs:/tmp/logs" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  prune-gr00t-fastv:latest \
  bash -lc "./run_libero_eval.sh libero_10 --headless --profile-server --print_step_latency --output-json results/libero_fastv50.json"
```

Available tasks:

```text
libero_spatial
libero_goal
libero_object
libero_90
libero_10
```

## 4. Latency-Only Benchmark (no LIBERO simulation)

Baseline:

```bash
docker run --gpus all -it --rm \
  -v "$PWD/results:/workspace/Prune_GR00T/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  prune-gr00t-fastv:latest \
  bash -lc "conda run -n groot_test python tools/benchmark_gr00t_fastv_profile.py \
    --use_fastv false \
    --warmup-steps 5 \
    --repeat-steps 20 \
    --output-json results/fastv_profile/baseline.json \
    --output-csv results/fastv_profile/baseline.csv"
```

FastV-50%:

```bash
docker run --gpus all -it --rm \
  -v "$PWD/results:/workspace/Prune_GR00T/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  prune-gr00t-fastv:latest \
  bash -lc "conda run -n groot_test python tools/benchmark_gr00t_fastv_profile.py \
    --use_fastv true \
    --fastv_k 2 \
    --fastv_r 0.5 \
    --warmup-steps 5 \
    --repeat-steps 20 \
    --baseline-json results/fastv_profile/baseline.json \
    --output-json results/fastv_profile/fastv50.json \
    --output-csv results/fastv_profile/fastv50.csv"
```

## 5. Key Parameters

| Parameter | Meaning | Default |
|---|---|---|
| `GR00T_FASTV_ENABLE` | Enable FastV pruning | `0` |
| `GR00T_FASTV_K` | Prune visual tokens after this layer | `2` |
| `GR00T_FASTV_R` | Visual token pruning ratio — `0.5` prunes ~50% | `0.5` |
| `GR00T_DENOISING_STEPS` | Action head denoising steps | `8` |

Results are written to:

```text
results/
logs/
```
