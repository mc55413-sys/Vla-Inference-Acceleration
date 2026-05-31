# Docker Run Guide

This project has been packaged as a Docker-based workflow. The image internally creates a Conda virtual environment `vlapruner`, installs OpenVLA/VLA-Pruner code and LIBERO evaluation dependencies, and defaults to the `src/openvla` working directory.

Large model checkpoints are not baked into the image. They are large and should be placed under `src/openvla/checkpoints` on the host machine and mounted into the container via Docker volumes. This keeps the image smaller and makes it easy to swap checkpoints for different tasks.

## 1. Requirements

- Docker and Docker Compose
- NVIDIA driver
- NVIDIA Container Toolkit
- A usable NVIDIA GPU with sufficient VRAM to load OpenVLA 7B
- Checkpoint(s) on the host machine, e.g.:

```bash
src/openvla/checkpoints/openvla-7b-finetuned-libero-spatial
```

The current local repository already includes the Spatial checkpoint:

```bash
src/openvla/checkpoints/openvla-7b-finetuned-libero-spatial
```

## 2. Build the Image

Run from the repository root (the directory containing this README):

```bash
cd /path/to/ATC/Openvla/Prune_Openvla/Openvla
docker-compose build
```

If your machine has Compose v2 installed, you can replace all `docker-compose` commands in this document with `docker compose`.

The resulting image name is:

```bash
vlapruner-openvla:latest
```

The default image is based on CUDA 13.0 and creates a Conda environment named `vlapruner` inside the container. The current project uses eager attention to load the local OpenVLA model for LIBERO evaluation, so FlashAttention is not installed by default. To install it:

```bash
docker-compose build --build-arg INSTALL_FLASH_ATTN=1
```

## 3. Verify the Container

Enter the container:

```bash
docker-compose run --rm vlapruner bash
```

Check Python, PyTorch, and GPU inside the container:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
pwd
```

The working directory should be:

```bash
/workspace/src/openvla
```

## 4. Run on LIBERO-Spatial

Full evaluation command:

```bash
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

To run a quick smoke test with only 1 rollout per task:

```bash
docker-compose run --rm vlapruner \
  python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --use_fastv True \
    --use_prefil_attention True \
    --use_temporal True \
    --fastv_r 0.75 \
    --seed 7 \
    --run_id_note docker_smoke \
    --num_trials_per_task 1
```

## 5. Run Bundled Scripts

The project provides a set of experiment scripts. Run Spatial + VLA-Pruner + prefill attention:

```bash
docker-compose run --rm vlapruner \
  bash vla_pruner_srcipts/run_vla_pruner/run_spatial_prefil.sh
```

Other commonly used scripts:

```bash
# VLA-Pruner, without prefill attention
docker-compose run --rm vlapruner \
  bash vla_pruner_srcipts/run_vla_pruner/run_spatial.sh

# FastV baseline
docker-compose run --rm vlapruner \
  bash vla_pruner_srcipts/run_fastv/run_spatial_fastv.sh

# SparseVLM baseline
docker-compose run --rm vlapruner \
  bash vla_pruner_srcipts/run_sparsevlm/run_spatial_sparsevlm.sh
```

## 6. Running Other LIBERO Task Suites

To run Object, Goal, or LIBERO-10, first place the corresponding checkpoint under `src/openvla/checkpoints` on the host machine, then adjust two parameters:

```bash
--pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-object
--task_suite_name libero_object
```

```bash
--pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal
--task_suite_name libero_goal
```

```bash
--pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10
--task_suite_name libero_10
```

The `docker-compose.yml` mounts the host's `src/openvla/checkpoints` as read-only into the container, so newly placed checkpoints are immediately visible inside the container.

## 7. Output Locations

Logs and rollout videos are written back to the host:

```bash
src/openvla/experiments/logs
src/openvla/rollouts_dev
```

Hugging Face cache is written to:

```bash
.cache/huggingface
```

## 8. Common Notes

- Run Docker commands from the repository root `Openvla`.
- The default directory inside the container is `/workspace/src/openvla`.
- The conda environment name inside the container is `vlapruner`.
- Checkpoints are mounted via volumes, not copied into the image.
- If Docker reports no GPU available, first verify the NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.3-base-ubuntu22.04 nvidia-smi
```
