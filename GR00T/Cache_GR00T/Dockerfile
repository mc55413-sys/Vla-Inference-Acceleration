# ============================================================================
# GR00T-Cache Docker Image
#
# Two conda environments matching local machine exactly:
#   groot_test   — GR00T N1.5 inference server
#   libero_test  — LIBERO evaluation client
#
# Conda environments are recreated from:
#   environment_groot_test.yml
#   environment_libero_test.yml
# ============================================================================

FROM nvidia/cuda:13.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ── System dependencies ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git wget curl ca-certificates \
    libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev libegl1 \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    ffmpeg libjpeg-dev libpng-dev \
    patchelf libeigen3-dev libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# ── Install Miniconda ────────────────────────────────────────────────
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"
RUN conda init bash

WORKDIR /workspace/GR00T-Cache

# ── Copy conda environment YAML files ────────────────────────────────
COPY environment_groot_test.yml  /tmp/environment_groot_test.yml
COPY environment_libero_test.yml /tmp/environment_libero_test.yml

# ── Conda env: groot_test (GR00T server) ─────────────────────────────
# Recreate from YAML, then ensure PyTorch matches local cu130
RUN conda env create -f /tmp/environment_groot_test.yml -n groot_test
RUN /opt/conda/envs/groot_test/bin/pip install --force-reinstall \
    torch==2.12.0 torchvision==0.27.0 \
    --index-url https://download.pytorch.org/whl/cu130

# ── Conda env: libero_test (LIBERO eval) ─────────────────────────────
# Recreate from YAML, then ensure PyTorch matches local cu130
RUN conda env create -f /tmp/environment_libero_test.yml -n libero_test
RUN /opt/conda/envs/libero_test/bin/pip install --force-reinstall \
    torch==2.12.0 torchvision==0.27.0 \
    --index-url https://download.pytorch.org/whl/cu130

# ── Copy project source ──────────────────────────────────────────────
COPY . /workspace/GR00T-Cache/

# Install gr00t package in both envs
RUN /opt/conda/envs/groot_test/bin/pip install -e /workspace/GR00T-Cache/ --no-build-isolation
RUN /opt/conda/envs/libero_test/bin/pip install -e /workspace/GR00T-Cache/ --no-build-isolation

# Install LIBERO in libero_test
RUN /opt/conda/envs/libero_test/bin/pip install -e /workspace/GR00T-Cache/LIBERO/ --no-build-isolation

# ── LIBERO config ────────────────────────────────────────────────────
RUN mkdir -p /root/.libero && cat > /root/.libero/config.yaml <<'EOF'
assets: /workspace/GR00T-Cache/LIBERO/libero/libero/assets
bddl_files: /workspace/GR00T-Cache/LIBERO/libero/libero/bddl_files
benchmark_root: /workspace/GR00T-Cache/LIBERO/libero/libero
datasets: /workspace/GR00T-Cache/LIBERO/datasets
init_states: /workspace/GR00T-Cache/LIBERO/libero/libero/init_files
EOF

# ── Output directories ───────────────────────────────────────────────
RUN mkdir -p /tmp/logs /tmp/numba_cache /tmp/matplotlib-cache /workspace/results /workspace/models

# ── Environment ──────────────────────────────────────────────────────
ENV PYTHONPATH="/workspace/GR00T-Cache:/workspace/GR00T-Cache/LIBERO:${PYTHONPATH}"
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV MPLCONFIGDIR=/tmp/matplotlib-cache
ENV MUJOCO_GL=osmesa

# ── Entrypoint scripts ───────────────────────────────────────────────
RUN cat > /usr/local/bin/run-server-baseline <<'SCRIPT'
#!/bin/bash
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate groot_test
MODEL_PATH="${MODEL_PATH:-/workspace/models/gr00t}"
PORT="${PORT:-5556}"
STEPS="${STEPS:-8}"
cd /workspace/GR00T-Cache
echo "[server-baseline] port=${PORT} (no cache)"
exec python scripts/inference_service_cache.py --server \
    --model-path "${MODEL_PATH}" \
    --embodiment-tag new_embodiment \
    --port "${PORT}" \
    --denoising-steps "${STEPS}" \
    --cache-mode none
SCRIPT
RUN chmod +x /usr/local/bin/run-server-baseline

RUN cat > /usr/local/bin/run-server-cache <<'SCRIPT'
#!/bin/bash
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate groot_test
MODEL_PATH="${MODEL_PATH:-/workspace/models/gr00t}"
PORT="${PORT:-5557}"
STEPS="${STEPS:-8}"
REUSE="${REUSE:-0.5}"
TOPK="${TOPK:-5}"
cd /workspace/GR00T-Cache
echo "[server-cache] port=${PORT} reuse=${REUSE} topk=${TOPK}"
exec python scripts/inference_service_cache.py --server \
    --model-path "${MODEL_PATH}" \
    --embodiment-tag new_embodiment \
    --port "${PORT}" \
    --denoising-steps "${STEPS}" \
    --cache-mode full_cache \
    --max-reuse-ratio "${REUSE}" \
    --task-topk "${TOPK}"
SCRIPT
RUN chmod +x /usr/local/bin/run-server-cache

RUN cat > /usr/local/bin/run-eval <<'SCRIPT'
#!/bin/bash
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate libero_test
HOST="${HOST:-localhost}"
PORT="${PORT:-5556}"
SUITE="${SUITE:-libero_spatial}"
OUTPUT="${OUTPUT:-/workspace/results/eval.json}"
cd /workspace/GR00T-Cache
echo "[eval] host=${HOST} port=${PORT} suite=${SUITE}"
exec python examples/Libero/eval/run_libero_eval.py \
    --task-suite-name "${SUITE}" \
    --port "${PORT}" \
    --profile-server \
    --print-step-latency \
    --output-json "${OUTPUT}"
SCRIPT
RUN chmod +x /usr/local/bin/run-eval

RUN cat > /usr/local/bin/download-model <<'SCRIPT'
#!/bin/bash
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate groot_test
MODEL_ID="${1:-nvidia/GR00T-N1.5-3B}"
python -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download('${MODEL_ID}', repo_type='model')
print(f'Downloaded to: {path}')
os.makedirs('/workspace/models', exist_ok=True)
target = '/workspace/models/gr00t'
if not os.path.exists(target):
    os.symlink(path, target)
    print(f'Symlink: {target} -> {path}')
"
SCRIPT
RUN chmod +x /usr/local/bin/download-model

CMD ["/bin/bash"]
