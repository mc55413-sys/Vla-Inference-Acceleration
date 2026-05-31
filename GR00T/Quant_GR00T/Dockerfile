ARG BASE_IMAGE=python:3.10-slim-bookworm
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV NO_ALBUMENTATIONS_UPDATE=1
ENV QUANTVLA_SKIP_CONDA=1

WORKDIR /workspace/QuantVLA

# System packages for PyTorch, MuJoCo/LIBERO rendering, video codecs, and editable installs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libeigen3-dev \
    libglew-dev \
    libgl1 \
    libglib2.0-0 \
    libglfw3 \
    libgomp1 \
    libosmesa6 \
    libosmesa6-dev \
    libsm6 \
    libstdc++6 \
    libxext6 \
    libxrender1 \
    patchelf \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# CUDA 13 PyTorch wheels are used so the image can run on RTX 50 / Blackwell
# systems through NVIDIA Container Toolkit.
RUN python -m pip install \
    torch==2.12.0 torchvision==0.27.0 triton==3.7.0 \
    --index-url https://download.pytorch.org/whl/cu130

COPY requirements.inference.txt /tmp/requirements.inference.txt
RUN sed -i \
    -e '/^-e git+/d' \
    -e '/^gr00t/d' \
    -e '/^torch==/d' \
    -e '/^torchvision==/d' \
    -e '/^triton==/d' \
    /tmp/requirements.inference.txt \
    && python -m pip install -r /tmp/requirements.inference.txt

COPY requirements.libero.txt /tmp/requirements.libero.txt
RUN sed -i \
    -e '/^-e git+/d' \
    -e '/^libero/d' \
    -e '/^torch==/d' \
    -e '/^torchvision==/d' \
    -e '/^triton==/d' \
    /tmp/requirements.libero.txt \
    && python -m pip install -r /tmp/requirements.libero.txt

COPY . /workspace/QuantVLA/

RUN python -m pip install -e /workspace/QuantVLA --no-build-isolation \
    && python -m pip install -e /workspace/QuantVLA/LIBERO --no-build-isolation

RUN mkdir -p /root/.libero /tmp/logs /tmp/numba_cache /tmp/matplotlib-cache /workspace/QuantVLA/results \
    && cat > /root/.libero/config.yaml <<'EOF'
assets: /workspace/QuantVLA/LIBERO/libero/libero/assets
bddl_files: /workspace/QuantVLA/LIBERO/libero/libero/bddl_files
benchmark_root: /workspace/QuantVLA/LIBERO/libero/libero
datasets: /workspace/QuantVLA/LIBERO/datasets
init_states: /workspace/QuantVLA/LIBERO/libero/libero/init_files
EOF

ENV PYTHONPATH="/workspace/QuantVLA:/workspace/QuantVLA/LIBERO"
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV MPLCONFIGDIR=/tmp/matplotlib-cache

CMD ["/bin/bash"]
