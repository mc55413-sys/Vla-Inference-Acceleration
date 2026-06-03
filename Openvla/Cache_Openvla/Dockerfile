FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libegl1 \
    libgl1 \
    libglvnd0 \
    libglx0 \
    libosmesa6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch CUDA wheels first. Override the base image or this line if your
# deployment target requires a different CUDA/PyTorch combination.
RUN python3 -m pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

WORKDIR /workspace/Cache_Openvla
COPY . .

# Install project dependencies when the corresponding source trees are present.
RUN if [ -f src/openvla/pyproject.toml ]; then python3 -m pip install -e src/openvla; fi && \
    if [ -f src/openvla/requirements-min.txt ]; then python3 -m pip install -r src/openvla/requirements-min.txt; fi && \
    if [ -f src/openvla/experiments/robot/libero/libero_requirements.txt ]; then python3 -m pip install -r src/openvla/experiments/robot/libero/libero_requirements.txt; fi && \
    python3 -m pip install draccus numpy pillow tqdm opencv-python-headless mujoco robosuite

ENV PYTHON_BIN=python3 \
    PROJECT_DIR=/workspace/Cache_Openvla \
    CHECKPOINT_PATH=/workspace/checkpoints/openvla-7b-finetuned-libero-spatial

ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["CHECK_ONLY=1 bash run_latency_experiment.sh"]
