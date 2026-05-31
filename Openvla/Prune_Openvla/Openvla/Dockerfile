# syntax=docker/dockerfile:1.6

ARG CUDA_IMAGE=nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG CONDA_ENV_NAME=vlapruner
ARG PYTHON_VERSION=3.10
ARG INSTALL_FLASH_ATTN=0

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    CONDA_DIR=/opt/conda \
    CONDA_ENV_NAME=${CONDA_ENV_NAME} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu130 \
    MUJOCO_GL=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libglvnd0 \
    libglx0 \
    libglfw3 \
    libosmesa6-dev \
    ninja-build \
    patchelf \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniconda.sh \
    && "${CONDA_DIR}/bin/conda" create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" \
    && "${CONDA_DIR}/bin/conda" clean -afy

ENV PATH=${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin:${CONDA_DIR}/bin:${PATH}

WORKDIR /workspace
COPY src/openvla /workspace/src/openvla
COPY LICENSE /workspace/src/openvla/LICENSE
COPY docker/entrypoint.sh /usr/local/bin/vlapruner-entrypoint

WORKDIR /workspace/src/openvla

RUN python -m pip install --upgrade pip setuptools wheel packaging ninja \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.10.0 \
        torchvision==0.25.0 \
        torchaudio==2.10.0 \
    && python -m pip install -r experiments/robot/libero/libero_requirements.txt \
    && python -m pip install -e . --no-build-isolation \
    && if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then \
        python -m pip install "flash-attn==2.5.5" --no-build-isolation; \
    fi \
    && chmod +x /usr/local/bin/vlapruner-entrypoint

ENTRYPOINT ["vlapruner-entrypoint"]
CMD ["bash"]
