FROM nvidia/cuda:13.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV NO_ALBUMENTATIONS_UPDATE=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV MPLCONFIGDIR=/tmp/matplotlib-cache

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash bzip2 ca-certificates curl ffmpeg git libegl1 libeigen3-dev libglew-dev \
    libglfw3 libglib2.0-0 libgl1 libgomp1 libosmesa6-dev libsm6 libxext6 \
    libxrender-dev patchelf wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /root/miniconda3 \
    && rm /tmp/miniconda.sh \
    && /root/miniconda3/bin/conda clean -afy

ENV PATH=/root/miniconda3/bin:${PATH}

WORKDIR /workspace/Prune_GR00T

COPY requirements.inference.txt /tmp/requirements.inference.txt
COPY requirements.libero.txt /tmp/requirements.libero.txt

RUN conda create -n groot_test python=3.10 -y \
    && conda run -n groot_test python -m pip install --upgrade pip setuptools wheel \
    && conda run -n groot_test python -m pip install \
        torch==2.12.0 torchvision==0.27.0 triton==3.7.0 \
        --index-url https://download.pytorch.org/whl/cu130 \
    && sed -i '/^-e git+/d; /^gr00t/d' /tmp/requirements.inference.txt \
    && conda run -n groot_test python -m pip install -r /tmp/requirements.inference.txt

RUN conda create -n libero_test python=3.10 -y \
    && conda run -n libero_test python -m pip install --upgrade pip setuptools wheel \
    && conda run -n libero_test python -m pip install \
        torch==2.12.0 torchvision==0.27.0 triton==3.7.0 \
        --index-url https://download.pytorch.org/whl/cu130 \
    && sed -i '/^-e git+/d; /^libero/d' /tmp/requirements.libero.txt \
    && conda run -n libero_test python -m pip install -r /tmp/requirements.libero.txt

COPY . /workspace/Prune_GR00T/

RUN conda run -n groot_test python -m pip install -e /workspace/Prune_GR00T --no-build-isolation \
    && conda run -n libero_test python -m pip install -e /workspace/Prune_GR00T --no-build-isolation --no-deps \
    && conda run -n libero_test python -m pip install -e /workspace/Prune_GR00T/LIBERO --no-build-isolation

RUN mkdir -p /root/.libero /tmp/logs /tmp/numba_cache /tmp/matplotlib-cache /workspace/Prune_GR00T/results \
    && cat > /root/.libero/config.yaml <<'EOF'
assets: /workspace/Prune_GR00T/LIBERO/libero/libero/assets
bddl_files: /workspace/Prune_GR00T/LIBERO/libero/libero/bddl_files
benchmark_root: /workspace/Prune_GR00T/LIBERO/libero/libero
datasets: /workspace/Prune_GR00T/LIBERO/datasets
init_states: /workspace/Prune_GR00T/LIBERO/libero/libero/init_files
EOF

ENV PYTHONPATH=/workspace/Prune_GR00T:/workspace/Prune_GR00T/LIBERO

CMD ["/bin/bash"]
