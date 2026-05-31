ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV NO_ALBUMENTATIONS_UPDATE=1
ENV QUANTVLA_SKIP_CONDA=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      ffmpeg \
      git \
      libatlas-base-dev \
      libgl1 \
      libgtk-3-0 \
      libhdf5-serial-dev \
      libopenblas-dev \
      libsm6 \
      libstdc++6 \
      libtbb12 \
      libtesseract-dev \
      libxext6 \
      make \
      nasm \
      python3 \
      python3-dev \
      python3-pip \
      python3-setuptools \
      wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace/QuantVLA

# cuDSS is required by recent Jetson PyTorch stacks.
RUN wget https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb && \
    dpkg -i cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb && \
    cp /var/cudss-local-tegra-repo-ubuntu2204-0.6.0/cudss-*-keyring.gpg /usr/share/keyrings/ && \
    chmod 777 /tmp && \
    apt-get update && \
    apt-get -y install cudss && \
    rm -f cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

COPY . /workspace/QuantVLA/

# Use Jetson AI Lab wheels for the ARM64 JetPack stack.
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126 \
    PIP_TRUSTED_HOST=pypi.jetson-ai-lab.io \
    python3 -m pip install -e ".[orin]" --no-build-isolation

# Build and install decord for video decoding on Jetson.
RUN git clone https://git.ffmpeg.org/ffmpeg.git /tmp/ffmpeg && \
    cd /tmp/ffmpeg && \
    git checkout n4.4.2 && \
    ./configure --enable-shared --enable-pic --prefix=/usr && \
    make -j"$(nproc)" && \
    make install && \
    git clone --recursive https://github.com/dmlc/decord /tmp/decord && \
    cd /tmp/decord && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j"$(nproc)" && \
    cd ../python && \
    python3 setup.py install --user && \
    rm -rf /tmp/ffmpeg /tmp/decord

RUN mkdir -p /tmp/logs /tmp/numba_cache /tmp/matplotlib-cache /workspace/QuantVLA/results

ENV PYTHONPATH="/workspace/QuantVLA:/workspace/QuantVLA/LIBERO"
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV MPLCONFIGDIR=/tmp/matplotlib-cache
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/root/.local/decord/"

CMD ["/bin/bash"]
