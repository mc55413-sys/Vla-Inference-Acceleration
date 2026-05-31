# QVLA Calibration-Free Quantization - Docker Image
# Uses miniconda3 with CUDA 13.0 for RTX 5090 (Blackwell)
#
# Build:
#   docker build -t qvla-quant .
#
# Run (interactive):
#   docker run --gpus all -it --rm qvla-quant bash
#
# Run (profiling):
#   docker run --gpus all --rm qvla-quant \
#     python experiments/robot/openvla_profile.py \
#       --model_path checkpoints/openvla-7b-finetuned-libero-spatial \
#       --quant_modes bf16,w8a16 --attn_implementation eager

FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda3
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && conda init bash

# Copy environment and create conda env
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# Activate QVLA env by default
RUN echo "conda activate QVLA" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Set working directory
WORKDIR /workspace

# Copy the stripped-down project
COPY openvla/ /workspace/openvla/

# Install openvla in editable mode
RUN pip install -e openvla/ && \
    pip install -r openvla/requirements-min.txt pillow tqdm uvicorn fastapi json-numpy

# Default command
CMD ["/bin/bash"]
