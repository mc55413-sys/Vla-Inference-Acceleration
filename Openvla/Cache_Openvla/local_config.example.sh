#!/usr/bin/env bash
# Copy this file to local_config.sh and edit it for your machine.
# local_config.sh is ignored by git.

# Python used by the experiment scripts.
PYTHON_BIN="${PYTHON_BIN:-/path/to/conda/env/bin/python}"

# Local OpenVLA checkpoint. You can also pass CHECKPOINT_PATH inline.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/path/to/openvla-7b-finetuned-libero-spatial}"

# Optional profiler binary overrides.
# NSYS_BIN="${NSYS_BIN:-/opt/nvidia/nsight-systems/<version>/target-linux-x64/nsys}"
# NCU_BIN="${NCU_BIN:-/opt/nvidia/nsight-compute/<version>/ncu}"
