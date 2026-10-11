# Detectron2 (ViTDet) + detrex (DINO) environment - Linux container with compiled CUDA ops.
# Build:  docker build -f docker/detectron2.Dockerfile -t defectbench/detectron2:cu128 .
# Run:    orchestrated automatically (pipeline.py mounts the project at /workspace).
# torch 2.11.0 + CUDA 12.8 (supports sm_120 / RTX 50xx as well as Ampere/Ada/Hopper).
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
# The 2.11 image uses the Ubuntu 24.04 system Python (PEP 668 "externally managed"); inside this isolated
# container installing on top of it is intended, hence PIP_BREAK_SYSTEM_PACKAGES=1 below.

ENV DEBIAN_FRONTEND=noninteractive \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MAX_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends git ninja-build libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "numpy<2.4" pyyaml pycocotools opencv-python-headless scipy matplotlib fvcore iopath \
    omegaconf hydra-core timm==1.0.30 cloudpickle tabulate termcolor yacs pillow

# detrex at the pinned commit; its detectron2 submodule (70f454304e1a38378200459dd2dbca0f0f4a5ab4) provides
# both detectron2 (incl. projects/ViTDet configs + the ViT/SimpleFeaturePyramid modeling code) and detrex.
ARG DETREX_COMMIT=e244e6c3da3e84566728c52c21fb061d23ce0e2f
RUN git clone https://github.com/IDEA-Research/detrex.git /opt/detrex \
    && cd /opt/detrex && git checkout ${DETREX_COMMIT} && git submodule update --init --recursive
RUN pip install --no-build-isolation -e /opt/detrex/detectron2 \
    && pip install --no-build-isolation -e /opt/detrex

ENV DETECTRON2_ROOT=/opt/detrex/detectron2 DETREX_ROOT=/opt/detrex DEFECTBENCH_ENV=detectron2
WORKDIR /workspace
