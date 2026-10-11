# MMDetection 3.3.0 environment (Cascade Mask R-CNN / Cascade R-CNN + Swin-T).
# mmdet 3.3.0 requires mmcv>=2.0.0rc4,<2.2.0 -> mmcv 2.1.0, which has no prebuilt wheels for CUDA 12.8,
# so it is compiled from source (MMCV_WITH_OPS=1) for the listed GPU architectures.
# Build:  docker build -f docker/mmdet.Dockerfile -t defectbench/mmdet:cu128 .
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    FORCE_CUDA=1 MMCV_WITH_OPS=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MAX_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends git ninja-build libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install "numpy<2" pyyaml pycocotools opencv-python-headless scipy matplotlib "setuptools<70" wheel \
    openmim mmengine==0.10.7

ARG MMCV_VERSION=2.1.0
RUN git clone --depth 1 --branch v${MMCV_VERSION} https://github.com/open-mmlab/mmcv.git /opt/mmcv \
    && cd /opt/mmcv && pip install --no-build-isolation -v -e .
ARG MMDET_COMMIT=cfd5d3a985b0249de009b67d04f37263e11cdf3d
RUN git clone https://github.com/open-mmlab/mmdetection.git /opt/mmdetection \
    && cd /opt/mmdetection && git checkout ${MMDET_COMMIT} && pip install --no-build-isolation -e .

ENV MMDET_ROOT=/opt/mmdetection DEFECTBENCH_ENV=mmdet
WORKDIR /workspace
