"""Collect reproducibility information: package versions, git commits, GPU.

Credentials are never collected: environment variables are NOT dumped except an
explicit allow-list (CUDA_VISIBLE_DEVICES etc.).
"""
from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path

from .io import PROJECT_ROOT

_PACKAGES = [
    "torch", "torchvision", "numpy", "scipy", "sklearn", "cv2", "PIL", "pycocotools",
    "segmentation_models_pytorch", "monai", "timm", "ultralytics", "rfdetr", "detectron2",
    "detrex", "mmdet", "mmcv", "mmengine", "transformers", "faster_coco_eval", "supervision",
]
_ENV_ALLOW = ["CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS"]


def package_versions(extra: list[str] | None = None) -> dict:
    out = {}
    for name in _PACKAGES + (extra or []):
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            continue
    return out


def git_commit(path: Path) -> dict | None:
    try:
        sha = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        if sha.returncode != 0:
            return None
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True,
                               text=True, timeout=10)
        return {"commit": sha.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except Exception:
        return None


def third_party_commits() -> dict:
    tp = PROJECT_ROOT / "third_party"
    out = {}
    if tp.exists():
        for d in sorted(tp.iterdir()):
            if (d / ".git").exists():
                out[d.name] = git_commit(d)
    return out


def gpu_info() -> dict:
    info: dict = {}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        if torch.cuda.is_available():
            info["devices"] = []
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index": i, "name": p.name, "total_memory_gb": round(p.total_memory / 1024 ** 3, 2),
                    "capability": f"{p.major}.{p.minor}",
                })
    except Exception as e:  # torch not installed in this environment
        info["error"] = repr(e)
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            info["nvidia_smi"] = [l.strip() for l in r.stdout.strip().splitlines()]
    except Exception:
        pass
    return info


def collect_environment(extra_packages: list[str] | None = None) -> dict:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(extra_packages),
        "project_git": git_commit(PROJECT_ROOT),
        "third_party_git": third_party_commits(),
        "gpu": gpu_info(),
        "env": {k: os.environ[k] for k in _ENV_ALLOW if k in os.environ},
    }
