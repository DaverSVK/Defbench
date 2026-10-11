"""Orchestration: run adapters in-process (core env) or as subprocesses in their own env.

All environments share only (a) the run directory with ``run_config.json`` and
(b) the unified prediction files. Evaluation always runs in the core environment.
"""
from __future__ import annotations

import importlib
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

from .config import CONFIG_DIR
from .runs import load_run, write_environment
from .utils.io import PROJECT_ROOT, dump_json, load_yaml

log = logging.getLogger(__name__)


def environments() -> dict:
    return load_yaml(CONFIG_DIR / "environments.yaml")


def env_python(env_name: str) -> Path:
    e = environments()[env_name]
    base = PROJECT_ROOT / e["path"]
    return base / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def running_in_env(env_name: str) -> bool:
    e = environments()[env_name]
    if e["type"] == "docker":
        return os.environ.get("DEFECTBENCH_ENV") == env_name
    try:
        return Path(sys.prefix).resolve() == (PROJECT_ROOT / e["path"]).resolve()
    except OSError:
        return False


def adapter_class(model_cfg: dict):
    mod, cls = model_cfg["adapter"].split(":")
    return getattr(importlib.import_module(mod), cls)


def _subprocess(env_name: str, args: list[str]) -> int:
    e = environments()[env_name]
    pp = [str(PROJECT_ROOT)]
    extra = {}
    if e.get("torchvision_v2_shim"):
        # see defectbench/runners/shims/sitecustomize.py (needed for spawned DataLoader workers)
        pp.insert(0, str(PROJECT_ROOT / "defectbench" / "runners" / "shims"))
        extra["DEFECTBENCH_TV_SHIM"] = "1"
    envvars = {**os.environ, "PYTHONPATH": os.pathsep.join(pp + [os.environ.get("PYTHONPATH", "")]),
               "DEFECTBENCH_ENV": env_name, "PYTHONIOENCODING": "utf-8", **extra}
    if e["type"] == "venv":
        py = env_python(env_name)
        if not py.exists():
            raise RuntimeError(f"environment '{env_name}' not installed ({py}). Create it with "
                               f"`python scripts/setup_env.py {env_name}` (see README).")
        cmd = [str(py), "-m", "defectbench.runners.run_adapter", *args]
    else:
        # run as the host user so exports/runs/weights written from the container stay writable for the venv
        # environments (root-owned files broke shared exports); HOME/caches point into the mounted project
        user = ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/workspace/.cache/home",
                "-e", "XDG_CACHE_HOME=/workspace/.cache", "-e", "TORCH_HOME=/workspace/.cache/torch",
                "-e", "MPLCONFIGDIR=/workspace/.cache/matplotlib"] if hasattr(os, "getuid") else []
        cmd = ["docker", "run", "--rm", "--gpus", "all", "--shm-size", "8g", "-e", f"DEFECTBENCH_ENV={env_name}",
               *user, "-v", f"{PROJECT_ROOT}:/workspace", "-w", "/workspace", e["image"],
               "python", "-m", "defectbench.runners.run_adapter", *args]
    log.info("launching in env %s: %s", env_name, " ".join(cmd))
    return subprocess.call(cmd, env=envvars, cwd=str(PROJECT_ROOT))


RESUME_FILES = [  # full-state checkpoints (model + optimizer + scheduler + AMP scaler) per framework
    "checkpoints/last.pt",                 # defectbench engine (torchvision, segmentation)
    "framework/weights/last.pt",           # Ultralytics
    "framework/last.ckpt",                 # RF-DETR (PyTorch Lightning)
    "framework/last.pth",                  # DEIMv2 / D-FINE / RT-DETRv2
    "framework/last_checkpoint",           # detectron2 / detrex (text file naming the last .pth)
]


def find_resume_checkpoint(run_dir: Path) -> str | None:
    """Locate the framework's own last full-state checkpoint of a run (``--resume auto``)."""
    run_dir = Path(run_dir)
    for rel in RESUME_FILES:
        p = run_dir / rel
        if p.exists():
            if p.name == "last_checkpoint":
                return str(run_dir / "framework" / p.read_text(encoding="utf-8").strip())
            return str(p)
    mm = sorted((run_dir / "framework").glob("epoch_*.pth"), key=lambda q: int(q.stem.split("_")[1])) \
        if (run_dir / "framework").exists() else []
    return str(mm[-1]) if mm else None  # MMDetection


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)


def execute(action: str, run_dir: Path, **kw) -> int | dict:
    """action in {train, predict, benchmark}. Returns a result dict (in-process) or exit code."""
    run_cfg = load_run(run_dir)
    env_name = run_cfg["model_cfg"]["environment"]
    status = run_cfg["model_cfg"].get("status")
    if status == "unsupported":
        raise RuntimeError(f"{run_cfg['model']} is unsupported: {run_cfg['model_cfg'].get('unsupported_reason')}")
    if kw.get("resume") == "auto":
        kw["resume"] = find_resume_checkpoint(run_dir)
        if kw["resume"] is None:
            raise RuntimeError(f"--resume auto: no full-state checkpoint found in {run_dir}")
        log.info("resuming from %s", kw["resume"])
    if env_name == "core" or running_in_env(env_name):
        return run_in_process(action, run_dir, **kw)
    args = [action, "--run-dir", _rel(run_dir)]
    for k, v in kw.items():
        if v is None or v is False:
            continue
        args += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    code = _subprocess(env_name, args)
    if code != 0:
        raise RuntimeError(f"{action} in environment {env_name} failed with exit code {code}"
                           + (" (CUDA OOM - see oom.json in the run directory)" if code == 3 else ""))
    return code


def run_in_process(action: str, run_dir: Path, resume: str | None = None, checkpoint: str | None = None,
                   splits: str = "val,test", **kw):
    run_cfg = load_run(run_dir)
    write_environment(run_dir, extra_packages=run_cfg["model_cfg"].get("extra_packages"))
    adapter = adapter_class(run_cfg["model_cfg"])(run_cfg)
    if action == "train":
        from .training.engine import OOMAbort
        try:
            return adapter.train(resume=resume)
        except OOMAbort:
            if os.environ.get("DEFECTBENCH_ENV"):
                sys.exit(3)
            raise
    if action == "predict":
        out = {}
        for s in splits.split(","):
            out[s] = str(adapter.predict(s, checkpoint))
        dump_json(out, Path(run_dir) / "predictions" / "index.json")
        return out
    if action == "benchmark":
        from .benchmark import benchmark_adapter
        return benchmark_adapter(adapter, run_cfg, checkpoint=checkpoint, **kw)
    raise ValueError(action)
