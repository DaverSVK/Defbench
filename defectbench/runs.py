"""Run specification, directory layout and the fully resolved run configuration.

runs/<dataset>/<model>/<task>/seed<seed>[_<tag>]/
    run_config.json      fully resolved config (dataset, model, profile, overrides, split/manifest hashes)
    environment.json     package versions, git commits (project + third_party), GPU, python
    train_log.csv        per-epoch train/val loss, val metric, lr, time, peak memory
    train_summary.json   best epoch, train time, peak GPU memory, early stopping
    checkpoints/{best,last}.pt  (or the framework's own checkpoint files, see run_config)
    predictions/         unified predictions (val/test) for independent re-computation
    eval/metrics.json    all metrics + reasons for unavailable metrics
    benchmark/latency.json
    figures/             curves and visualisations
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

from .config import load_dataset_config, load_model_config
from .data.split import load_manifest, load_split
from .utils.envinfo import collect_environment
from .utils.io import PROJECT_ROOT, deep_update, dump_json, load_json, parse_override

RUNS_ROOT = PROJECT_ROOT / "runs"


def run_dir_for(dataset: str, model: str, task: str, seed: int, tag: str | None = None,
                root: Path = RUNS_ROOT) -> Path:
    return Path(root) / dataset / model / task / (f"seed{seed}" + (f"_{tag}" if tag else ""))


def resolve_model_cfg(model: str, dataset: str, profile: str, overrides: list[str] | None) -> dict:
    cfg = load_model_config(model)
    # dataset-specific settings of the model (e.g. input size), then hardware/profile settings
    cfg = deep_update(cfg, (cfg.get("datasets") or {}).get(dataset, {}))
    prof = (cfg.get("profiles") or {}).get(profile)
    if prof:
        cfg = deep_update(cfg, prof)
    for o in overrides or []:
        cfg = deep_update(cfg, parse_override(o))
    cfg["_profile"] = profile
    cfg["_overrides"] = list(overrides or [])
    return cfg


def smoke_overrides(cfg: dict) -> dict:
    """Tiny settings for smoke tests: few iterations, small subsets. Never used for results."""
    s = copy.deepcopy(cfg)
    t = s.setdefault("train", {})
    t.update({"epochs": 1, "max_iters_per_epoch": 3, "eval_interval": 1})
    if isinstance(t.get("early_stopping"), dict):
        t["early_stopping"]["patience"] = 0
    s["smoke"] = {"n_train": 8, "n_val": 4, "n_test": 4}
    return s


def create_run(dataset: str, model: str, task: str, seed: int, profile: str = "default",
               overrides: list[str] | None = None, tag: str | None = None, split_name: str = "default",
               device: str = "cuda", smoke: bool = False, root: Path = RUNS_ROOT, dataset_overrides=None) -> dict:
    ds = load_dataset_config(dataset, dataset_overrides)
    mc = resolve_model_cfg(model, dataset, profile, overrides)
    model = mc["name"]  # a config file path may be passed; runs are always named after the config name
    if smoke:
        mc = smoke_overrides(mc)
        tag = tag or "smoke"
    manifest = load_manifest(ds)
    split = load_split(ds, split_name)
    if split.get("manifest_hash") != manifest["manifest_hash"]:
        raise RuntimeError("split was generated from a different manifest - re-run prepare-data")
    rd = run_dir_for(dataset, model, task, seed, tag, root)
    rd.mkdir(parents=True, exist_ok=True)
    run_cfg = {
        "dataset": dataset, "model": model, "task": task, "seed": seed, "profile": profile, "tag": tag,
        "device": device, "smoke": smoke, "split_name": split_name, "split_hash": split["split_hash"],
        "manifest_hash": manifest["manifest_hash"], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(rd), "run_dir_rel": (rd.resolve().relative_to(PROJECT_ROOT).as_posix()
                                            if rd.resolve().is_relative_to(PROJECT_ROOT) else str(rd)),
        "dataset_cfg": ds, "model_cfg": mc,
        "input": {"channel_policy": (ds.get("input") or {}).get("channel_policy", "replicate_gray_to_rgb"),
                  "source_channels": (ds.get("input") or {}).get("channels")},
        "pretraining": mc.get("pretrained", {}),
    }
    old = rd / "run_config.json"
    if old.exists():
        prev = load_json(old)
        if prev.get("split_hash") != run_cfg["split_hash"]:
            raise RuntimeError(f"{rd} was created with split {prev.get('split_hash')}; refusing to mix splits")
    dump_json(run_cfg, old)
    return run_cfg


def load_run(run_dir: str | Path) -> dict:
    """Load a run config and re-resolve absolute paths for THIS machine / container
    (configs store project-relative paths, so the same run works on Windows and in /workspace)."""
    from .utils.io import resolve_path
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    cfg = load_json(run_dir / "run_config.json")
    cfg["run_dir"] = str(run_dir.resolve())
    ds = cfg["dataset_cfg"]
    ds["raw_root_abs"] = str(resolve_path(ds["raw_root"]))
    ds["processed_root_abs"] = str(resolve_path(ds["processed_root"]))
    return cfg


def write_environment(run_dir: str | Path, extra_packages=None) -> None:
    dump_json(collect_environment(extra_packages), Path(run_dir) / "environment.json")
