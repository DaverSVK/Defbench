"""Configuration loading (datasets, models, experiment profiles) with CLI overrides."""
from __future__ import annotations

from pathlib import Path

from .utils.io import PROJECT_ROOT, deep_update, load_yaml, parse_override, resolve_path

CONFIG_DIR = PROJECT_ROOT / "configs"


def _load_named(kind: str, name_or_path: str) -> dict:
    p = Path(name_or_path)
    if not p.suffix:
        p = CONFIG_DIR / kind / f"{name_or_path}.yaml"
    elif not p.is_absolute():
        p = resolve_path(p)
    if not p.exists():
        avail = sorted(x.stem for x in (CONFIG_DIR / kind).glob("*.yaml"))
        raise FileNotFoundError(f"{kind} config '{name_or_path}' not found. Available: {avail}")
    cfg = load_yaml(p)
    # simple inheritance: `_base_: other_name` (the `abstract` flag of a base is not inherited)
    base = cfg.pop("_base_", None)
    if base:
        b = _load_named(kind, base)
        b.pop("abstract", None)
        b.pop("_config_path", None)
        cfg = deep_update(b, cfg)
    cfg.setdefault("_config_path", p.relative_to(PROJECT_ROOT).as_posix() if p.is_relative_to(PROJECT_ROOT) else str(p))
    return cfg


def load_dataset_config(name: str, overrides: list[str] | None = None) -> dict:
    cfg = _load_named("datasets", name)
    for o in overrides or []:
        cfg = deep_update(cfg, parse_override(o))
    cfg["raw_root_abs"] = str(resolve_path(cfg["raw_root"]))
    cfg["processed_root_abs"] = str(resolve_path(cfg["processed_root"]))
    return cfg


def load_model_config(name: str, overrides: list[str] | None = None) -> dict:
    cfg = _load_named("models", name)
    for o in overrides or []:
        cfg = deep_update(cfg, parse_override(o))
    return cfg


def load_profile(name: str = "default") -> dict:
    return _load_named("profiles", name)


def list_configs(kind: str) -> list[str]:
    return sorted(x.stem for x in (CONFIG_DIR / kind).glob("*.yaml"))
