"""Small IO helpers: JSON/YAML, hashing, stable object hashes."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if math.isnan(v) else v
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def _sanitize(o: Any):
    """Replace NaN/inf by None so that the JSON is standard-compliant."""
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


def dump_json(obj: Any, path: str | os.PathLike, indent: int | None = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_sanitize(json.loads(json.dumps(obj, default=_json_default))), f,
                  indent=indent, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(obj: Any, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(json.loads(json.dumps(obj, default=_json_default)), f,
                       sort_keys=False, allow_unicode=True)


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_hash(obj: Any, length: int = 16) -> str:
    """Deterministic hash of a JSON-serialisable object (key order independent)."""
    s = json.dumps(obj, sort_keys=True, default=_json_default, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:length]


def resolve_path(p: str | os.PathLike, base: Path = PROJECT_ROOT) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(p))))
    return p if p.is_absolute() else (base / p)


def deep_update(base: dict, upd: dict) -> dict:
    """Recursive dict merge (returns a new dict)."""
    out = dict(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def parse_override(expr: str) -> dict:
    """Parse ``a.b.c=value`` into ``{"a": {"b": {"c": value}}}`` (value parsed as YAML)."""
    if "=" not in expr:
        raise ValueError(f"Override must be key=value, got {expr!r}")
    key, val = expr.split("=", 1)
    value = yaml.safe_load(val)
    out: dict = {}
    cur = out
    parts = key.strip().split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value
    return out
