"""Download pretrained weights into ``weights/`` and record their SHA-256."""
from __future__ import annotations

import importlib
import logging
import sys
import urllib.request
from pathlib import Path

from .utils.io import PROJECT_ROOT, dump_json, load_json, sha256_file

log = logging.getLogger(__name__)
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def fetch(url: str, filename: str | None = None, expected_sha256: str | None = None) -> Path:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    fn = filename or url.split("/")[-1].split("?")[0]
    dst = WEIGHTS_DIR / fn
    if not dst.exists():
        log.info("downloading %s -> %s", url, dst)
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dst)
    h = sha256_file(dst)
    if expected_sha256 and h != expected_sha256:
        raise RuntimeError(f"checksum mismatch for {dst}: {h} != {expected_sha256}")
    reg_p = WEIGHTS_DIR / "registry.json"
    reg = load_json(reg_p) if reg_p.exists() else {}
    reg[fn] = {"url": url, "sha256": h, "bytes": dst.stat().st_size}
    dump_json(reg, reg_p)
    return dst


def import_from_repo(repo_dir: Path, dotted: str):
    """Import ``dotted`` from a cloned repository without leaking its top-level package
    name into ``sys.modules`` (TransUNet and Swin-Unet both define a package ``networks``)."""
    repo_dir = str(Path(repo_dir).resolve())
    top = dotted.split(".")[0]
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == top or k.startswith(top + ".")}
    sys.path.insert(0, repo_dir)
    try:
        mod = importlib.import_module(dotted)
    finally:
        sys.path.remove(repo_dir)
        for k in [k for k in sys.modules if k == top or k.startswith(top + ".")]:
            sys.modules.pop(k)
        sys.modules.update(saved)
    return mod


def third_party_dir(name: str) -> Path:
    p = PROJECT_ROOT / "third_party" / name
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `python scripts/clone_third_party.py {name}` "
                                "(pinned commit from configs/third_party.lock.yaml)")
    return p
