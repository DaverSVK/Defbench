"""prepare-data: audit -> manifest -> leakage-free split (+ integrity check of an existing split)."""
from __future__ import annotations

import logging
from pathlib import Path

from ..utils.io import dump_json, load_json
from .audit import run_audit, write_audit_outputs
from .split import make_split, write_manifest

log = logging.getLogger(__name__)


def prepare_dataset(cfg: dict, split_name: str = "default", force_split: bool = False) -> dict:
    res = run_audit(cfg)
    write_audit_outputs(res, cfg)
    near = res["report"].get("_near_pairs_full", [])
    manifest = write_manifest(cfg, res["records"], res["report"])
    split = make_split(cfg, [r for r in res["records"] if not r["excluded"]], near)
    split["manifest_hash"] = manifest["manifest_hash"]
    split["split_name"] = split_name
    out = Path(cfg["processed_root_abs"]) / "splits" / f"{split_name}.json"
    if out.exists():
        old = load_json(out)
        if old.get("split_hash") != split["split_hash"] and not force_split:
            raise RuntimeError(
                f"Existing split {out} (hash {old.get('split_hash')}) differs from the regenerated one "
                f"({split['split_hash']}). Runs already trained on the old split would become incomparable. "
                "Use --force-split to overwrite deliberately, or --split-name to create a new split.")
    dump_json(split, out, indent=1)
    log.info("%s: manifest %s, split %s -> %s", cfg["name"], manifest["manifest_hash"], split["split_hash"], out)
    return {"manifest": manifest, "split": split, "audit": res["report"]}
