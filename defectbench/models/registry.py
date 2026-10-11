"""Model registry (configs/models/*.yaml) and the model x dataset x task matrix.

Every registry entry declares: unique name, family, task(s), adapter, environment,
exact source (repo + commit / package version), license, pretrained weights,
implementation status and - if unsupported - the reason.

Status values
  verified        adapter executed in this project (smoke test and/or training run)
  implemented     adapter code complete, NOT executed here (environment not installed)
  blocked         official code exists but cannot run on the available hardware/software
  unsupported     the requested combination does not exist officially (not replaced silently)
"""
from __future__ import annotations

from ..config import CONFIG_DIR, load_dataset_config, load_model_config
from ..utils.io import load_yaml


def load_registry() -> dict[str, dict]:
    reg = {}
    for p in sorted((CONFIG_DIR / "models").glob("*.yaml")):
        cfg = load_yaml(p)
        if cfg.get("abstract"):
            continue
        cfg = load_model_config(p.stem)
        name = cfg["name"]
        if name in reg:
            raise ValueError(f"duplicate model name {name}")
        if name != p.stem:
            raise ValueError(f"model name {name} must equal file name {p.stem}")
        reg[name] = cfg
    return reg


def task_entry(model: dict, ds: dict, task: str, allow_weak: bool = False) -> dict:
    """Decide whether (model, dataset, task) is a valid experiment and how it is labelled."""
    t = ds.get("tasks", {})
    status = model.get("status", "implemented")
    base = {"model": model["name"], "dataset": ds["name"], "task": task}
    if status in ("unsupported",):
        return {**base, "status": "unsupported", "reason": model.get("unsupported_reason", "")}
    if task == "bbox":
        src = t.get("bbox", "none")
        if src == "annotation":
            e = {**base, "status": "run", "annotation": "bounding boxes (original annotation)"}
        elif src == "derived_from_mask":
            e = {**base, "status": "run", "annotation": "boxes DERIVED from semantic masks (connected components)",
                 "flag": "derived_boxes"}
        else:
            return {**base, "status": "not_applicable", "reason": "dataset has no box or mask annotation"}
    elif task == "seg":
        if t.get("mask") == "annotation":
            e = {**base, "status": "run", "annotation": "pixel masks (original annotation)"}
        elif t.get("bbox") == "annotation" and (allow_weak or model.get("allow_weak_box_supervision", False)):
            e = {**base, "status": "optional_weak", "annotation": "box-filled pseudo masks",
                 "flag": "weakly_supervised", "reason": "no pixel masks; only as explicitly enabled weak experiment"}
        else:
            return {**base, "status": "not_applicable",
                    "reason": "no ground-truth pixel masks (boxes are not converted to masks)"}
    elif task == "instance_seg":
        if t.get("mask") == "annotation":
            e = {**base, "status": "run", "annotation": "instance masks DERIVED from semantic masks (connected "
                 "components) - real pixels, derived instance partition", "flag": "derived_instances"}
        else:
            return {**base, "status": "not_applicable",
                    "reason": "no pixel masks - use the box-only variant (mask head disabled)"}
    else:
        raise ValueError(task)
    e["image_level_anomaly_metrics"] = bool(t.get("normal_images"))
    if not t.get("normal_images"):
        e["image_level_reason"] = "dataset has no normal (defect-free) images -> image AUROC/AUPRC N/A"
    e["pixel_metrics"] = task in ("seg",) and t.get("mask") == "annotation"
    if status == "blocked":
        e["status"] = "blocked"
        e["reason"] = model.get("blocked_reason", "")
    e["model_status"] = status
    return e


def build_task_matrix(datasets: list[str]) -> list[dict]:
    reg = load_registry()
    rows = []
    for dsn in datasets:
        ds = load_dataset_config(dsn)
        for name, m in reg.items():
            for task in m.get("tasks", []):
                rows.append(task_entry(m, ds, task))
    return rows


def matrix_markdown(rows: list[dict]) -> str:
    L = ["| model | dataset | task | status | annotation / reason | image-level AUROC | pixel metrics |",
         "|---|---|---|---|---|---|---|"]
    def short(t, n=160):
        t = " ".join(str(t).split())
        return t if len(t) <= n else t[: n - 3] + "... (full reason in the model config / task_matrix.json)"

    for r in rows:
        L.append(f"| {r['model']} | {r['dataset']} | {r['task']} | {r['status']} | "
                 f"{short(r.get('annotation') or r.get('reason', ''))}{(' - ' + short(r['reason'])) if r.get('annotation') and r.get('reason') else ''} | "
                 f"{'yes' if r.get('image_level_anomaly_metrics') else 'N/A'} | {'yes' if r.get('pixel_metrics') else 'N/A'} |")
    return "\n".join(L)
