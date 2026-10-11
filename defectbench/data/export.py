"""Export the manifest + split into framework formats (shared by all runs on the same split).

Exports live in ``<processed_root>/exports/<split_hash>/<format>/`` so every model on a
dataset reads exactly the same images/annotations. Images are hard-linked (copied if
hard links are impossible) - never re-encoded.

* ``coco0``  : COCO json with category_id = canonical label (0-based) - DETR family
               (remap_mscoco_category=False uses category_id directly as label) and RF-DETR
               (flat categories sorted by id -> label index = canonical). Folder layout
               ``{train,valid,test}/_annotations.coco.json`` + images (RF-DETR convention).
* ``yolo``   : ``images/{train,val,test}``, ``labels/{...}/*.txt`` (class cx cy w h, normalised),
               empty label file for negative images, ``data.yaml`` with ``names``.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from ..utils.io import dump_json, stable_hash
from .geometry import xyxy_to_cxcywh_norm, xyxy_to_xywh
from .split import split_records

SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export_root(ds_cfg: dict, split: dict, fmt: str) -> Path:
    return Path(ds_cfg["processed_root_abs"]) / "exports" / split["split_hash"] / fmt


def _file_name(r: dict) -> str:
    # unique, flat file name (ids may contain '/')
    return r["image_id"].split("/", 1)[1].replace("/", "__") + Path(r["file"]).suffix.lower()


def export_coco0(ds_cfg: dict, manifest: dict, split: dict, subset: dict | None = None) -> Path:
    """subset: optional {split: [records]} (smoke tests) -> exported to a separate hashed folder."""
    root = export_root(ds_cfg, split, "coco0")
    if subset:
        root = root.parent / f"coco0_subset_{stable_hash({k: [r['image_id'] for r in v] for k, v in subset.items()}, 8)}"
    done = root / ".complete"
    if done.exists():
        return root
    raw = Path(ds_cfg["raw_root_abs"])
    classes = manifest["classes"]
    for which, d in SPLIT_DIR.items():
        recs = subset[which] if subset else split_records(manifest, split, which, task="bbox")
        images, anns = [], []
        aid = 0
        for n, r in enumerate(recs):
            fn = _file_name(r)
            _link(raw / r["file"], root / d / fn)
            images.append({"id": n, "file_name": fn, "width": r["width"], "height": r["height"],
                           "defectbench_image_id": r["image_id"]})
            for b, l in zip(r["boxes"], r["labels"]):
                x, y, w, h = xyxy_to_xywh(b)[0].tolist()
                anns.append({"id": aid, "image_id": n, "category_id": int(l), "bbox": [x, y, w, h],
                             "area": w * h, "iscrowd": 0})
                aid += 1
        cats = [{"id": i, "name": c, "supercategory": "none"} for i, c in enumerate(classes)]
        with open(root / d / "_annotations.coco.json", "w", encoding="utf-8") as f:
            json.dump({"images": images, "annotations": anns, "categories": cats,
                       "info": {"description": f"defectbench {ds_cfg['name']} {which}", "split_hash": split["split_hash"],
                                "category_id": "0-based canonical label"}}, f)
    dump_json({"format": "coco0", "split_hash": split["split_hash"], "classes": classes}, root / "export_info.json")
    done.write_text("ok", encoding="utf-8")
    return root


def export_coco0_instances(ds_cfg: dict, manifest: dict, split: dict, subset: dict | None = None) -> Path:
    """COCO json with DERIVED instance masks (RLE) for datasets with semantic masks (Magnetic Tile).

    Every 8-connected component of the binarised semantic mask becomes one instance (bbox = component
    box, i.e. identical to the derived boxes of the manifest). The pixels are real annotations, the
    partition into instances is derived - flagged in ``info`` and in every run that uses it.
    """
    import numpy as np
    from pycocotools import mask as mask_util
    from .torch_datasets import load_mask
    root = export_root(ds_cfg, split, "coco0_inst")
    if subset:
        root = root.parent / f"coco0_inst_subset_{stable_hash({k: [r['image_id'] for r in v] for k, v in subset.items()}, 8)}"
    done = root / ".complete"
    if done.exists():
        return root
    import cv2
    raw = Path(ds_cfg["raw_root_abs"])
    thr = int((ds_cfg.get("mask") or {}).get("binarize_threshold", 128))
    conn = int((ds_cfg.get("derived_boxes") or {}).get("connectivity", 8))
    classes = manifest["classes"]
    for which, d in SPLIT_DIR.items():
        recs = subset[which] if subset else split_records(manifest, split, which, task="bbox")
        images, anns = [], []
        aid = 0
        for n, r in enumerate(recs):
            fn = _file_name(r)
            _link(raw / r["file"], root / d / fn)
            images.append({"id": n, "file_name": fn, "width": r["width"], "height": r["height"],
                           "defectbench_image_id": r["image_id"]})
            if not r.get("mask_file") or r.get("mask_class") is None:
                continue
            m = load_mask(raw / r["mask_file"], thr).astype(np.uint8)
            k, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=conn)
            for i in range(1, k):
                x, y, w, h, area = stats[i]
                rle = mask_util.encode(np.asfortranarray((lab == i).astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                anns.append({"id": aid, "image_id": n, "category_id": int(r["mask_class"]),
                             "bbox": [float(x), float(y), float(w), float(h)], "area": float(area),
                             "segmentation": rle, "iscrowd": 0})
                aid += 1
        cats = [{"id": i, "name": c, "supercategory": "none"} for i, c in enumerate(classes)]
        with open(root / d / "_annotations.coco.json", "w", encoding="utf-8") as f:
            json.dump({"images": images, "annotations": anns, "categories": cats,
                       "info": {"description": f"defectbench {ds_cfg['name']} {which} DERIVED instances",
                                "instances": "connected components of real semantic masks (derived partition)",
                                "split_hash": split["split_hash"]}}, f)
    done.write_text("ok", encoding="utf-8")
    return root


def export_yolo(ds_cfg: dict, manifest: dict, split: dict, subset: dict | None = None) -> Path:
    root = export_root(ds_cfg, split, "yolo")
    if subset:
        root = root.parent / f"yolo_subset_{stable_hash({k: [r['image_id'] for r in v] for k, v in subset.items()}, 8)}"
    done = root / ".complete"
    if done.exists():
        return root
    raw = Path(ds_cfg["raw_root_abs"])
    for which in ("train", "val", "test"):
        recs = subset[which] if subset else split_records(manifest, split, which, task="bbox")
        for r in recs:
            fn = _file_name(r)
            _link(raw / r["file"], root / "images" / which / fn)
            lp = root / "labels" / which / (Path(fn).stem + ".txt")
            lp.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            if r["boxes"]:
                nb = xyxy_to_cxcywh_norm(r["boxes"], r["width"], r["height"])
                for l, b in zip(r["labels"], nb):
                    lines.append(f"{int(l)} " + " ".join(f"{v:.6f}" for v in b))
            lp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    data = {"path": root.as_posix(), "train": "images/train", "val": "images/val", "test": "images/test",
            "names": {i: c for i, c in enumerate(manifest["classes"])}}
    import yaml
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    done.write_text("ok", encoding="utf-8")
    return root


def file_name_to_image_id(recs: list[dict]) -> dict[str, str]:
    return {_file_name(r): r["image_id"] for r in recs}
