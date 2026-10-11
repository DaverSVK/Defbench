"""Unified prediction formats shared by all adapters (numpy + stdlib only).

Detection (``defectbench.det.v1``) - one JSON file per (run, split)::

    {
      "format": "defectbench.det.v1",
      "dataset": "neu_det", "split": "test", "model": "faster_rcnn_r50_fpn",
      "class_names": [...],                      # canonical order, label = index
      "coordinate_system": "original_image_pixels_xyxy_pixel_edges",
      "score_threshold_export": 0.001,           # low threshold -> AP over full score range
      "max_detections_per_image": 100,
      "predictions": [
         {"image_id": "neu_det/000123", "boxes_xyxy": [[x1,y1,x2,y2], ...],
          "scores": [...], "labels": [...]}, ...]
    }

Every image of the split must be present (possibly with empty lists), so that
"no prediction" is distinguishable from "image not processed".

Segmentation (``defectbench.seg.v1``): a directory with ``index.json`` plus one
``<safe_image_id>.npz`` per image containing ``prob`` (float16, CxHxW at ORIGINAL
resolution; C = 1 for binary defect probability, K+1 for multiclass softmax with
channel 0 = background) and ``defect`` (float32 HxW defect score map, computed in
float32 before storage; used for all threshold-free pixel metrics).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np

DET_FORMAT = "defectbench.det.v1"
SEG_FORMAT = "defectbench.seg.v1"
COORD_SYSTEM = "original_image_pixels_xyxy_pixel_edges"
DEFAULT_EXPORT_SCORE_THR = 0.001
DEFAULT_MAX_DETS = 100


def safe_id(image_id: str) -> str:
    return image_id.replace("/", "__").replace("\\", "__").replace(":", "_")


def make_det_record(image_id: str, boxes, scores, labels) -> dict:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError(f"{image_id}: inconsistent lengths {len(boxes)}, {len(scores)}, {len(labels)}")
    return {"image_id": image_id, "boxes_xyxy": boxes.round(3).tolist(),
            "scores": [float(s) for s in scores], "labels": [int(l) for l in labels]}


def postprocess_det_record(rec: dict, width: int, height: int, num_classes: int,
                           score_thr: float = DEFAULT_EXPORT_SCORE_THR,
                           max_dets: int = DEFAULT_MAX_DETS) -> dict:
    """Clip to image, drop degenerate boxes, apply the common score threshold and top-k."""
    b = np.asarray(rec["boxes_xyxy"], dtype=np.float64).reshape(-1, 4)
    s = np.asarray(rec["scores"], dtype=np.float64).reshape(-1)
    l = np.asarray(rec["labels"], dtype=np.int64).reshape(-1)
    if len(l) and (l.min() < 0 or l.max() >= num_classes):
        raise ValueError(f"{rec['image_id']}: label outside [0,{num_classes}) - class-id mapping bug")
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, width)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, height)
    keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1]) & np.isfinite(s) & (s >= score_thr)
    b, s, l = b[keep], s[keep], l[keep]
    order = np.argsort(-s, kind="stable")[:max_dets]
    return make_det_record(rec["image_id"], b[order], s[order], l[order])


def save_det_predictions(path: str | os.PathLike, predictions: Iterable[dict], *, dataset: str, split: str,
                         model: str, class_names: list[str], score_thr: float = DEFAULT_EXPORT_SCORE_THR,
                         max_dets: int = DEFAULT_MAX_DETS, extra: dict | None = None) -> None:
    payload = {
        "format": DET_FORMAT, "dataset": dataset, "split": split, "model": model,
        "class_names": list(class_names), "coordinate_system": COORD_SYSTEM,
        "score_threshold_export": score_thr, "max_detections_per_image": max_dets,
        "predictions": list(predictions),
    }
    if extra:
        payload["meta"] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def load_det_predictions(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format") != DET_FORMAT:
        raise ValueError(f"{path}: unexpected format {payload.get('format')}")
    return payload


def validate_det_predictions(payload: dict, image_sizes: dict[str, tuple[int, int]]) -> list[str]:
    """Return a list of problems (empty list = valid). ``image_sizes``: id -> (w, h)."""
    problems = []
    k = len(payload["class_names"])
    seen = set()
    for rec in payload["predictions"]:
        iid = rec["image_id"]
        if iid in seen:
            problems.append(f"duplicate image_id {iid}")
        seen.add(iid)
        if iid not in image_sizes:
            problems.append(f"unknown image_id {iid}")
            continue
        w, h = image_sizes[iid]
        b = np.asarray(rec["boxes_xyxy"], dtype=np.float64).reshape(-1, 4)
        if len(b) != len(rec["scores"]) or len(b) != len(rec["labels"]):
            problems.append(f"{iid}: length mismatch")
        if len(b):
            if not np.isfinite(b).all():
                problems.append(f"{iid}: non-finite box")
            if (b[:, 2] < b[:, 0]).any() or (b[:, 3] < b[:, 1]).any():
                problems.append(f"{iid}: x2<x1 or y2<y1")
            if (b[:, [0, 2]] < -1).any() or (b[:, [0, 2]] > w + 1).any() or \
               (b[:, [1, 3]] < -1).any() or (b[:, [1, 3]] > h + 1).any():
                problems.append(f"{iid}: box outside original image {w}x{h} (resize/letterbox not reverted?)")
        labels = rec["labels"]
        if labels and (min(labels) < 0 or max(labels) >= k):
            problems.append(f"{iid}: label outside [0,{k})")
        sc = rec["scores"]
        if sc and any((not math.isfinite(x)) or x < 0 or x > 1.0 + 1e-6 for x in sc):
            problems.append(f"{iid}: score outside [0,1]")
    missing = set(image_sizes) - seen
    if missing:
        problems.append(f"{len(missing)} images of the split have no prediction record")
    return problems


# --------------------------------------------------------------------------- segmentation
def save_seg_prediction(out_dir: str | os.PathLike, image_id: str, prob: np.ndarray, mode: str) -> str:
    """Store class probabilities (float16) AND the defect score map computed in float32 and stored as
    float32. The defect map must not be derived later from float16 P(background): near 1.0 float16 has a
    step of ~5e-4, which would collapse all low defect scores into ties (coarse pixel ROC / PRO curves)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = safe_id(image_id) + ".npz"
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim == 2:
        prob = prob[None]
    defect = prob[0] if mode == "binary" else 1.0 - prob[0]
    np.savez_compressed(out_dir / fn, prob=prob.astype(np.float16), defect=defect.astype(np.float32))
    return fn


def save_seg_index(out_dir: str | os.PathLike, files: dict[str, str], *, dataset: str, split: str, model: str,
                   class_names: list[str], mode: str, extra: dict | None = None) -> None:
    payload = {"format": SEG_FORMAT, "dataset": dataset, "split": split, "model": model,
               "class_names": list(class_names), "mode": mode, "files": files,
               "resolution": "original_image", "meta": extra or {}}
    with open(Path(out_dir) / "index.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def load_seg_index(out_dir: str | os.PathLike) -> dict:
    with open(Path(out_dir) / "index.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format") != SEG_FORMAT:
        raise ValueError("unexpected segmentation prediction format")
    return payload


def load_seg_prob(out_dir: str | os.PathLike, fn: str) -> np.ndarray:
    with np.load(Path(out_dir) / fn) as z:
        return z["prob"].astype(np.float32)


def load_seg_defect(out_dir: str | os.PathLike, fn: str, mode: str) -> np.ndarray:
    """Exact float32 defect score map (falls back to the float16 probabilities for old files)."""
    with np.load(Path(out_dir) / fn) as z:
        if "defect" in z.files:
            return z["defect"].astype(np.float32)
        return defect_score_map(z["prob"].astype(np.float32), mode)


def defect_score_map(prob: np.ndarray, mode: str) -> np.ndarray:
    """Continuous defect-vs-background score map from a probability tensor.

    binary: channel 0 is P(defect). multiclass: 1 - P(background) = sum of defect classes.
    """
    if mode == "binary":
        return prob[0]
    return 1.0 - prob[0]
