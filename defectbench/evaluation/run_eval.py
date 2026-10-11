"""Evaluate a run from its saved unified predictions (core environment only).

The evaluation never looks at checkpoints or framework code - only at
``predictions/{val,test}_*`` - so every number can be recomputed independently.
Thresholds (detection working point, segmentation binarisation, seg->det parameters)
are selected on VALIDATION and frozen before the test split is scored.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..data.split import load_manifest, load_split, split_records
from ..data.torch_datasets import gt_class_map
from ..formats import load_det_predictions, load_seg_defect, load_seg_index, load_seg_prob, validate_det_predictions
from ..runs import load_run
from ..utils.io import dump_json
from . import anomaly as A
from .detection import coco_evaluate, evaluate_detection
from .segmentation import SEG2DET_GRID, segmentation_metrics, select_binarization_threshold, seg_to_detections

log = logging.getLogger(__name__)


def _records(run_cfg, which, task):
    ds = run_cfg["dataset_cfg"]
    recs = split_records(load_manifest(ds), load_split(ds, run_cfg.get("split_name", "default")), which,
                         task="bbox" if task in ("bbox", "instance_seg") else None)
    sm = run_cfg["model_cfg"].get("smoke")
    if sm:  # same subset as the adapter used
        from ..models.adapters.base import BaseAdapter
        a = BaseAdapter(run_cfg)
        a.task = task
        recs = a.records(which)
    return recs


def _image_level_det(recs, payload, rule):
    preds = {p["image_id"]: p for p in payload["predictions"]}
    y = [bool(r["labels"]) or (r.get("mask_class") is not None and not r["is_normal"]) for r in recs]
    s = [A.detector_image_score(preds.get(r["image_id"], {"scores": []}), rule) for r in recs]
    return A.image_level_metrics(y, s, f"detector:{rule}")


def evaluate_detection_run(run_cfg: dict) -> dict:
    rd = Path(run_cfg["run_dir"])
    ds = run_cfg["dataset_cfg"]
    classes = ds["classes"]
    task = run_cfg["task"]
    rv, rt = _records(run_cfg, "val", task), _records(run_cfg, "test", task)
    pv = load_det_predictions(rd / "predictions" / "val_det.json")
    pt = load_det_predictions(rd / "predictions" / "test_det.json")
    for recs, p, n in ((rv, pv, "val"), (rt, pt, "test")):
        probs = validate_det_predictions(p, {r["image_id"]: (r["width"], r["height"]) for r in recs})
        if probs:
            raise RuntimeError(f"{n} predictions invalid: {probs[:5]}")
        if p["class_names"] != classes:
            raise RuntimeError(f"class order mismatch: {p['class_names']} vs {classes}")
    res = evaluate_detection(rv, pv, rt, pt, classes)
    gt_src = (ds.get("tasks") or {}).get("bbox")
    res["gt_boxes"] = gt_src
    if gt_src == "derived_from_mask":
        res["gt_boxes_note"] = "ground-truth boxes are DERIVED from semantic masks (connected components)"
    rule = run_cfg["model_cfg"].get("image_score_rule", "max")
    if (ds.get("tasks") or {}).get("normal_images"):
        res["image_level"] = _image_level_det(rt, pt, rule)
    else:
        res["image_level"] = {"auroc": None, "auprc": None, "reason": "dataset has no normal (defect-free) images"}
    res["pixel_level"] = {"auroc": None, "auprc": None,
                          "reason": "detector outputs boxes only; no pixel score map is derived from boxes"}
    res["aupro"] = {"aupro": None, "reason": "requires a continuous pixel score map"}
    return res


def _load_gt_masks(recs, raw_root, thr):
    gms, gcs = [], []
    for r in recs:
        m, cm = gt_class_map(r, raw_root, thr)
        gms.append(m)
        gcs.append(cm)
    return gms, gcs


def _load_probs(pred_dir, recs):
    idx = load_seg_index(pred_dir)
    probs = [load_seg_prob(pred_dir, idx["files"][r["image_id"]]) for r in recs]
    defects = [load_seg_defect(pred_dir, idx["files"][r["image_id"]], idx["mode"]) for r in recs]
    return probs, defects, idx["mode"]


def evaluate_segmentation_run(run_cfg: dict) -> dict:
    rd = Path(run_cfg["run_dir"])
    ds = run_cfg["dataset_cfg"]
    classes = ds["classes"]
    raw = Path(ds["raw_root_abs"])
    thr_mask = int((ds.get("mask") or {}).get("binarize_threshold", 128))
    rv, rt = _records(run_cfg, "val", "seg"), _records(run_cfg, "test", "seg")
    if (ds.get("tasks") or {}).get("mask") != "annotation":
        return _evaluate_weak_segmentation_run(run_cfg, rd, rv, rt, classes, ds)
    pv, dmv, mode = _load_probs(rd / "predictions" / "val_seg", rv)
    pt, dmt, mode_t = _load_probs(rd / "predictions" / "test_seg", rt)
    assert mode == mode_t
    gv, gcv = _load_gt_masks(rv, raw, thr_mask)
    gt, gct = _load_gt_masks(rt, raw, thr_mask)
    for g, p, r in zip(gt, pt, rt):
        if p.shape[1:] != g.shape:
            raise RuntimeError(f"{r['image_id']}: prediction {p.shape} not at original resolution {g.shape}")
    sel = select_binarization_threshold(gv, dmv)
    res = {"threshold_selection": sel, "mode": mode}
    res["val"] = segmentation_metrics(gv, gcv, pv, mode, sel["threshold"], classes, defect_maps=dmv)
    res["test"] = segmentation_metrics(gt, gct, pt, mode, sel["threshold"], classes, defect_maps=dmt)
    res["pixel_level"] = A.pixel_level_metrics(gt, dmt)
    res["aupro"] = A.aupro(gt, dmt, float(run_cfg["model_cfg"].get("aupro_fpr_limit", A.FPR_LIMIT_DEFAULT)))
    agg = run_cfg["model_cfg"].get("image_score_rule", {"rule": "max"})
    if isinstance(agg, str):
        agg = {"rule": agg}
    if (ds.get("tasks") or {}).get("normal_images"):
        y = [not r["is_normal"] for r in rt]
        s = [A.segmenter_image_score(d, agg["rule"], int(agg.get("topk", 100))) for d in dmt]
        res["image_level"] = A.image_level_metrics(y, s, f"segmenter:{agg['rule']}" +
                                                   (f"(k={agg.get('topk', 100)})" if agg["rule"] == "topk_mean" else ""))
    else:
        res["image_level"] = {"auroc": None, "auprc": None, "reason": "dataset has no normal images"}
    # --------------------------------------------- segmentation-derived detection (flagged)
    if (ds.get("tasks") or {}).get("bbox") in ("derived_from_mask", "annotation"):
        res["segmentation_derived_detection"] = _seg_derived_detection(rv, pv, dmv, rt, pt, dmt, mode, classes, ds)
    return res


def _evaluate_weak_segmentation_run(run_cfg, rd, rv, rt, classes, ds):
    """Weakly-supervised segmenter on a box-only dataset: no ground-truth pixel masks exist, so pixel metrics
    are N/A; the model is evaluated ONLY as segmentation-derived detection against the real boxes."""
    if not run_cfg["model_cfg"].get("weak_box_supervision"):
        raise RuntimeError("segmentation run on a dataset without masks that is not flagged weakly supervised")
    pv, dmv, mode = _load_probs(rd / "predictions" / "val_seg", rv)
    pt, dmt, _ = _load_probs(rd / "predictions" / "test_seg", rt)
    na = "no ground-truth pixel masks (weakly-supervised run trained on box-filled pseudo masks)"
    res = {"experiment": "WEAKLY SUPERVISED segmentation (box-filled pseudo masks)", "mode": mode,
           "test": {"pooled": None, "reason": na}, "pixel_level": {"auroc": None, "auprc": None, "reason": na},
           "aupro": {"aupro": None, "reason": na},
           "image_level": {"auroc": None, "auprc": None, "reason": "dataset has no normal images"}}
    res["segmentation_derived_detection"] = _seg_derived_detection(rv, pv, dmv, rt, pt, dmt, mode, classes, ds)
    res["segmentation_derived_detection"]["label"] = "segmentation-derived detection (weakly supervised)"
    return res


def _seg_derived_detection(rv, pv, dmv, rt, pt, dmt, mode, classes, ds):
    """Grid-search threshold / min_area / score rule on VAL (COCO AP), evaluate once on test."""
    cls = classes if mode == "multiclass" else ["defect"]

    def recs_for(recs):
        if mode == "multiclass":
            return recs
        return [{**r, "labels": [0] * len(r["labels"])} for r in recs]

    def payload(recs, probs, defects, t, a, s):
        preds = []
        for r, p, dm in zip(recs, probs, defects):
            d = seg_to_detections(p, mode, t, a, s, defect=dm)
            preds.append({"image_id": r["image_id"], **d})
        return {"predictions": preds, "max_detections_per_image": 100}

    best = None
    for t in SEG2DET_GRID["threshold"]:
        for a in SEG2DET_GRID["min_area"]:
            for s in SEG2DET_GRID["score"]:
                ap = coco_evaluate(recs_for(rv), payload(rv, pv, dmv, t, a, s), cls)["coco"]["AP"] or 0.0
                if best is None or ap > best[0]:
                    best = (ap, t, a, s)
    _, t, a, s = best
    out = evaluate_detection(recs_for(rv), payload(rv, pv, dmv, t, a, s), recs_for(rt), payload(rt, pt, dmt, t, a, s), cls)
    out["label"] = "segmentation-derived detection"
    out["selected_on_val"] = {"threshold": t, "min_area": a, "region_score": s, "val_AP": best[0],
                              "grid": SEG2DET_GRID}
    out["class_agnostic"] = mode != "multiclass"
    out["gt_boxes"] = (ds.get("tasks") or {}).get("bbox")
    return out


def evaluate_run(run_dir: str | Path) -> dict:
    run_cfg = load_run(run_dir)
    task = run_cfg["task"]
    if task in ("bbox", "instance_seg"):
        res = evaluate_detection_run(run_cfg)
    elif task == "seg":
        res = evaluate_segmentation_run(run_cfg)
    else:
        raise ValueError(task)
    res["run"] = {k: run_cfg[k] for k in ("dataset", "model", "task", "seed", "split_hash", "manifest_hash",
                                          "profile", "smoke")}
    res["conventions"] = {"auprc": A.AUPRC_CONVENTION, "coco_ap": "pycocotools 101-point interpolated",
                          "working_point": "class-aware one-to-one greedy matching, IoU 0.5, threshold from val"}
    dump_json(res, Path(run_dir) / "eval" / "metrics.json")
    try:
        from ..reporting.plots import plot_run
        plot_run(Path(run_dir), res)
    except Exception as e:  # plotting must never invalidate an evaluation
        log.warning("plotting failed: %s", e)
    return res
