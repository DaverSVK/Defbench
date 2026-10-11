"""Common detection evaluation for ALL detectors (unified prediction format).

1. COCO-style metrics with pycocotools (AP computed from the full score ranking,
   101-point interpolation, IoU 0.50:0.05:0.95, maxDets 1/10/100):
   AP, AP50, AP75, per-class AP / AP50, AR@1, AR@10, AR@100.
2. Operating point (working point) metrics: class-aware ONE-TO-ONE greedy matching
   at IoU >= 0.50. Predictions are processed in descending score order; each one is
   matched to the unmatched ground-truth box of the SAME class with the highest IoU
   (>= 0.5). A second prediction on an already matched object is a false positive
   (duplicate). The confidence threshold is selected on the VALIDATION split
   (max F1, ties -> higher threshold) and then frozen for the test split.

COCO AR vs. working-point recall: COCO AR@k is the recall averaged over the ten
IoU thresholds 0.50:0.95, using the top-k detections per image with NO score
threshold; the working-point recall uses a single IoU (0.50) and only detections
above the frozen confidence threshold. They are therefore not interchangeable.
"""
from __future__ import annotations

import contextlib
import io
from collections import defaultdict

import numpy as np

from ..data.geometry import box_iou, xyxy_to_xywh

IOU_WORKING_POINT = 0.5


# ----------------------------------------------------------------------------- COCO
def build_coco_gt(records: list[dict], classes: list[str]) -> tuple[dict, dict]:
    images, anns = [], []
    idmap = {}
    aid = 1
    for n, r in enumerate(records, start=1):
        idmap[r["image_id"]] = n
        images.append({"id": n, "file_name": r["image_id"], "width": int(r["width"]), "height": int(r["height"])})
        for b, l in zip(r["boxes"], r["labels"]):
            x, y, w, h = xyxy_to_xywh(b)[0].tolist()
            anns.append({"id": aid, "image_id": n, "category_id": int(l) + 1, "bbox": [x, y, w, h],
                         "area": float(w * h), "iscrowd": 0})
            aid += 1
    cats = [{"id": i + 1, "name": c} for i, c in enumerate(classes)]
    return {"images": images, "annotations": anns, "categories": cats,
            "info": {"description": "defectbench GT", "category_id": "canonical label + 1"}}, idmap


def _coco_dets(pred_payload: dict, idmap: dict) -> list[dict]:
    dets = []
    for rec in pred_payload["predictions"]:
        if rec["image_id"] not in idmap:
            continue
        iid = idmap[rec["image_id"]]
        for b, s, l in zip(rec["boxes_xyxy"], rec["scores"], rec["labels"]):
            x, y, w, h = xyxy_to_xywh(b)[0].tolist()
            dets.append({"image_id": iid, "category_id": int(l) + 1, "bbox": [x, y, w, h], "score": float(s)})
    return dets


def coco_evaluate(records: list[dict], pred_payload: dict, classes: list[str], max_dets: int = 100) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_dict, idmap = build_coco_gt(records, classes)
    dets = _coco_dets(pred_payload, idmap)
    n_gt = len(gt_dict["annotations"])
    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100",
            "AR_small", "AR_medium", "AR_large"]
    if n_gt == 0:
        return {"coco": {k: None for k in keys}, "per_class": {},
                "note": "no ground-truth boxes in this split: COCO AP undefined"}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()
        if not dets:
            stats = [0.0] * 12
            per_class = {c: {"AP": 0.0, "AP50": 0.0, "n_gt": 0} for c in classes}
            ev = None
        else:
            coco_dt = coco_gt.loadRes(dets)
            ev = COCOeval(coco_gt, coco_dt, "bbox")
            ev.params.maxDets = [1, 10, max_dets]
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            stats = [float(x) for x in ev.stats]
    n_gt_cls = defaultdict(int)
    for a in gt_dict["annotations"]:
        n_gt_cls[a["category_id"] - 1] += 1
    per_class = {}
    for k, c in enumerate(classes):
        if n_gt_cls[k] == 0:
            per_class[c] = {"AP": None, "AP50": None, "AP75": None, "n_gt": 0,
                            "note": "no ground-truth instances of this class in the split"}
            continue
        if ev is None:
            per_class[c] = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "n_gt": n_gt_cls[k]}
            continue
        p = ev.eval["precision"][:, :, k, 0, -1]  # T x R, area=all, maxDets=max
        ap = float(p[p > -1].mean()) if (p > -1).any() else None
        p50 = p[0][p[0] > -1]
        p75 = p[5][p[5] > -1]
        per_class[c] = {"AP": ap, "AP50": float(p50.mean()) if len(p50) else None,
                        "AP75": float(p75.mean()) if len(p75) else None, "n_gt": n_gt_cls[k]}
    # pycocotools reports -1 for undefined entries (e.g. no small objects)
    coco = {k: (None if v < 0 else v) for k, v in zip(keys, stats)}
    return {"coco": coco, "per_class": per_class, "n_gt_boxes": n_gt, "n_images": len(records),
            "n_detections": len(dets), "max_dets": max_dets,
            "protocol": "pycocotools COCOeval bbox, IoU .50:.05:.95, 101-pt interpolated AP, maxDets [1,10,%d]" % max_dets}


# ----------------------------------------------------------------------------- matching
def match_predictions(records: list[dict], pred_payload: dict, iou_thr: float = IOU_WORKING_POINT) -> dict:
    """Class-aware one-to-one greedy matching of ALL exported detections.

    Because matching is greedy in descending score order, the result for any
    confidence threshold t equals the result of the full matching restricted to
    detections with score >= t. Hence one pass yields the whole PR/F1 curve.
    """
    preds = {p["image_id"]: p for p in pred_payload["predictions"]}
    det_rows = []   # (score, is_tp, label, image_id, error_type, box, iou)
    gt_rows = []    # (image_id, label, box, matched_by_score or None)
    for r in records:
        gb = np.asarray(r["boxes"], dtype=float).reshape(-1, 4)
        gl = np.asarray(r["labels"], dtype=int)
        p = preds.get(r["image_id"], {"boxes_xyxy": [], "scores": [], "labels": []})
        pb = np.asarray(p["boxes_xyxy"], dtype=float).reshape(-1, 4)
        ps = np.asarray(p["scores"], dtype=float)
        pl = np.asarray(p["labels"], dtype=int)
        order = np.argsort(-ps, kind="stable")
        ious = box_iou(pb, gb)
        matched = np.full(len(gb), -1.0)
        for i in order:
            same = np.nonzero(gl == pl[i])[0]
            cand = [j for j in same if matched[j] < 0 and ious[i, j] >= iou_thr]
            if cand:
                j = max(cand, key=lambda j: ious[i, j])
                matched[j] = ps[i]
                det_rows.append((ps[i], True, int(pl[i]), r["image_id"], "tp", pb[i].tolist(), float(ious[i, j])))
                continue
            # error taxonomy of the false positive
            best_same = max((ious[i, j] for j in same), default=0.0)
            other = np.nonzero(gl != pl[i])[0]
            best_other = max((ious[i, j] for j in other), default=0.0)
            if best_same >= iou_thr:
                et = "duplicate"
            elif best_other >= iou_thr:
                et = "classification"
            elif best_same >= 0.1:
                et = "localization"
            else:
                et = "background"
            det_rows.append((ps[i], False, int(pl[i]), r["image_id"], et, pb[i].tolist(), float(max(best_same, best_other))))
        for j in range(len(gb)):
            gt_rows.append((r["image_id"], int(gl[j]), gb[j].tolist(), float(matched[j]) if matched[j] >= 0 else None))
    return {"dets": det_rows, "gts": gt_rows, "iou_thr": iou_thr}


def curve_from_matches(m: dict, label: int | None = None) -> dict:
    dets = [d for d in m["dets"] if label is None or d[2] == label]
    n_gt = sum(1 for g in m["gts"] if label is None or g[1] == label)
    if not dets:
        return {"thresholds": np.array([]), "precision": np.array([]), "recall": np.array([]),
                "f1": np.array([]), "n_gt": n_gt}
    s = np.array([d[0] for d in dets])
    tp = np.array([d[1] for d in dets], dtype=float)
    o = np.argsort(-s, kind="stable")
    s, tp = s[o], tp[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    # evaluate only at the last index of each distinct score (threshold = that score)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    thr = s[last]
    prec = ctp[last] / np.maximum(ctp[last] + cfp[last], 1)
    rec = ctp[last] / max(n_gt, 1) if n_gt else np.zeros_like(thr)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return {"thresholds": thr, "precision": prec, "recall": rec, "f1": f1, "n_gt": n_gt}


def select_threshold(val_matches: dict) -> dict:
    c = curve_from_matches(val_matches)
    if len(c["thresholds"]) == 0:
        return {"threshold": 1.0, "val_f1": 0.0, "note": "no validation detections"}
    best = c["f1"].max()
    idx = np.nonzero(c["f1"] == best)[0]
    i = idx[np.argmax(c["thresholds"][idx])]  # tie -> highest threshold
    return {"threshold": float(c["thresholds"][i]), "val_f1": float(best),
            "val_precision": float(c["precision"][i]), "val_recall": float(c["recall"][i]),
            "rule": "argmax F1 on validation at IoU 0.50 (ties -> higher threshold)"}


def operating_point(m: dict, threshold: float, classes: list[str]) -> dict:
    def at(dets, n_gt):
        sel = [d for d in dets if d[0] >= threshold]
        tp = sum(1 for d in sel if d[1])
        fp = len(sel) - tp
        fn = n_gt - tp
        p = tp / (tp + fp) if tp + fp else (None if n_gt == 0 else 0.0)
        r = tp / n_gt if n_gt else None
        f1 = (2 * p * r / (p + r) if (p and r) else 0.0) if (p is not None and r is not None) else None
        return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}

    out = at(m["dets"], len(m["gts"]))
    out["threshold"] = threshold
    out["iou"] = m["iou_thr"]
    out["fp_by_type"] = dict(sorted(
        __import__("collections").Counter(d[4] for d in m["dets"] if d[0] >= threshold and not d[1]).items()))
    out["per_class"] = {}
    for k, c in enumerate(classes):
        n_gt = sum(1 for g in m["gts"] if g[1] == k)
        out["per_class"][c] = at([d for d in m["dets"] if d[2] == k], n_gt)
    return out


def error_lists(m: dict, threshold: float, limit: int = 200) -> dict:
    """False positives by type and false negatives at the working point (for visualisation)."""
    fps = [{"image_id": d[3], "label": d[2], "score": d[0], "box": d[5], "type": d[4], "iou": d[6]}
           for d in m["dets"] if d[0] >= threshold and not d[1]]
    fns = [{"image_id": g[0], "label": g[1], "box": g[2]} for g in m["gts"] if g[3] is None or g[3] < threshold]
    fps.sort(key=lambda x: -x["score"])
    return {"false_positives": fps[:limit], "false_negatives": fns[:limit],
            "n_false_positives": len(fps), "n_false_negatives": len(fns)}


def evaluate_detection(records_val: list[dict] | None, pred_val: dict | None, records_test: list[dict],
                       pred_test: dict, classes: list[str], threshold: float | None = None) -> dict:
    """Full detection evaluation. The working-point threshold comes from validation
    (or is passed explicitly, e.g. frozen from an earlier evaluation)."""
    out = {"test": coco_evaluate(records_test, pred_test, classes,
                                 max_dets=int(pred_test.get("max_detections_per_image", 100)))}
    if threshold is None:
        if records_val is None or pred_val is None:
            raise ValueError("validation predictions are required to select the working point")
        out["val"] = coco_evaluate(records_val, pred_val, classes,
                                   max_dets=int(pred_val.get("max_detections_per_image", 100)))
        sel = select_threshold(match_predictions(records_val, pred_val))
    else:
        sel = {"threshold": float(threshold), "rule": "provided (frozen)"}
    mt = match_predictions(records_test, pred_test)
    out["threshold_selection"] = sel
    out["working_point"] = operating_point(mt, sel["threshold"], classes)
    c = curve_from_matches(mt)
    out["pr_curve_iou50_test"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in c.items()}
    out["errors"] = error_lists(mt, sel["threshold"])
    return out
