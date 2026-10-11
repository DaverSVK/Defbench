"""Segmentation metrics, validation-selected binarisation and segmentation-derived detection.

Definitions
-----------
* Defect map d(x) = P(defect) (binary models) or 1 - P(background) (multiclass).
* Binarisation: pixel is predicted defective if d(x) >= t. t is chosen on the
  VALIDATION split (grid step 0.001, maximise pooled Dice) and frozen for test.
* Multiclass label of a predicted defect pixel: argmax over the defect channels.
* Pooled ("micro") Dice / IoU / precision / recall: TP, FP, FN summed over all pixels
  of the split - the primary numbers.
* Per-image mean Dice / IoU with explicit empty-mask rules:
    GT empty & prediction empty      -> 1.0 (correct negative image)
    GT empty & prediction non-empty  -> 0.0
    GT non-empty & prediction empty  -> 0.0
  Per-image precision is undefined when the prediction is empty; such images are
  excluded from the mean precision (their count is reported), analogously recall
  for empty GT.
* Per-class metrics (multiclass): pooled over pixels of class c; None + reason if the
  class has neither GT nor predicted pixels in the split.
"""
from __future__ import annotations

import numpy as np

N_BINS = 1001  # threshold grid 0.000 .. 1.000


def _hist(gt_masks, defect_maps):
    pos = np.zeros(N_BINS, np.int64)
    neg = np.zeros(N_BINS, np.int64)
    for g, d in zip(gt_masks, defect_maps):
        b = np.clip((np.asarray(d, np.float64) * (N_BINS - 1)).round().astype(np.int64), 0, N_BINS - 1)
        g = np.asarray(g, bool)
        pos += np.bincount(b[g], minlength=N_BINS)
        neg += np.bincount(b[~g], minlength=N_BINS)
    return pos, neg


def select_binarization_threshold(gt_masks, defect_maps) -> dict:
    pos, neg = _hist(gt_masks, defect_maps)
    tp = np.cumsum(pos[::-1])[::-1]          # pixels with bin >= t
    fp = np.cumsum(neg[::-1])[::-1]
    fn = pos.sum() - tp
    dice = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    if pos.sum() == 0:
        return {"threshold": 0.5, "note": "no positive pixels in validation - default 0.5 used"}
    i = int(np.argmax(dice))
    return {"threshold": i / (N_BINS - 1), "val_pooled_dice": float(dice[i]),
            "rule": "argmax pooled Dice on validation, grid step 0.001"}


def _prf(tp, fp, fn):
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else None
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else None
    prec = tp / (tp + fp) if (tp + fp) > 0 else None
    rec = tp / (tp + fn) if (tp + fn) > 0 else None
    return {"dice": dice, "iou": iou, "precision": prec, "recall": rec, "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def segmentation_metrics(gt_masks, gt_classes, probs, mode: str, threshold: float, class_names: list[str],
                         defect_maps=None) -> dict:
    """gt_masks: list of bool HxW; gt_classes: list of int class map HxW (0=bg, c+1 = class c) or None;
    probs: list of CxHxW probability arrays at original resolution."""
    from ..formats import defect_score_map
    TP = FP = FN = 0
    per_img = {"dice": [], "iou": [], "precision": [], "recall": []}
    n_empty_pred = n_empty_gt = 0
    K = len(class_names)
    cls_counts = np.zeros((K, 3), np.int64)
    for i, (g, pr) in enumerate(zip(gt_masks, probs)):
        g = np.asarray(g, bool)
        d = defect_maps[i] if defect_maps is not None else defect_score_map(pr, mode)
        p = d >= threshold
        tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
        TP += tp; FP += fp; FN += fn
        gs, ps = g.any(), p.any()
        if not gs and not ps:
            per_img["dice"].append(1.0); per_img["iou"].append(1.0)
        elif not gs or not ps:
            per_img["dice"].append(0.0); per_img["iou"].append(0.0)
        else:
            per_img["dice"].append(2 * tp / (2 * tp + fp + fn)); per_img["iou"].append(tp / (tp + fp + fn))
        if ps:
            per_img["precision"].append(tp / (tp + fp))
        else:
            n_empty_pred += 1
        if gs:
            per_img["recall"].append(tp / (tp + fn))
        else:
            n_empty_gt += 1
        if mode == "multiclass" and gt_classes is not None:
            pc = np.argmax(pr[1:], axis=0) + 1
            pc = np.where(p, pc, 0)
            gc = np.asarray(gt_classes[i])
            for k in range(K):
                a, b = pc == k + 1, gc == k + 1
                cls_counts[k] += [int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum())]
    out = {"threshold": threshold, "mode": mode,
           "pooled": _prf(TP, FP, FN),
           "per_image_mean": {k: (float(np.mean(v)) if v else None) for k, v in per_img.items()},
           "per_image_counts": {"images": len(gt_masks), "empty_prediction": n_empty_pred, "empty_gt": n_empty_gt},
           "empty_rules": "per-image Dice/IoU: both empty=1, exactly one empty=0; precision skipped if prediction "
                          "empty, recall skipped if GT empty"}
    if mode == "multiclass" and gt_classes is not None:
        pcs = {}
        for k, c in enumerate(class_names):
            tp, fp, fn = cls_counts[k]
            if tp + fp + fn == 0:
                pcs[c] = {"dice": None, "iou": None, "precision": None, "recall": None,
                          "reason": "class has neither GT nor predicted pixels in this split"}
            else:
                pcs[c] = _prf(tp, fp, fn)
        out["per_class"] = pcs
        vals = [v["dice"] for v in pcs.values() if v["dice"] is not None]
        out["mean_class_dice"] = float(np.mean(vals)) if vals else None
    return out


# ----------------------------------------------------------------- segmentation-derived detection
def seg_to_detections(prob: np.ndarray, mode: str, threshold: float, min_area: int, score_rule: str,
                      connectivity: int = 8, defect: np.ndarray | None = None) -> dict:
    """Connected components of the binarised defect map -> boxes (pixel-edge convention)."""
    import cv2
    from ..formats import defect_score_map
    d = defect if defect is not None else defect_score_map(prob, mode)
    b = (d >= threshold).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=connectivity)
    boxes, scores, labels = [], [], []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        m = lab == i
        s = float(d[m].max()) if score_rule == "max" else float(d[m].mean())
        if mode == "multiclass":
            cls = int(np.argmax(prob[1:, m].mean(axis=1)))
        else:
            cls = 0
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
        scores.append(s)
        labels.append(cls)
    return {"boxes_xyxy": boxes, "scores": scores, "labels": labels}


SEG2DET_GRID = {"threshold": [0.3, 0.4, 0.5, 0.6, 0.7], "min_area": [1, 5, 10, 25, 50], "score": ["max", "mean"]}
