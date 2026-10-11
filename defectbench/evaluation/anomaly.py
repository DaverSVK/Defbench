"""AUROC / AUPRC (image- and pixel-level) and AUPRO.

Conventions (recorded in every report)
--------------------------------------
* AUROC: area under the ROC curve (sklearn ``roc_auc_score``, exact, ties handled by
  the trapezoidal rule of the empirical ROC).
* AUPRC: the PRIMARY value is **average precision** (AP, step-wise sum
  ``sum_n (R_n - R_{n-1}) P_n``, sklearn ``average_precision_score``); it is NOT
  interpolated. A secondary value ``auprc_trapezoid`` (trapezoidal integration of the
  PR curve, sklearn ``auc(recall, precision)``) is stored separately and is known to
  be optimistic. Neither is the COCO detection AP.
* A metric is ``None`` together with a ``reason`` when it is undefined (only one class
  present, no ground-truth masks, no continuous score map, ...). No artificial
  negatives/positives are ever created.

Image-level anomaly detection: positive = image with >= 1 defect, negative = normal
image (annotated as defect-free). The image score aggregation rule is fixed BEFORE
testing in the model config (detectors: max defect confidence, 0 if no detection;
segmenters: max or mean of top-k pixels of the defect map).

Pixel-level: POOLED over all pixels of all images of the split (one ROC over the
concatenation) - not a mean of per-image AUROCs. Binary task defect-vs-background,
score = defect probability (multiclass: 1 - P(background)). Normal images contribute
only negative pixels.

AUPRO (Bergmann et al., MVTec AD): connected components (8-connectivity) of the
ground-truth defect mask of every image are the regions. For a threshold t the PRO is
the mean over ALL regions of |region ∩ {score >= t}| / |region|, and the FPR is the
fraction of ALL ground-truth background pixels (all images) with score >= t. The exact
curve is computed at every distinct score, integrated with the trapezoidal rule over
FPR in [0, fpr_limit] (default 0.30); the curve is linearly interpolated at
FPR = fpr_limit, and the area is normalised by fpr_limit (perfect map -> 1.0).
"""
from __future__ import annotations

import numpy as np

FPR_LIMIT_DEFAULT = 0.30
AUPRC_CONVENTION = ("primary: average precision (step-wise, sklearn average_precision_score); "
                    "secondary: trapezoidal area under PR curve (auprc_trapezoid)")


def _na(reason: str, **extra) -> dict:
    d = {"auroc": None, "auprc": None, "auprc_trapezoid": None, "reason": reason}
    d.update(extra)
    return d


def binary_auroc_auprc(labels, scores, curves: bool = False) -> dict:
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    y = np.asarray(labels).astype(int).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return _na(f"only one class present (positives={n_pos}, negatives={n_neg}); "
                   "AUROC/AUPRC undefined - no artificial samples were created", n_pos=n_pos, n_neg=n_neg)
    if not np.isfinite(s).all():
        return _na("non-finite scores", n_pos=n_pos, n_neg=n_neg)
    out = {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
           "n_pos": n_pos, "n_neg": n_neg, "prevalence": n_pos / (n_pos + n_neg),
           "auprc_convention": AUPRC_CONVENTION}
    p, r, _ = precision_recall_curve(y, s)
    out["auprc_trapezoid"] = float(auc(r, p))
    if curves:
        fpr, tpr, _ = roc_curve(y, s)
        k = max(1, len(fpr) // 2000)
        out["roc_curve"] = {"fpr": fpr[::k].tolist() + [float(fpr[-1])], "tpr": tpr[::k].tolist() + [float(tpr[-1])]}
        k = max(1, len(p) // 2000)
        out["pr_curve"] = {"precision": p[::k].tolist(), "recall": r[::k].tolist()}
    return out


# ------------------------------------------------------------------- image-level scores
def detector_image_score(pred_record: dict, rule: str = "max") -> float:
    """Image anomaly score from detections: max confidence over all defect classes, 0 if none."""
    sc = pred_record.get("scores", [])
    if rule != "max":
        raise ValueError(f"unsupported detector aggregation {rule}")
    return float(max(sc)) if sc else 0.0


def segmenter_image_score(defect_map: np.ndarray, rule: str = "max", topk: int = 100) -> float:
    """Aggregation of a defect probability map. rule in {max, topk_mean, mean}."""
    v = np.asarray(defect_map, dtype=np.float64).ravel()
    if rule == "max":
        return float(v.max())
    if rule == "topk_mean":
        k = min(topk, len(v))
        return float(np.partition(v, len(v) - k)[-k:].mean())
    if rule == "mean":
        return float(v.mean())
    raise ValueError(f"unknown aggregation {rule}")


def image_level_metrics(is_defective: list[bool], scores: list[float], rule: str) -> dict:
    out = binary_auroc_auprc(np.asarray(is_defective, dtype=int), np.asarray(scores), curves=True)
    out["aggregation_rule"] = rule
    out["task"] = "image-level: normal vs defective"
    return out


# ------------------------------------------------------------------- pixel-level
def pixel_level_metrics(gt_masks: list[np.ndarray] | None, score_maps: list[np.ndarray] | None,
                        max_pixels_for_curves: int = 200000) -> dict:
    if gt_masks is None:
        return _na("no ground-truth pixel masks for this dataset/task (boxes are never rasterised into masks)")
    if score_maps is None:
        return _na("model does not produce a continuous pixel score map")
    y = np.concatenate([np.asarray(m, dtype=bool).ravel() for m in gt_masks])
    s = np.concatenate([np.asarray(m, dtype=np.float32).ravel() for m in score_maps])
    out = binary_auroc_auprc(y, s, curves=False)
    out["aggregation"] = "pooled over all pixels of the split (not mean of per-image values)"
    out["n_pixels"] = int(len(y))
    if out.get("auroc") is not None:
        # store downsampled curves for plotting
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), size=min(len(y), max_pixels_for_curves), replace=False)
        c = binary_auroc_auprc(y[idx], s[idx], curves=True)
        out["roc_curve_subsampled"] = c.get("roc_curve")
        out["pr_curve_subsampled"] = c.get("pr_curve")
    return out


def compute_pro_curve(gt_masks: list[np.ndarray], score_maps: list[np.ndarray]):
    """Exact PRO-FPR curve (descending thresholds). Returns (fpr, pro, thresholds)."""
    import cv2
    scores_all, w_fpr, w_pro = [], [], []
    n_regions = 0
    n_neg = 0
    region_sizes = []
    per_img = []
    for gt, sm in zip(gt_masks, score_maps):
        g = np.asarray(gt, dtype=bool)
        s = np.asarray(sm, dtype=np.float64)
        if g.shape != s.shape:
            raise ValueError(f"mask/score shape mismatch {g.shape} vs {s.shape}")
        n, lab = cv2.connectedComponents(g.astype(np.uint8), connectivity=8)
        per_img.append((lab, n - 1, s, g))
        n_regions += n - 1
        n_neg += int((~g).sum())
    if n_regions == 0 or n_neg == 0:
        return None
    for lab, nr, s, g in per_img:
        neg = ~g
        scores_all.append(s[neg])
        w_fpr.append(np.full(int(neg.sum()), 1.0 / n_neg))
        w_pro.append(np.zeros(int(neg.sum())))
        if nr:
            areas = np.bincount(lab.ravel(), minlength=nr + 1)
            pos = lab > 0
            scores_all.append(s[pos])
            w_fpr.append(np.zeros(int(pos.sum())))
            w_pro.append(1.0 / (areas[lab[pos]] * n_regions))
            region_sizes.extend(areas[1:].tolist())
    s = np.concatenate(scores_all)
    wf = np.concatenate(w_fpr)
    wp = np.concatenate(w_pro)
    o = np.argsort(-s, kind="stable")
    s, wf, wp = s[o], wf[o], wp[o]
    fpr = np.cumsum(wf)
    pro = np.cumsum(wp)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    fpr = np.r_[0.0, fpr[last]]
    pro = np.r_[0.0, pro[last]]
    thr = np.r_[np.inf, s[last]]
    return np.clip(fpr, 0, 1), np.clip(pro, 0, 1), thr, n_regions, region_sizes


def trapezoid_limited(x: np.ndarray, y: np.ndarray, x_max: float) -> float:
    """Trapezoidal area under y(x) for x in [0, x_max]; linear interpolation at x_max."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x_max <= x[0]:
        return 0.0
    if x_max >= x[-1]:
        # extend the curve horizontally (cannot happen for a full curve where x ends at 1)
        x = np.r_[x, x_max]
        y = np.r_[y, y[-1]]
    i = np.searchsorted(x, x_max, side="right")  # first index with x > x_max
    xi, yi = x[:i], y[:i]
    if xi[-1] < x_max:
        x0, x1 = x[i - 1], x[i]
        y0, y1 = y[i - 1], y[i]
        y_int = y0 + (y1 - y0) * (x_max - x0) / (x1 - x0) if x1 > x0 else y1
        xi = np.r_[xi, x_max]
        yi = np.r_[yi, y_int]
    return float(np.trapezoid(yi, xi) if hasattr(np, "trapezoid") else np.trapz(yi, xi))


def aupro(gt_masks: list[np.ndarray] | None, score_maps: list[np.ndarray] | None,
          fpr_limit: float = FPR_LIMIT_DEFAULT) -> dict:
    if gt_masks is None:
        return {"aupro": None, "reason": "no ground-truth pixel masks"}
    if score_maps is None:
        return {"aupro": None, "reason": "no continuous score map"}
    res = compute_pro_curve(gt_masks, score_maps)
    if res is None:
        return {"aupro": None, "reason": "split has no ground-truth defect region or no background pixel"}
    fpr, pro, thr, n_regions, sizes = res
    # collapse vertical steps are fine for trapezoid; area normalised by fpr_limit
    area = trapezoid_limited(fpr, pro, fpr_limit) / fpr_limit
    k = max(1, len(fpr) // 3000)
    sel = np.r_[np.arange(0, len(fpr), k), len(fpr) - 1]
    fs = fpr[sel]
    beyond = np.nonzero(fs > fpr_limit)[0]
    keep = np.arange(len(fs)) <= (beyond[0] if len(beyond) else len(fs) - 1)  # include the interpolation segment
    return {"aupro": float(area), "fpr_limit": fpr_limit, "n_regions": int(n_regions),
            "region_size_px": {"min": int(min(sizes)), "median": float(np.median(sizes)), "max": int(max(sizes))},
            "definition": "mean per-region overlap vs pooled pixel FPR; exact curve; trapezoid on [0, limit], "
                          "linear interpolation at the limit, normalised by the limit",
            "curve": {"fpr": fpr[sel][keep].tolist(), "pro": pro[sel][keep].tolist()}}
