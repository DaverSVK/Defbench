"""summarize: collect all evaluated runs -> results.csv/json, per-dataset tables, plots, Markdown report.

Rules implemented here
* datasets are NEVER averaged together; every table is per dataset,
* bbox detection, segmentation, image-level anomaly detection and segmentation-derived detection
  are separate tables,
* mean and sample standard deviation (ddof=1) over training seeds; with a single seed the std is
  reported as "n/a (1 seed)" - no variability is invented,
* runs on different split hashes of the same dataset are never aggregated together (flagged),
* metrics that are undefined are shown as "N/A" with their reason in a separate section,
* smoke-test runs are excluded unless explicitly requested.
"""
from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..utils.io import dump_json, load_json


def _get(d, path, default=None):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


DET_METRICS = [("AP", "test.coco.AP"), ("AP50", "test.coco.AP50"), ("AP75", "test.coco.AP75"),
               ("AR1", "test.coco.AR1"), ("AR10", "test.coco.AR10"), ("AR100", "test.coco.AR100"),
               ("P@wp", "working_point.precision"), ("R@wp", "working_point.recall"), ("F1@wp", "working_point.f1"),
               ("thr_wp", "working_point.threshold")]
SEG_METRICS = [("Dice", "test.pooled.dice"), ("IoU", "test.pooled.iou"), ("Prec", "test.pooled.precision"),
               ("Rec", "test.pooled.recall"), ("Dice_img", "test.per_image_mean.dice"),
               ("mDice_cls", "test.mean_class_dice"), ("pxAUROC", "pixel_level.auroc"),
               ("pxAUPRC", "pixel_level.auprc"), ("AUPRO30", "aupro.aupro"), ("thr", "threshold_selection.threshold")]
IMG_METRICS = [("imgAUROC", "image_level.auroc"), ("imgAUPRC", "image_level.auprc"),
               ("imgAUPRC_trapz", "image_level.auprc_trapezoid")]
S2D_METRICS = [("AP", "segmentation_derived_detection.test.coco.AP"),
               ("AP50", "segmentation_derived_detection.test.coco.AP50"),
               ("F1@wp", "segmentation_derived_detection.working_point.f1")]


def collect(runs_root: Path, include_smoke: bool = False) -> list[dict]:
    rows = []
    for mp in sorted(runs_root.glob("*/*/*/*/eval/metrics.json")):
        rd = mp.parent.parent
        rc = load_json(rd / "run_config.json")
        if rc.get("smoke") and not include_smoke:
            continue
        res = load_json(mp)
        ts = load_json(rd / "train_summary.json") if (rd / "train_summary.json").exists() else {}
        lat = load_json(rd / "benchmark" / "latency.json") if (rd / "benchmark" / "latency.json").exists() else {}
        params = ts.get("parameters") or (load_json(rd / "params.json") if (rd / "params.json").exists() else {})
        mc = rc["model_cfg"]
        m = mc.get("model", {})
        res_in = (m.get("input_size") or m.get("imgsz") or m.get("resolution") or
                  (f"min{m.get('min_size')}/max{m.get('max_size')}" if m.get("min_size") else None) or
                  ("native" if m.get("variant") else None))
        row = {"dataset": rc["dataset"], "model": rc["model"], "task": rc["task"], "seed": rc["seed"],
               "split_hash": rc["split_hash"], "manifest_hash": rc["manifest_hash"], "profile": rc.get("profile"),
               "tag": rc.get("tag"), "run_dir": str(rd),
               "pretraining": _get(mc, "pretrained.source") or _get(mc, "pretrained.init"),
               "input_resolution": str(res_in), "params_total": params.get("total"),
               "train_time_s": ts.get("train_time_s"),
               "peak_gpu_mem_gb": ts.get("peak_gpu_mem_allocated_gb"),
               "best_epoch": ts.get("best_epoch"), "stopped_early": ts.get("stopped_early"),
               "latency_model_ms_median": _get(lat, "model_ms.median"), "latency_model_ms_p95": _get(lat, "model_ms.p95"),
               "latency_e2e_ms_median": _get(lat, "end2end_ms.median"), "latency_e2e_ms_p95": _get(lat, "end2end_ms.p95"),
               "fps_e2e": lat.get("fps_end2end"), "latency_hw": _get(lat, "hardware.gpu"),
               "latency_precision": lat.get("precision"), "gt_boxes": res.get("gt_boxes")}
        for group in (DET_METRICS, SEG_METRICS, IMG_METRICS, S2D_METRICS):
            prefix = "s2d_" if group is S2D_METRICS else ""
            for name, path in group:
                row[prefix + name] = _get(res, path)
        row["_reasons"] = {k: v.get("reason") for k, v in res.items() if isinstance(v, dict) and v.get("reason")}
        row["_per_class"] = _get(res, "test.per_class")
        rows.append(row)
    return rows


DIGITS = {"train_time_s": 0, "peak_gpu_mem_gb": 2, "latency_e2e_ms_median": 2, "latency_model_ms_median": 2,
          "latency_e2e_ms_p95": 2, "thr_wp": 3, "thr": 3}


def _fmt(vals, key: str = ""):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return "N/A", None, None
    d = DIGITS.get(key, 4)
    mu = float(np.mean(vals))
    if len(vals) == 1:
        return f"{mu:.{d}f} (n=1)", mu, None
    sd = float(np.std(vals, ddof=1))
    return f"{mu:.{d}f} ± {sd:.{d}f} (n={len(vals)})", mu, sd


def aggregate(rows: list[dict], metrics: list[tuple[str, str]], prefix: str = "") -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["model"], r["task"], r["split_hash"])].append(r)
    out = []
    for (ds, model, task, sh), rs in sorted(groups.items()):
        a = {"dataset": ds, "model": model, "task": task, "split_hash": sh, "seeds": sorted(r["seed"] for r in rs)}
        for name, _ in metrics:
            s, mu, sd = _fmt([r.get(prefix + name) for r in rs], name)
            a[name] = s
            a[name + "_mean"] = mu
            a[name + "_std"] = sd
        for k in ("params_total", "input_resolution", "pretraining"):
            a[k] = rs[0].get(k)
        for k in ("train_time_s", "peak_gpu_mem_gb", "latency_e2e_ms_median", "latency_model_ms_median",
                  "latency_e2e_ms_p95"):
            s, mu, sd = _fmt([r.get(k) for r in rs], k)
            a[k] = s
            a[k + "_mean"] = mu
        out.append(a)
    return out


def _md_table(rows, cols):
    if not rows:
        return "_no runs_\n"
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        L.append("| " + " | ".join(str(r.get(c, "")) if r.get(c) is not None else "N/A" for c in cols) + " |")
    return "\n".join(L) + "\n"


def plot_accuracy_latency(agg_det: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outs = []
    by_ds = defaultdict(list)
    for a in agg_det:
        if a.get("AP_mean") is not None and a.get("latency_e2e_ms_median_mean") is not None:
            by_ds[a["dataset"]].append(a)
    for ds, rs in by_ds.items():
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for a in rs:
            ax.scatter(a["latency_e2e_ms_median_mean"], a["AP_mean"])
            ax.annotate(a["model"], (a["latency_e2e_ms_median_mean"], a["AP_mean"]), fontsize=7)
        ax.set_xlabel("end-to-end latency, median ms (batch 1)"); ax.set_ylabel("COCO AP@[.50:.95] (test)")
        ax.set_title(f"{ds}: accuracy vs latency")
        p = out_dir / f"accuracy_vs_latency_{ds}.png"
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        outs.append(p)
    return outs


def summarize(runs_root: Path, out_dir: Path, include_smoke: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect(runs_root, include_smoke)
    flat = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    dump_json(rows, out_dir / "results.json")
    if flat:
        keys = list(dict.fromkeys(k for r in flat for k in r))
        with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(flat)
    det = [r for r in rows if r["task"] in ("bbox", "instance_seg")]
    seg = [r for r in rows if r["task"] == "seg"]
    agg_det = aggregate(det, DET_METRICS)
    agg_seg = aggregate(seg, SEG_METRICS)
    agg_img = aggregate([r for r in rows if r.get("imgAUROC") is not None or r["dataset"] == "magnetic_tile"], IMG_METRICS)
    agg_s2d = aggregate([r for r in seg if r.get("s2d_AP") is not None], S2D_METRICS, prefix="s2d_")
    dump_json({"bbox_detection": agg_det, "segmentation": agg_seg, "image_level": agg_img,
               "segmentation_derived_detection": agg_s2d}, out_dir / "results_aggregated.json")
    figs = plot_accuracy_latency(agg_det, out_dir)
    split_conflicts = defaultdict(set)
    for r in rows:
        split_conflicts[r["dataset"]].add(r["split_hash"])
    L = [f"# Results report", "", f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} from `{runs_root}` "
         f"({len(rows)} evaluated runs{', smoke runs included' if include_smoke else ''}).", "",
         "Values are mean ± sample std (ddof=1) over training seeds; n = number of seeds. With n=1 no "
         "standard deviation is reported (none is invented). Datasets are never averaged together. Only runs whose training AND evaluation actually "
         "finished appear here.", ""]
    for ds, hs in split_conflicts.items():
        if len(hs) > 1:
            L.append(f"**WARNING**: {ds} has runs on {len(hs)} different splits {sorted(hs)} - they are in separate rows.")
    for ds in sorted({r["dataset"] for r in rows}):
        L += [f"## {ds}", ""]
        d = [a for a in agg_det if a["dataset"] == ds]
        if d:
            gt = {r.get("gt_boxes") for r in det if r["dataset"] == ds}
            L += ["### Bounding-box detection (test split)", ""]
            if "derived_from_mask" in gt:
                L += ["> Ground-truth boxes of this dataset are DERIVED from semantic masks (connected components).", ""]
            L.append(_md_table(d, ["model", "task", "seeds", "AP", "AP50", "AP75", "AR1", "AR10", "AR100", "P@wp",
                                   "R@wp", "F1@wp", "input_resolution", "params_total", "train_time_s",
                                   "peak_gpu_mem_gb", "latency_e2e_ms_median", "latency_model_ms_median"]))
        s = [a for a in agg_seg if a["dataset"] == ds]
        if s:
            L += ["### Segmentation (test split, pooled pixels)", "",
                  _md_table(s, ["model", "seeds", "Dice", "IoU", "Prec", "Rec", "Dice_img", "mDice_cls", "pxAUROC",
                                "pxAUPRC", "AUPRO30", "params_total", "train_time_s", "peak_gpu_mem_gb",
                                "latency_e2e_ms_median"])]
        i = [a for a in agg_img if a["dataset"] == ds and a.get("imgAUROC_mean") is not None]
        if i:
            L += ["### Image-level anomaly detection (normal vs defective, test split)", "",
                  _md_table(i, ["model", "task", "seeds", "imgAUROC", "imgAUPRC", "imgAUPRC_trapz"])]
        x = [a for a in agg_s2d if a["dataset"] == ds]
        if x:
            L += ["### Segmentation-derived detection (connected components of predicted masks)", "",
                  _md_table(x, ["model", "seeds", "AP", "AP50", "F1@wp"])]
        # per-class
        pcs = [r for r in rows if r["dataset"] == ds and r.get("_per_class")]
        if pcs:
            L += ["### Per-class results (per run)", ""]
            for r in pcs:
                pc = r["_per_class"]
                items = ", ".join(f"{k}: {_short(v)}" for k, v in pc.items())
                L.append(f"- {r['model']} seed {r['seed']}: {items}")
            L.append("")
        reasons = [(r["model"], r["seed"], r["_reasons"]) for r in rows if r["dataset"] == ds and r["_reasons"]]
        if reasons:
            L += ["### Unavailable metrics and reasons", ""]
            seen = set()
            for m, sd, rs in reasons:
                for k, v in rs.items():
                    key = (m, k, v)
                    if key not in seen:
                        seen.add(key)
                        L.append(f"- {m} / {k}: {v}")
            L.append("")
    L += ["## Conventions", "",
          "- COCO AP/AR: pycocotools, IoU 0.50:0.05:0.95, maxDets 1/10/100, computed over the full score ranking "
          "(export threshold 0.001, max 100 detections per image for every detector).",
          "- Working point: class-aware one-to-one greedy matching at IoU 0.50 (duplicates = FP); confidence "
          "threshold = argmax F1 on the validation split, frozen for test.",
          "- COCO AR@k averages recall over IoU 0.50:0.95 with the top-k detections and no score threshold; "
          "R@wp is recall at IoU 0.50 above the frozen threshold - they are different quantities.",
          "- AUPRC: average precision (step-wise, sklearn) is the primary value; imgAUPRC_trapz is the "
          "trapezoidal PR area. Neither is COCO AP.",
          "- Pixel AUROC/AUPRC pooled over all test pixels; AUPRO integrated over FPR in [0, 0.3], "
          "normalised by 0.3, linear interpolation at the limit.",
          "- Latency: batch 1, warmup, CUDA-synchronised, median and p95; FPS = 1000 / median latency.", ""]
    if figs:
        L += ["## Figures", ""] + [f"- ![{p.stem}]({p.name})" for p in figs] + [""]
    out = out_dir / "report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def _short(v):
    if isinstance(v, dict):
        for k in ("AP", "dice"):
            if k in v:
                return "N/A" if v[k] is None else f"{v[k]:.3f}"
    return str(v)
