"""Visualisations of ground truth vs. predictions at the frozen working point.

Detection colours: GT = green, TP = blue, FP = orange (label: error type), FN (missed GT) = red.
Segmentation: GT contour green, predicted mask (val-selected threshold) red overlay, plus the
continuous defect map.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..data.split import load_manifest, load_split, split_records
from ..data.torch_datasets import gt_class_map, load_image
from ..runs import load_run

COL = {"gt": (0, 200, 0), "tp": (30, 90, 255), "fp": (255, 150, 0), "fn": (230, 0, 0)}


def _draw_boxes(img, boxes, color, labels=None, width=2):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    for i, b in enumerate(boxes):
        d.rectangle([float(b[0]), float(b[1]), float(b[2]) - 1, float(b[3]) - 1], outline=color, width=width)
        if labels:
            d.text((float(b[0]) + 2, float(b[1]) + 1), labels[i], fill=color)
    return img


def _det_panels(run_cfg, res, recs, n_err=8, n_ok=4):
    from PIL import Image
    from ..evaluation.detection import match_predictions
    from ..formats import load_det_predictions
    rd = Path(run_cfg["run_dir"])
    pt = load_det_predictions(rd / "predictions" / "test_det.json")
    thr = res["working_point"]["threshold"]
    m = match_predictions(recs, pt)
    by_img = defaultdict(lambda: {"tp": [], "fp": [], "fn": []})
    for d in m["dets"]:
        if d[0] >= thr:
            by_img[d[3]]["tp" if d[1] else "fp"].append(d)
    for g in m["gts"]:
        if g[3] is None or g[3] < thr:
            by_img[g[0]]["fn"].append(g)
    classes = run_cfg["dataset_cfg"]["classes"]
    raw = Path(run_cfg["dataset_cfg"]["raw_root_abs"])
    err = sorted(recs, key=lambda r: -(len(by_img[r["image_id"]]["fp"]) + len(by_img[r["image_id"]]["fn"])))
    chosen = [r for r in err[:n_err] if by_img[r["image_id"]]["fp"] or by_img[r["image_id"]]["fn"]]
    ok = [r for r in recs if not by_img[r["image_id"]]["fp"] and not by_img[r["image_id"]]["fn"] and r["boxes"]][:n_ok]
    panels = []
    for r in chosen + ok:
        img = Image.fromarray(load_image(raw / r["file"]))
        s = max(1.0, 256 / min(img.size))
        img = img.resize((int(img.width * s), int(img.height * s)))
        sc = lambda b: [v * s for v in b]
        b = by_img[r["image_id"]]
        _draw_boxes(img, [sc(x) for x in r["boxes"]], COL["gt"])
        _draw_boxes(img, [sc(d[5]) for d in b["tp"]], COL["tp"], [f"{classes[d[2]]} {d[0]:.2f}" for d in b["tp"]])
        _draw_boxes(img, [sc(d[5]) for d in b["fp"]], COL["fp"], [f"FP:{d[4]} {classes[d[2]]} {d[0]:.2f}" for d in b["fp"]])
        _draw_boxes(img, [sc(g[2]) for g in b["fn"]], COL["fn"], [f"FN {classes[g[1]]}" for g in b["fn"]], width=3)
        panels.append((img, r["image_id"], "errors" if r in chosen else "correct"))
    return panels


def _grid(panels, path: Path, title: str, cols=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not panels:
        return None
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for a in axes:
        a.axis("off")
    for a, (img, iid, kind) in zip(axes, panels):
        a.imshow(img)
        a.set_title(f"{iid.split('/')[-1]} [{kind}]", fontsize=8)
    fig.suptitle(title, fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _seg_panels(run_cfg, res, recs, n=8):
    from PIL import Image
    from ..formats import load_seg_defect, load_seg_index
    rd = Path(run_cfg["run_dir"])
    idx = load_seg_index(rd / "predictions" / "test_seg")
    mode = idx["mode"]
    thr = res["threshold_selection"]["threshold"]
    raw = Path(run_cfg["dataset_cfg"]["raw_root_abs"])
    mthr = int((run_cfg["dataset_cfg"].get("mask") or {}).get("binarize_threshold", 128))
    scored = []
    for r in recs:
        d = load_seg_defect(rd / "predictions" / "test_seg", idx["files"][r["image_id"]], mode)
        g, _ = gt_class_map(r, raw, mthr)
        p = d >= thr
        err = int((p ^ g).sum())
        scored.append((err, r, d, g, p))
    scored.sort(key=lambda t: -t[0])
    pick = scored[: n // 2] + [s for s in scored[::-1] if s[3].any()][: n // 2]
    panels = []
    import cv2
    for err, r, d, g, p in pick:
        img = load_image(raw / r["file"]).astype(np.float32)
        ov = img.copy()
        ov[p] = 0.5 * ov[p] + 0.5 * np.array([255, 0, 0])
        cs, _ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ov = np.ascontiguousarray(ov.astype(np.uint8))
        cv2.drawContours(ov, cs, -1, (0, 200, 0), 1)
        heat = (np.stack([d, np.zeros_like(d), 1 - d], -1) * 255).astype(np.uint8)
        both = np.concatenate([ov, heat], axis=1)
        panels.append((Image.fromarray(both), r["image_id"], f"err px={err}"))
    return panels


def visualize_run(run_dir: Path, res: dict) -> list[Path]:
    run_cfg = load_run(run_dir)
    ds = run_cfg["dataset_cfg"]
    task = run_cfg["task"]
    from ..evaluation.run_eval import _records
    recs = _records(run_cfg, "test", task)
    outs = []
    if task in ("bbox", "instance_seg"):
        p = _grid(_det_panels(run_cfg, res, recs), Path(run_dir) / "figures" / "vis_test_errors.png",
                  "GT green | TP blue | FP orange (type) | FN red - working point frozen on val")
    else:
        p = _grid(_seg_panels(run_cfg, res, recs), Path(run_dir) / "figures" / "vis_test_segmentation.png",
                  "left: prediction (red) + GT contour (green); right: defect probability map")
    if p:
        outs.append(p)
    return outs
