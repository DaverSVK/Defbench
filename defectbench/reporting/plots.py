"""Per-run figures: PR/ROC/PRO curves, training curves, per-class AP, visualisations."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_training_curves(run_dir: Path) -> Path | None:
    p = run_dir / "train_log.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        return None

    def col(k):
        xs, ys = [], []
        for r in rows:
            try:
                xs.append(float(r.get("epoch", len(xs))))
                ys.append(float(r[k]))
            except (KeyError, TypeError, ValueError):
                xs.pop() if len(xs) > len(ys) else None
        return xs, ys

    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, lab in (("train_loss", "train loss"), ("val_loss", "val loss")):
        x, y = col(k)
        if y:
            ax[0].plot(x, y, label=lab)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("losses")
    x, y = col("val_metric")
    if y:
        ax[1].plot(x, y, color="tab:green")
        ax[1].set_title("validation metric (model selection)")
    ax[1].set_xlabel("epoch")
    out = run_dir / "figures" / "training_curves.png"
    _save(fig, out)
    return out


def plot_run(run_dir: Path, res: dict) -> list[Path]:
    plt = _plt()
    outs = []
    t = plot_training_curves(run_dir)
    if t:
        outs.append(t)
    fig_dir = run_dir / "figures"
    # detection PR at IoU 0.5 with working point
    c = res.get("pr_curve_iou50_test")
    if c and c.get("recall"):
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(c["recall"], c["precision"])
        wp = res["working_point"]
        if wp.get("recall") is not None and wp.get("precision") is not None:
            ax.scatter([wp["recall"]], [wp["precision"]], color="red", zorder=3,
                       label=f"val-selected thr={wp['threshold']:.3f}")
            ax.legend(loc="lower left")
        ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title("test PR, IoU 0.50, class-aware 1:1 matching")
        _save(fig, fig_dir / "pr_iou50_test.png")
        outs.append(fig_dir / "pr_iou50_test.png")
    pc = (res.get("test") or {}).get("per_class")
    if pc and isinstance(next(iter(pc.values())), dict) and "AP" in next(iter(pc.values())):
        names = list(pc)
        vals = [pc[n]["AP"] if pc[n]["AP"] is not None else np.nan for n in names]
        fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(names)), 3.5))
        ax.bar(range(len(names)), vals)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_ylabel("AP@[.50:.95]"); ax.set_title("per-class AP (test)")
        _save(fig, fig_dir / "per_class_ap.png")
        outs.append(fig_dir / "per_class_ap.png")
    # image-level ROC/PR
    il = res.get("image_level") or {}
    if il.get("roc_curve"):
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
        ax[0].plot(il["roc_curve"]["fpr"], il["roc_curve"]["tpr"]); ax[0].plot([0, 1], [0, 1], "k--", lw=0.6)
        ax[0].set_title(f"image ROC, AUROC={il['auroc']:.3f}"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
        ax[1].plot(il["pr_curve"]["recall"], il["pr_curve"]["precision"])
        ax[1].set_title(f"image PR, AP={il['auprc']:.3f}"); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
        _save(fig, fig_dir / "image_level_roc_pr.png")
        outs.append(fig_dir / "image_level_roc_pr.png")
    pl = res.get("pixel_level") or {}
    if pl.get("roc_curve_subsampled"):
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
        r = pl["roc_curve_subsampled"]; p = pl["pr_curve_subsampled"]
        ax[0].plot(r["fpr"], r["tpr"]); ax[0].set_title(f"pixel ROC (pooled), AUROC={pl['auroc']:.3f}")
        ax[1].plot(p["recall"], p["precision"]); ax[1].set_title(f"pixel PR (pooled), AP={pl['auprc']:.3f}")
        for a in ax:
            a.set_xlim(0, 1); a.set_ylim(0, 1.02)
        _save(fig, fig_dir / "pixel_roc_pr.png")
        outs.append(fig_dir / "pixel_roc_pr.png")
    au = res.get("aupro") or {}
    if au.get("curve"):
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(au["curve"]["fpr"], au["curve"]["pro"])
        ax.axvline(au["fpr_limit"], color="red", ls="--", lw=0.8, label=f"FPR limit {au['fpr_limit']}")
        ax.set_xlabel("pixel FPR"); ax.set_ylabel("PRO"); ax.legend()
        ax.set_title(f"PRO-FPR, AUPRO@{au['fpr_limit']}={au['aupro']:.3f}")
        _save(fig, fig_dir / "pro_fpr.png")
        outs.append(fig_dir / "pro_fpr.png")
    try:
        from .visualize import visualize_run
        outs += visualize_run(run_dir, res)
    except Exception as e:  # noqa: BLE001
        (fig_dir / "visualization_error.txt").parent.mkdir(parents=True, exist_ok=True)
        (fig_dir / "visualization_error.txt").write_text(repr(e), encoding="utf-8")
    return outs
