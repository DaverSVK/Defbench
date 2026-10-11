"""Ultralytics adapter for YOLO26, YOLOv10 and YOLO12 (runs inside envs/ultralytics).

Source: ultralytics==8.4.165 (AGPL-3.0). Weights: official Ultralytics release assets
``https://github.com/ultralytics/assets/releases/download/v8.4.0/<name>.pt`` (verified to
exist: yolo26n/s, yolov10n/s, yolo12n/s). Training uses the Ultralytics trainer with its
family-default recipe (SGD/AdamW 'auto', mosaic with close_mosaic, warmup, EMA, AMP,
early stopping ``patience`` on the val fitness), our YOLO export, and the fixed split.

Coordinates: ``Results.boxes.xyxy`` is already mapped back to the original image
(letterbox padding removed, resize reverted by Ultralytics' ``scale_boxes``); the
unified-format validator re-checks it. Labels: YOLO class index = canonical label.
YOLO26 / YOLOv10 predict NMS-free (end-to-end head) by default; YOLO12 uses NMS
(iou 0.7, Ultralytics default). Grayscale images are read by OpenCV as 3-channel BGR
(gray replicated).
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import numpy as np

from ...data.export import export_yolo
from ...formats import make_det_record, postprocess_det_record, save_det_predictions, validate_det_predictions
from ...utils.io import dump_json
from ...weights import fetch
from .base import BaseAdapter

log = logging.getLogger(__name__)
ASSETS = "https://github.com/ultralytics/assets/releases/download/v8.4.0/{}.pt"


class UltralyticsAdapter(BaseAdapter):
    task = "bbox"

    def _data_yaml(self) -> Path:
        subset = {w: self.records(w) for w in ("train", "val", "test")} if self.mc.get("smoke") else None
        return export_yolo(self.ds, self.manifest, self.split, subset) / "data.yaml"

    def _weights(self) -> Path:
        name = self.mc["model"]["weights"]
        return fetch(ASSETS.format(name), filename=f"{name}.pt")

    def train(self, resume: str | None = None) -> dict:
        import torch
        from ultralytics import YOLO
        t = self.mc["train"]
        if resume:
            model = YOLO(resume)
            kw = {"resume": True}
        else:
            model = YOLO(str(self._weights()))
            kw = dict(data=str(self._data_yaml()), epochs=int(t["epochs"]), imgsz=int(self.mc["model"]["imgsz"]),
                      batch=int(t["batch_size"]), device=0 if self.run_cfg.get("device", "cuda") != "cpu" else "cpu",
                      seed=int(self.run_cfg["seed"]), deterministic=bool(t.get("deterministic", False)),
                      project=str(self.run_dir), name="framework", exist_ok=True, patience=int(t["patience"]),
                      workers=int(t.get("num_workers", 2)), amp=bool(t.get("amp", True)), plots=True, val=True,
                      pretrained=True, verbose=False)
            for k in ("optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "cos_lr",
                      "close_mosaic", "mosaic", "mixup", "scale", "fliplr", "flipud", "degrees", "translate",
                      "hsv_h", "hsv_s", "hsv_v", "nbs", "freeze"):
                if k in t:
                    kw[k] = t[k]
            if t.get("max_iters_per_epoch"):   # smoke test: one tiny epoch
                kw.update(epochs=1, fraction=1.0, plots=False)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model.train(**kw)
        dt = time.time() - t0
        fw = self.run_dir / "framework"
        summary = {"train_time_s": round(dt, 1),
                   "framework": "ultralytics " + __import__("ultralytics").__version__,
                   "peak_gpu_mem_allocated_gb": (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
                                                 if torch.cuda.is_available() else None),
                   "checkpoints": {"best": str(fw / "weights" / "best.pt"), "last_full_state": str(fw / "weights" / "last.pt")},
                   "train_kwargs": {k: v for k, v in kw.items() if k not in ("data", "project")},
                   "early_stopping": f"ultralytics patience={t['patience']} on val fitness (0.1*mAP50 + 0.9*mAP50-95)",
                   "parameters": self._count(fw / "weights" / "best.pt")}
        self._convert_log(fw / "results.csv")
        dump_json(summary, self.run_dir / "train_summary.json")
        return summary

    def _count(self, ck: Path):
        try:
            from ultralytics import YOLO
            m = YOLO(str(ck)).model
            return {"total": sum(p.numel() for p in m.parameters())}
        except Exception as e:  # noqa: BLE001
            return {"error": repr(e)}

    def _convert_log(self, results_csv: Path) -> None:
        """Ultralytics results.csv (train/val box, cls, dfl losses, mAP per epoch) -> train_log.csv."""
        if not results_csv.exists():
            return
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(open(results_csv, encoding="utf-8"))]
        for r in rows:
            tl = [float(r[k]) for k in r if k.startswith("train/") and k.endswith("loss") and r[k]]
            vl = [float(r[k]) for k in r if k.startswith("val/") and k.endswith("loss") and r[k]]
            r["train_loss"] = sum(tl) if tl else ""
            r["val_loss"] = sum(vl) if vl else ""
            r["val_metric"] = r.get("metrics/mAP50-95(B)", "")
        if rows:
            with open(self.run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    def _load(self, checkpoint):
        from ultralytics import YOLO
        ck = Path(checkpoint) if checkpoint else self.run_dir / "framework" / "weights" / "best.pt"
        m = YOLO(str(ck))
        names = [m.names[i] for i in range(len(m.names))]
        if names != list(self.classes):
            raise RuntimeError(f"class mapping mismatch: {names} vs {self.classes}")
        return m, ck

    def _predict_one(self, model, path: Path, dev):
        inf = self.mc["inference"]
        return model.predict(str(path), imgsz=int(self.mc["model"]["imgsz"]), conf=float(inf["score_thresh"]),
                             iou=float(inf.get("nms_iou", 0.7)), max_det=int(inf["max_dets"]), device=dev,
                             half=False, verbose=False)[0]

    def predict(self, which: str, checkpoint: str | None = None) -> Path:
        model, ck = self._load(checkpoint)
        dev = 0 if self.run_cfg.get("device", "cuda") != "cpu" else "cpu"
        inf = self.mc["inference"]
        recs = self.records(which)
        outs = []
        t0 = time.time()
        for r in recs:
            res = self._predict_one(model, self.raw_root / r["file"], dev)
            b = res.boxes
            rec = make_det_record(r["image_id"], b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int))
            outs.append(postprocess_det_record(rec, r["width"], r["height"], self.num_classes,
                                               float(inf["score_thresh"]), int(inf["max_dets"])))
        out = self.pred_path(which)
        save_det_predictions(out, outs, dataset=self.ds["name"], split=which, model=self.mc["name"],
                             class_names=self.classes, score_thr=float(inf["score_thresh"]), max_dets=int(inf["max_dets"]),
                             extra={"checkpoint": str(ck), "seconds": round(time.time() - t0, 2),
                                    "end2end_nms_free": bool(getattr(model.model, "end2end", False)),
                                    "coordinates": "ultralytics Results.boxes.xyxy (letterbox reverted)"})
        from ...formats import load_det_predictions
        probs = validate_det_predictions(load_det_predictions(out), {r["image_id"]: (r["width"], r["height"]) for r in recs})
        if probs:
            raise RuntimeError(f"invalid predictions: {probs[:5]}")
        return out

    def benchmark_model(self, checkpoint=None):
        model, _ = self._load(checkpoint)
        dev = 0 if self.run_cfg.get("device", "cuda") != "cpu" else "cpu"

        def pre(rec):
            return self.raw_root / rec["file"]

        def fwd(path):
            return self._predict_one(model, path, dev)

        def post(res, rec):
            b = res.boxes
            return make_det_record(rec["image_id"], b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy())

        return fwd, pre, post, ("model() = ultralytics predict (read+letterbox+forward+NMS/end2end); Ultralytics' own "
                                "per-stage timings (Results.speed, CUDA-synchronised) are recorded separately")
