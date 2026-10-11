"""RF-DETR adapter (runs inside envs/rfdetr).

Source: roboflow/rf-detr, PyPI ``rfdetr==1.11.0`` (Apache-2.0 for Nano/Small/Medium/Large
code + weights). Training uses the package's own PyTorch-Lightning loop (EMA, AMP,
gradient accumulation, early stopping on val mAP, ``last.ckpt`` full-state resume,
``checkpoint_best_total.pth`` best model) on our ``coco0`` export (train/valid/test).
The test folder is exported but ``run_test=False`` - the test split is only used by our
common evaluator after training.

Class ids: the export uses category_id = canonical label with flat categories, so
RF-DETR's label index (position of the category sorted by id) equals the canonical
label; ``predict`` returns 0-based indices into ``class_names`` for fine-tuned models.
Both facts are asserted at runtime (class_names order must equal the dataset classes).
Coordinates: ``predict`` resizes internally and returns xyxy in the ORIGINAL image frame.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from ...data.export import export_coco0
from ...formats import make_det_record, postprocess_det_record, save_det_predictions, validate_det_predictions
from ...utils.io import dump_json
from .base import BaseAdapter

log = logging.getLogger(__name__)

VARIANTS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}


class RFDETRAdapter(BaseAdapter):
    task = "bbox"

    def _cls(self):
        import rfdetr
        return getattr(rfdetr, VARIANTS[self.mc["model"]["variant"]])

    def _export(self) -> Path:
        subset = None
        if self.mc.get("smoke"):
            subset = {w: self.records(w) for w in ("train", "val", "test")}
        return export_coco0(self.ds, self.manifest, self.split, subset)

    def train(self, resume: str | None = None) -> dict:
        import torch
        from ...utils.seed import seed_everything
        seed_everything(int(self.run_cfg["seed"]))
        data_dir = self._export()
        out = self.run_dir / "framework"
        out.mkdir(parents=True, exist_ok=True)
        t = self.mc["train"]
        model = self._cls()()  # downloads the official COCO checkpoint of the variant
        kw = dict(dataset_dir=str(data_dir), output_dir=str(out), epochs=int(t["epochs"]),
                  batch_size=int(t["batch_size"]), grad_accum_steps=int(t.get("grad_accum", 1)),
                  lr=float(t["lr"]), lr_encoder=float(t["lr_encoder"]), weight_decay=float(t["weight_decay"]),
                  warmup_epochs=float(t.get("warmup_epochs", 0.0)), num_workers=int(t.get("num_workers", 2)),
                  early_stopping=bool(t.get("early_stopping", True)),
                  early_stopping_patience=int(t.get("early_stopping_patience", 10)),
                  use_ema=bool(t.get("use_ema", True)), run_test=False, tensorboard=False, seed=int(self.run_cfg["seed"]),
                  eval_max_dets=100, class_names=list(self.classes), progress_bar=None,
                  checkpoint_interval=int(t.get("checkpoint_interval", 10)),
                  eval_backend=t.get("eval_backend", "faster_coco_eval"),
                  compute_val_loss=True)  # logs val/loss per epoch (criterion on the validation split)
        if self.mc["model"].get("resolution"):
            kw["resolution"] = int(self.mc["model"]["resolution"])
        if t.get("max_iters_per_epoch"):     # smoke test
            kw["epochs"] = 1
        if resume:
            kw["resume"] = resume
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model.train(**{k: v for k, v in kw.items() if k in _allowed_train_kwargs(model)})
        dt = time.time() - t0
        summary = {"train_time_s": round(dt, 1), "framework": "rfdetr " + __import__("importlib.metadata").metadata.version("rfdetr"),
                   "peak_gpu_mem_allocated_gb": (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
                                                 if torch.cuda.is_available() else None),
                   "checkpoints": {"best": str(out / "checkpoint_best_total.pth"), "last_full_state": str(out / "last.ckpt")},
                   "train_kwargs": {k: v for k, v in kw.items() if k not in ("dataset_dir", "output_dir")},
                   "export_dir": str(data_dir)}
        summary.update(self._convert_log(out))
        dump_json(summary, self.run_dir / "train_summary.json")
        return summary

    def _convert_log(self, out: Path) -> dict:
        """Copy the Lightning CSV metrics (train/val loss, val mAP per epoch) to train_log.csv."""
        import csv
        files = sorted(out.rglob("metrics.csv"))
        if not files:
            return {"train_log": "metrics.csv not found"}
        rows = list(csv.DictReader(open(files[-1], encoding="utf-8")))
        by_epoch: dict = {}
        for r in rows:
            e = r.get("epoch")
            if e in (None, ""):
                continue
            d = by_epoch.setdefault(int(float(e)), {"epoch": int(float(e))})
            for k, v in r.items():
                if v not in ("", None) and k != "epoch":
                    d[k] = v
        hist = [by_epoch[k] for k in sorted(by_epoch)]
        for h in hist:  # common column names used by the reports
            h.setdefault("train_loss", h.get("train/loss", ""))
            h.setdefault("val_loss", h.get("val/loss", ""))
            h.setdefault("val_metric", h.get("val/ema_mAP_50_95", h.get("val/mAP_50_95", "")))
        if hist:
            keys = sorted({k for h in hist for k in h})
            with open(self.run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(hist)
        return {"epochs_logged": len(hist), "log_columns": sorted({k for h in hist for k in h})[:40]}

    def _load(self, checkpoint: str | None):
        ck = Path(checkpoint) if checkpoint else self.run_dir / "framework" / "checkpoint_best_total.pth"
        # our own checkpoint -> trusted (it contains non-tensor training metadata)
        model = self._cls().from_checkpoint(str(ck), trust_checkpoint=True)
        names = list(model.class_names)
        if names != list(self.classes):
            raise RuntimeError(f"class mapping mismatch: checkpoint {names} vs dataset {self.classes}")
        return model, ck

    def _to_record(self, image_id, det):
        """sv.Detections -> unified record. The RF-DETR head has K+1 sigmoid slots; index K is the
        no-object slot that rfdetr itself names "__background__" - such detections are not a class and
        are dropped (count reported). Any other out-of-range id raises in postprocess_det_record."""
        cid = np.asarray(det.class_id).astype(int).reshape(-1)
        keep = cid != self.num_classes
        rec = make_det_record(image_id, np.asarray(det.xyxy).reshape(-1, 4)[keep], np.asarray(det.confidence)[keep],
                              cid[keep])
        return rec, int((~keep).sum())

    def predict(self, which: str, checkpoint: str | None = None) -> Path:
        from PIL import Image
        model, ck = self._load(checkpoint)
        inf = self.mc["inference"]
        recs = self.records(which)
        out_recs = []
        n_bg = 0
        t0 = time.time()
        for r in recs:
            with Image.open(self.raw_root / r["file"]) as im:
                img = im.convert("RGB")  # grayscale replicated to 3 channels
            det = model.predict(img, threshold=float(inf["score_thresh"]), include_source_image=False)
            rec, nb = self._to_record(r["image_id"], det)
            n_bg += nb
            out_recs.append(postprocess_det_record(rec, r["width"], r["height"], self.num_classes,
                                                   float(inf["score_thresh"]), int(inf["max_dets"])))
        out = self.pred_path(which)
        save_det_predictions(out, out_recs, dataset=self.ds["name"], split=which, model=self.mc["name"],
                             class_names=self.classes, score_thr=float(inf["score_thresh"]),
                             max_dets=int(inf["max_dets"]),
                             extra={"checkpoint": str(ck), "seconds": round(time.time() - t0, 2),
                                    "coordinates": "rfdetr predict() returns original-image xyxy",
                                    "label_mapping": "class_id = index into class_names (asserted == dataset classes)",
                                    "dropped_no_object_slot_detections": n_bg})
        from ...formats import load_det_predictions
        probs = validate_det_predictions(load_det_predictions(out), {r["image_id"]: (r["width"], r["height"]) for r in recs})
        if probs:
            raise RuntimeError(f"invalid predictions: {probs[:5]}")
        return out

    def benchmark_model(self, checkpoint=None):
        from PIL import Image
        model, _ = self._load(checkpoint)

        def pre(rec):
            with Image.open(self.raw_root / rec["file"]) as im:
                return im.convert("RGB")

        def fwd(img):
            return model.predict(img, threshold=float(self.mc["inference"]["score_thresh"]), include_source_image=False)

        def post(det, rec):
            return self._to_record(rec["image_id"], det)[0]

        return fwd, pre, post, "model() = rfdetr predict() (includes its resize/normalize and top-k postprocess)"


def _allowed_train_kwargs(model) -> set:
    try:
        from rfdetr.config import TrainConfig
        return set(TrainConfig.model_fields) | {"resolution"}
    except Exception:
        return {"dataset_dir", "output_dir", "epochs", "batch_size", "grad_accum_steps", "lr", "resume"}
