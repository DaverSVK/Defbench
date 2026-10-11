"""Detectron2 (ViTDet) and detrex (DINO) adapter - runs inside the ``defectbench/detectron2:cu128`` container.

ViTDet: facebookresearch/detectron2 ``projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py`` (smallest
released ViTDet backbone = ViT-B, MAE pre-trained), initialised from the official COCO checkpoint
``model_final_61ccd1.pkl``. Our datasets have no instance masks, so the mask branch is disabled
(``roi_heads.mask_in_features=None`` -> StandardROIHeads.mask_on=False) and the model is registered as
``vitdet_b_mask_rcnn_boxonly`` ("ViTDet-B, box-only Mask R-CNN = Faster R-CNN heads").
Training loop = detectron2 ``tools/lazyconfig_train_net.py:do_train`` (AMPTrainer + LR scheduler hooks).

DINO: IDEA-Research/detrex ``projects/dino/configs/dino-swin/dino_swin_tiny_224_4scale_12ep.py`` initialised
from the official ``dino_swin_tiny_224_4scale_12ep.pth`` (COCO). Training loop = detrex ``tools/train_net.py``
``Trainer`` (AMP + gradient clipping), loaded from the detrex checkout.

Added hooks (both): validation loss (train-mode forward with no_grad on the validation split), COCO
evaluation on the validation split every epoch, best checkpoint by val bbox AP, early stopping.
Class ids: datasets are registered with ``thing_classes`` = canonical class order and 0-based category ids,
so detectron2's contiguous id == canonical label (asserted). Predictions are produced by the model's own
test-time resize and ``detector_postprocess`` to the ORIGINAL image size.
"""
from __future__ import annotations

import importlib.util
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from ...data.export import export_coco0
from ...formats import make_det_record, postprocess_det_record, save_det_predictions, validate_det_predictions
from ...utils.io import dump_json, load_json
from ...weights import fetch
from .base import BaseAdapter

log = logging.getLogger(__name__)


class _EarlyStop(Exception):
    pass


def _test_mapper_fields(cfg):
    """detectron2 DatasetMapper uses augmentations/image_format, detrex DetrDatasetMapper augmentation/img_format."""
    m = cfg.dataloader.test.mapper
    augs = m.get("augmentations") if "augmentations" in m else m.get("augmentation")
    fmt = m.get("image_format") if "image_format" in m else m.get("img_format")
    return augs, fmt


class Detectron2Adapter(BaseAdapter):
    task = "bbox"

    # ------------------------------------------------------------------ datasets
    def _register(self):
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.data.datasets import register_coco_instances
        subset = {w: self.records(w) for w in ("train", "val", "test")} if self.mc.get("smoke") else None
        root = export_coco0(self.ds, self.manifest, self.split, subset)
        names = {}
        for which, d in (("train", "train"), ("val", "valid"), ("test", "test")):
            n = f"defectbench_{self.ds['name']}_{self.split['split_hash']}_{which}" + ("_smoke" if subset else "")
            if n not in DatasetCatalog.list():
                register_coco_instances(n, {}, str(root / d / "_annotations.coco.json"), str(root / d))
                MetadataCatalog.get(n).thing_classes = list(self.classes)
            names[which] = n
        # sanity: contiguous id == canonical label
        DatasetCatalog.get(names["train"])
        md = MetadataCatalog.get(names["train"])
        m = getattr(md, "thing_dataset_id_to_contiguous_id", None)
        if m is not None and any(int(k) != int(v) for k, v in m.items()):
            raise RuntimeError(f"unexpected category id mapping {m}")
        return names, root

    # ------------------------------------------------------------------ config
    def _cfg(self, names):
        from detectron2.config import LazyConfig
        from detectron2.config import LazyCall as L
        from detectron2.evaluation import COCOEvaluator
        m, t = self.mc["model"], self.mc["train"]
        root = Path(os.environ.get(m["root_env"], m.get("root_default", "")))
        if str(root) not in sys.path:  # configs import `projects.<name>...` relative to the repo root
            sys.path.insert(0, str(root))
        cfg = LazyConfig.load(str(root / m["config"]))
        fw = self.run_dir / "framework"
        n_train = len(self.records("train"))
        bs = int(t["batch_size"])
        iters_per_epoch = max(1, math.ceil(n_train / bs))
        epochs = int(t["epochs"]) if not t.get("max_iters_per_epoch") else 1
        max_iter = iters_per_epoch * epochs if not t.get("max_iters_per_epoch") else int(t["max_iters_per_epoch"])
        cfg.train.output_dir = str(fw)
        cfg.train.max_iter = max_iter
        cfg.train.eval_period = iters_per_epoch if not t.get("max_iters_per_epoch") else max_iter
        cfg.train.checkpointer = dict(period=iters_per_epoch, max_to_keep=2)
        cfg.train.log_period = min(20, iters_per_epoch)
        cfg.train.seed = int(self.run_cfg["seed"])
        cfg.train.amp.enabled = bool(t.get("amp", True))
        cfg.train.device = "cuda" if self.run_cfg.get("device", "cuda") != "cpu" else "cpu"
        cfg.train.init_checkpoint = str(fetch(self.mc["pretrained"]["url"]))
        cfg.dataloader.train.dataset.names = names["train"]
        cfg.dataloader.train.total_batch_size = bs
        cfg.dataloader.train.num_workers = int(t.get("num_workers", 2))
        cfg.dataloader.test.dataset.names = names["val"]
        cfg.dataloader.test.num_workers = int(t.get("num_workers", 2))
        cfg.dataloader.evaluator = L(COCOEvaluator)(dataset_name=names["val"], output_dir=str(fw / "val_eval"),
                                                    max_dets_per_image=100)
        if m["kind"] == "vitdet":
            # box-only: disable the mask branch and the mask loading of the LSJ mapper
            cfg.model.roi_heads.mask_in_features = None
            cfg.model.roi_heads.mask_pooler = None
            cfg.model.roi_heads.mask_head = None
            cfg.model.roi_heads.num_classes = self.num_classes
            cfg.dataloader.train.mapper.use_instance_mask = False
            cfg.dataloader.train.mapper.recompute_boxes = False
            cfg.model.roi_heads.box_predictor.test_score_thresh = float(self.mc["inference"]["score_thresh"])
            cfg.model.roi_heads.box_predictor.test_topk_per_image = int(self.mc["inference"]["max_dets"])
            if t.get("act_checkpoint"):
                cfg.model.backbone.net.use_act_checkpoint = True
        else:  # detrex DINO
            cfg.model.num_classes = self.num_classes
            cfg.model.criterion.num_classes = self.num_classes
            cfg.model.device = cfg.train.device
        # schedule: keep the recipe's relative milestones, re-expressed for max_iter
        from fvcore.common.param_scheduler import MultiStepParamScheduler
        from detectron2.solver import WarmupParamScheduler
        # fvcore needs strictly increasing milestones inside (0, max_iter); very short runs (smoke) collapse them
        ms = sorted({int(max_iter * f) for f in t.get("milestones", [0.8, 0.95])} & set(range(1, max_iter)))
        cfg.lr_multiplier = L(WarmupParamScheduler)(
            scheduler=L(MultiStepParamScheduler)(values=[1.0, 0.1, 0.01][: len(ms) + 1], milestones=ms,
                                                 num_updates=max_iter),
            warmup_length=min(0.2, float(t.get("warmup_iters", 250)) / max_iter), warmup_factor=0.001)
        cfg.optimizer.lr = float(t["lr"])
        cfg.optimizer.weight_decay = float(t.get("weight_decay", cfg.optimizer.get("weight_decay", 0.1)))
        LazyConfig.save(cfg, str(fw / "config.yaml")) if fw.exists() else None
        return cfg, iters_per_epoch

    # ------------------------------------------------------------------ train
    def train(self, resume: str | None = None) -> dict:
        import torch
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import instantiate
        from detectron2.engine import AMPTrainer, SimpleTrainer, default_writers, hooks
        from detectron2.evaluation import inference_on_dataset
        from detectron2.utils import comm
        from detectron2.utils.events import get_event_storage
        (self.run_dir / "framework").mkdir(parents=True, exist_ok=True)
        names, _ = self._register()
        cfg, ipe = self._cfg(names)
        from detectron2.config import LazyConfig
        LazyConfig.save(cfg, str(self.run_dir / "framework" / "config.yaml"))
        torch.manual_seed(int(self.run_cfg["seed"]))
        model = instantiate(cfg.model).to(cfg.train.device)
        cfg.optimizer.params.model = model
        optim = instantiate(cfg.optimizer)
        train_loader = instantiate(cfg.dataloader.train)
        if self.mc["model"]["kind"] == "dino":
            # detrex tools/train_net.py Trainer: AMP + gradient clipping exactly as in the official recipe
            tr_mod = self._detrex_train_module()
            trainer = tr_mod.Trainer(model=model, dataloader=train_loader, optimizer=optim, amp=cfg.train.amp.enabled,
                                     clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None)
        else:
            trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
        class _Ck(DetectionCheckpointer):  # records incompatible keys of the initial load
            incompatible = None

            def _load_model(self, checkpoint):
                inc = super()._load_model(checkpoint)
                self.incompatible = inc
                return inc

        ck = _Ck(model, cfg.train.output_dir, trainer=trainer)
        adapter = self
        es = {"best": None, "best_iter": -1, "bad": 0, "patience": int(self.mc["train"].get("patience", 10))}
        hist = []

        def do_test():
            ret = inference_on_dataset(model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator))
            return ret

        class ValLossAndEarlyStop(hooks.HookBase):
            def after_step(self):
                it = self.trainer.iter + 1
                if it % cfg.train.eval_period != 0 and it != cfg.train.max_iter:
                    return
                st = get_event_storage()
                ap = st.latest().get("bbox/AP", (None,))[0]
                ap = None if ap is None else float(ap) / 100.0  # detectron2 reports AP in percent
                vl = adapter._val_loss(model, cfg)
                hist.append({"epoch": it // ipe - 1, "iter": it, "val_metric": ap, "val_loss": vl,
                             "train_loss": float(st.history("total_loss").median(ipe)) if "total_loss" in st.histories() else None,
                             "lr": float(st.history("lr").latest()) if "lr" in st.histories() else None})
                adapter._write_log(hist)
                if ap is not None and (es["best"] is None or ap > es["best"]):
                    es.update(best=ap, best_iter=it, bad=0)
                    ck.save("model_best")
                else:
                    es["bad"] += 1
                if es["patience"] > 0 and es["bad"] >= es["patience"]:
                    raise _EarlyStop(f"no val AP improvement for {es['bad']} evaluations")

        trainer.register_hooks([
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(ck, **cfg.train.checkpointer) if comm.is_main_process() else None,
            hooks.EvalHook(cfg.train.eval_period, do_test),
            ValLossAndEarlyStop(),
            hooks.PeriodicWriter(default_writers(cfg.train.output_dir, cfg.train.max_iter), period=cfg.train.log_period),
        ])
        # official detrex checkpoints pickle omegaconf containers; allow-list them (keeps torch.load weights_only=True)
        import typing
        from collections import defaultdict
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.nodes import AnyNode
        torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode,
                                              typing.Any, list, dict, int, defaultdict])
        ck.resume_or_load(resume or cfg.train.init_checkpoint, resume=bool(resume))
        if not resume and ck.incompatible is not None:
            inc = ck.incompatible
            miss = [k for k in inc.missing_keys if k.startswith("backbone.")]
            shape = [k[0] for k in (inc.incorrect_shapes or []) if k[0].startswith("backbone.")]
            dump_json({"init_checkpoint": cfg.train.init_checkpoint, "missing": list(inc.missing_keys),
                       "unexpected": list(inc.unexpected_keys), "incorrect_shapes": [k[0] for k in inc.incorrect_shapes or []]},
                      self.run_dir / "framework" / "init_load_report.json")
            if miss or shape:
                raise RuntimeError(f"pretrained backbone not fully loaded ({(miss + shape)[:3]}...) - refusing to train "
                                   "a silently re-initialised backbone")
        start = trainer.iter + 1 if (resume and ck.has_checkpoint()) else 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        stopped = False
        try:
            trainer.train(start, cfg.train.max_iter)
        except _EarlyStop as e:
            stopped = True
            log.info("early stopping: %s", e)
        summary = {"train_time_s": round(time.time() - t0, 1), "stopped_early": stopped, "early_stopping": es,
                   "iters_per_epoch": ipe, "max_iter": cfg.train.max_iter,
                   "peak_gpu_mem_allocated_gb": (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
                                                 if torch.cuda.is_available() else None),
                   "parameters": {"total": int(sum(p.numel() for p in model.parameters()))},
                   "init_checkpoint": cfg.train.init_checkpoint,
                   "resume_state": "detectron2 checkpointer: model + optimizer + scheduler + iteration (+ grad_scaler for AMP)"}
        dump_json(summary, self.run_dir / "train_summary.json")
        return summary

    def _detrex_train_module(self):
        p = Path(os.environ.get("DETREX_ROOT", "/opt/detrex")) / "tools" / "train_net.py"
        spec = importlib.util.spec_from_file_location("detrex_train_net", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _val_loss(self, model, cfg):
        """Mean training loss on the validation split (train mode, no_grad, test-time resize)."""
        import torch
        from detectron2.config import instantiate
        from detectron2.data import DatasetMapper, build_detection_test_loader, get_detection_dataset_dicts
        try:
            augs_cfg, fmt = _test_mapper_fields(cfg)
            augs = [instantiate(a) for a in augs_cfg]
            mapper = DatasetMapper(is_train=True, augmentations=augs, image_format=fmt,
                                   use_instance_mask=False, recompute_boxes=False)
            dicts = get_detection_dataset_dicts(cfg.dataloader.test.dataset.names, filter_empty=True)
            loader = build_detection_test_loader(dicts, mapper=mapper, num_workers=0)
            was = model.training
            model.train()
            tot, n = 0.0, 0
            with torch.no_grad():
                for batch in loader:
                    ld = model(batch)
                    tot += float(sum(ld.values()))
                    n += 1
            model.train(was)
            return tot / max(1, n)
        except Exception as e:  # noqa: BLE001
            log.warning("val loss not computed: %s", e)
            return None

    def _write_log(self, hist):
        import csv
        keys = ["epoch", "iter", "train_loss", "val_loss", "val_metric", "lr"]
        with open(self.run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(hist)

    # ------------------------------------------------------------------ predict
    def _load(self, checkpoint):
        import torch
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import instantiate
        names, _ = self._register()
        cfg, _ = self._cfg(names)
        model = instantiate(cfg.model).to(cfg.train.device).eval()
        fw = self.run_dir / "framework"
        ck = Path(checkpoint) if checkpoint else (fw / "model_best.pth" if (fw / "model_best.pth").exists()
                                                   else fw / "model_final.pth")
        DetectionCheckpointer(model).load(str(ck))
        augs_cfg, fmt = _test_mapper_fields(cfg)
        augs = [instantiate(a) for a in augs_cfg]
        return model, augs, fmt, ck

    def _infer(self, model, augs, fmt, path, h, w):
        import torch
        from detectron2.data import transforms as T
        from detectron2.data.detection_utils import read_image
        img = read_image(str(path), format=fmt)
        inp = T.AugInput(img)
        T.AugmentationList(augs)(inp)
        x = torch.as_tensor(np.ascontiguousarray(inp.image.transpose(2, 0, 1)))
        with torch.no_grad():
            out = model([{"image": x, "height": h, "width": w}])[0]["instances"].to("cpu")
        return out

    def predict(self, which: str, checkpoint: str | None = None) -> Path:
        model, augs, fmt, ck = self._load(checkpoint)
        inf = self.mc["inference"]
        recs = self.records(which)
        outs = []
        t0 = time.time()
        for r in recs:
            ins = self._infer(model, augs, fmt, self.raw_root / r["file"], r["height"], r["width"])
            rec = make_det_record(r["image_id"], ins.pred_boxes.tensor.numpy(), ins.scores.numpy(),
                                  ins.pred_classes.numpy().astype(int))
            outs.append(postprocess_det_record(rec, r["width"], r["height"], self.num_classes,
                                               float(inf["score_thresh"]), int(inf["max_dets"])))
        out = self.pred_path(which)
        save_det_predictions(out, outs, dataset=self.ds["name"], split=which, model=self.mc["name"],
                             class_names=self.classes, score_thr=float(inf["score_thresh"]), max_dets=int(inf["max_dets"]),
                             extra={"checkpoint": str(ck), "seconds": round(time.time() - t0, 2),
                                    "coordinates": "detector_postprocess to (height,width) of the original image"})
        from ...formats import load_det_predictions
        probs = validate_det_predictions(load_det_predictions(out), {r["image_id"]: (r["width"], r["height"]) for r in recs})
        if probs:
            raise RuntimeError(f"invalid predictions: {probs[:5]}")
        return out

    def benchmark_model(self, checkpoint=None):
        import torch
        from detectron2.data import transforms as T
        from detectron2.data.detection_utils import read_image
        model, augs, fmt, _ = self._load(checkpoint)

        def pre(rec):
            img = read_image(str(self.raw_root / rec["file"]), format=fmt)
            inp = T.AugInput(img)
            T.AugmentationList(augs)(inp)
            return [{"image": torch.as_tensor(np.ascontiguousarray(inp.image.transpose(2, 0, 1))),
                     "height": rec["height"], "width": rec["width"]}]

        def fwd(x):
            with torch.no_grad():
                return model(x)

        def post(out, rec):
            ins = out[0]["instances"].to("cpu")
            return make_det_record(rec["image_id"], ins.pred_boxes.tensor.numpy(), ins.scores.numpy(),
                                   ins.pred_classes.numpy())

        return fwd, pre, post, "model() = detectron2 model forward incl. normalisation and detector_postprocess"
