"""Faster R-CNN ResNet-50-FPN (torchvision) and its CBAM variant - in-process adapter.

Source: torchvision.models.detection.fasterrcnn_resnet50_fpn (torchvision 0.26.0, BSD-3),
COCO_V1 weights (box AP 37.0 on COCO val2017 per torchvision docs).
Resizing/normalisation: torchvision's GeneralizedRCNNTransform (min_size/max_size,
ImageNet mean/std) - predictions are returned in the coordinates of the image passed
to the model, which is the ORIGINAL image (we never pre-resize), so no manual
inverse transform is needed; ``validate_det_predictions`` double-checks it.
Labels: torchvision uses 0 = background, so canonical = native - 1 (class_mapping).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from ...data.torch_datasets import DetectionDataset, det_collate
from ...evaluation.detection import coco_evaluate
from ...formats import make_det_record, postprocess_det_record, save_det_predictions, validate_det_predictions
from ...training.engine import Trainer, TrainerHooks
from ...utils.io import dump_json
from ...utils.seed import seed_everything, worker_init_fn
from ..class_mapping import to_canonical, to_native
from .base import BaseAdapter

log = logging.getLogger(__name__)


def build_frcnn(mc: dict, num_classes: int) -> torch.nn.Module:
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_Weights,
                                              fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    m = mc["model"]
    pre = mc["pretrained"]["init"]  # coco | imagenet | none
    variant = m.get("variant", "v1")
    fn = fasterrcnn_resnet50_fpn if variant == "v1" else fasterrcnn_resnet50_fpn_v2
    wcls = FasterRCNN_ResNet50_FPN_Weights if variant == "v1" else FasterRCNN_ResNet50_FPN_V2_Weights
    kw = dict(min_size=m.get("min_size", 800), max_size=m.get("max_size", 1333),
              box_score_thresh=float(mc["inference"]["score_thresh"]),
              box_detections_per_img=int(mc["inference"]["max_dets"]),
              box_nms_thresh=float(mc["inference"].get("nms_iou", 0.5)),
              trainable_backbone_layers=m.get("trainable_backbone_layers", 3))
    if pre == "coco":
        model = fn(weights=wcls.COCO_V1, **kw)
    elif pre == "imagenet":
        model = fn(weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V1, **kw)
    elif pre == "none":
        if not mc["pretrained"].get("from_scratch_explicit"):
            raise ValueError("training from scratch must be requested explicitly: pretrained.from_scratch_explicit=true")
        model = fn(weights=None, weights_backbone=None, **{**kw, "trainable_backbone_layers": 5})
    else:
        raise ValueError(pre)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    cb = m.get("cbam")
    if cb:
        from ..cbam import BackboneWithFPNAndCBAM
        model.backbone = BackboneWithFPNAndCBAM(model.backbone, levels=cb.get("levels", ["P2", "P3", "P4", "P5"]),
                                                reduction_ratio=int(cb.get("reduction_ratio", 16)),
                                                recompute_p6_from_refined_p5=bool(cb.get("recompute_p6", True)))
    return model


def _bn_eval(model):
    for mod in model.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.eval()


class TorchvisionFasterRCNN(BaseAdapter):
    task = "bbox"

    def __init__(self, run_cfg: dict):
        super().__init__(run_cfg)
        self.device = torch.device(run_cfg.get("device", "cuda") if torch.cuda.is_available() or
                                   run_cfg.get("device") == "cpu" else "cpu")
        self.label_offset = to_native("torchvision", 0)

    def build(self):
        seed_everything(int(self.run_cfg["seed"]), bool(self.mc["train"].get("deterministic", False)))
        model = build_frcnn(self.mc, self.num_classes)
        return model.to(self.device)

    def count_parameters(self, model=None) -> dict:
        model = model or build_frcnn(self.mc, self.num_classes)
        d = {"total": sum(p.numel() for p in model.parameters()),
             "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        if hasattr(model.backbone, "describe"):
            d["cbam"] = model.backbone.describe()
        return d

    def _loader(self, which, train):
        t = self.mc["train"]
        ds = DetectionDataset(self.records(which), self.raw_root, self.label_offset,
                              aug=self.mc.get("augment") if train else None, train=train,
                              channel_policy=self.run_cfg["input"]["channel_policy"])
        g = torch.Generator()
        g.manual_seed(int(self.run_cfg["seed"]))
        return torch.utils.data.DataLoader(
            ds, batch_size=int(t["batch_size"]) if train else int(t.get("eval_batch_size", 4)), shuffle=train,
            num_workers=int(t.get("num_workers", 2)), collate_fn=det_collate, worker_init_fn=worker_init_fn,
            generator=g, persistent_workers=False, pin_memory=self.device.type == "cuda")

    @torch.no_grad()
    def _predict_records(self, model, which) -> list[dict]:
        model.eval()
        loader = self._loader(which, train=False)
        amp = bool(self.mc["inference"].get("amp", False)) and self.device.type == "cuda"
        out = []
        recs = {r["image_id"]: r for r in self.records(which)}
        for imgs, tgts in loader:
            imgs = [i.to(self.device, non_blocking=True) for i in imgs]
            with torch.autocast(device_type=self.device.type, enabled=amp):
                preds = model(imgs)
            for p, t in zip(preds, tgts):
                r = recs[t["image_id"]]
                labels = [to_canonical("torchvision", int(l)) for l in p["labels"].cpu()]
                rec = make_det_record(t["image_id"], p["boxes"].float().cpu().numpy(), p["scores"].float().cpu().numpy(),
                                      labels)
                out.append(postprocess_det_record(rec, r["width"], r["height"], self.num_classes,
                                                  float(self.mc["inference"]["score_thresh"]),
                                                  int(self.mc["inference"]["max_dets"])))
        return out

    def train(self, resume: str | None = None) -> dict:
        model = self.build()
        params = self.count_parameters(model)
        dump_json(params, self.run_dir / "params.json")
        train_loader = self._loader("train", True)
        val_loader = self._loader("val", False)
        val_recs = self.records("val")

        def train_step(model, batch, dev):
            imgs, tgts = batch
            imgs = [i.to(dev, non_blocking=True) for i in imgs]
            tg = [{"boxes": t["boxes"].to(dev), "labels": t["labels"].to(dev)} for t in tgts]
            ld = model(imgs, tg)
            loss = sum(ld.values())
            return loss, {k: v.detach() for k, v in ld.items()}

        @torch.no_grad()
        def val_loss(model, loader, dev, amp_ctx):
            model.train()
            _bn_eval(model)  # no BN statistics update (CBAM spatial gate has BatchNorm)
            tot, n = 0.0, 0
            for imgs, tgts in loader:
                imgs = [i.to(dev) for i in imgs]
                tg = [{"boxes": t["boxes"].to(dev), "labels": t["labels"].to(dev)} for t in tgts]
                with amp_ctx():
                    ld = model(imgs, tg)
                tot += float(sum(ld.values()))
                n += 1
            model.eval()
            return tot / max(1, n)

        def evaluate(model, dev):
            preds = self._predict_records(model, "val")
            payload = {"predictions": preds, "max_detections_per_image": int(self.mc["inference"]["max_dets"])}
            res = coco_evaluate(val_recs, payload, self.classes)
            c = res["coco"]
            return {"val_metric": c["AP"], "AP50": c["AP50"], "AR100": c["AR100"]}

        t = dict(self.mc["train"])
        t["_input_desc"] = f"min_size={self.mc['model'].get('min_size')}, max_size={self.mc['model'].get('max_size')}"
        if t.get("freeze_backbone_epochs", 0):
            # only freeze parameters that torchvision left trainable (trainable_backbone_layers)
            trainable = {id(p) for p in model.backbone.parameters() if p.requires_grad}
            bp = lambda m: [p for p in (m.backbone.inner.body if hasattr(m.backbone, "inner") else m.backbone.body).parameters()
                            if id(p) in trainable]
        else:
            bp = None
        trainer = Trainer(model, TrainerHooks(train_step, val_loss, evaluate, bp), train_loader, val_loader, t,
                          self.run_dir, self.device)
        if resume:
            trainer.resume(resume)
        summary = trainer.fit()
        summary["parameters"] = params
        dump_json(summary, self.run_dir / "train_summary.json")
        return summary

    def load(self, checkpoint: str | None = None):
        ck = Path(checkpoint) if checkpoint else self.run_dir / "checkpoints" / "best.pt"
        # rebuild exactly as in training (same norm layers: FrozenBN for pretrained init), then load weights
        model = build_frcnn(self.mc, self.num_classes)
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"] if "model" in st else st)
        return model.to(self.device).eval(), ck

    def predict(self, which: str, checkpoint: str | None = None) -> Path:
        model, ck = self.load(checkpoint)
        t0 = time.time()
        preds = self._predict_records(model, which)
        out = self.pred_path(which)
        recs = self.records(which)
        save_det_predictions(out, preds, dataset=self.ds["name"], split=which, model=self.mc["name"],
                             class_names=self.classes, score_thr=float(self.mc["inference"]["score_thresh"]),
                             max_dets=int(self.mc["inference"]["max_dets"]),
                             extra={"checkpoint": str(ck), "seconds": round(time.time() - t0, 2),
                                    "label_mapping": "torchvision native-1", "coordinates": "model output in "
                                    "original image frame (GeneralizedRCNNTransform.postprocess)"})
        from ...formats import load_det_predictions
        probs = validate_det_predictions(load_det_predictions(out), {r["image_id"]: (r["width"], r["height"]) for r in recs})
        if probs:
            raise RuntimeError(f"invalid predictions: {probs[:5]}")
        return out

    def benchmark_model(self, checkpoint: str | None = None):
        model, _ = self.load(checkpoint)
        from ...data.torch_datasets import load_image
        cp = self.run_cfg["input"]["channel_policy"]

        def preprocess(rec):
            import numpy as np
            img = load_image(self.raw_root / rec["file"], cp)
            return [torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255).to(self.device)]

        def model_fn(x):
            return model(x)

        def postprocess(out, rec):
            p = out[0]
            return make_det_record(rec["image_id"], p["boxes"].cpu().numpy(), p["scores"].cpu().numpy(),
                                   [to_canonical("torchvision", int(l)) for l in p["labels"].cpu()])

        return model_fn, preprocess, postprocess, "model() includes torchvision's internal resize+normalize " \
                                                   "(GeneralizedRCNNTransform) and NMS"
