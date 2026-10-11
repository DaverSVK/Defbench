"""MMDetection 3.3.0 adapter: Cascade Mask R-CNN + Swin-T and its box-only variant (container ``mmdet``).

Configuration = official MMDetection components, composed exactly as the official Swin detection repo
(SwinTransformer/Swin-Transformer-Object-Detection, config cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_
480-800_giou_4conv1f_adamw_3x_coco.py) combines them, but on the maintained MMDetection 3.x code base:

* base: ``configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py`` (mask) or
  ``configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py`` (box-only),
* backbone replaced by ``SwinTransformer`` (Swin-T: embed 96, depths 2-2-6-2, heads 3-6-12-24, window 7,
  drop_path 0.2) - identical to mmdet's own ``configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py``,
  FPN in_channels [96, 192, 384, 768]; AdamW (lr 1e-4, wd 0.05, no decay for norm/position tables);
* box heads = the official Swin-OD heads: 3x ConvFCBBoxHead (4 shared convs + 1 fc, BN, GIoU loss x10,
  reg_decoded_bbox, stage stds 0.1/0.05/0.033) so that the official COCO checkpoint loads into them.
* initialisation (``pretrained.init``):
    coco      the OFFICIAL Swin-OD Cascade Mask R-CNN Swin-T COCO checkpoint (mmdet 2.x format); the Swin
              backbone keys are converted with mmdet's own ``swin_converter``; class-specific layers are
              re-initialised (size mismatch, reported);
    imagenet  Swin-T ImageNet-1k backbone only.

Mask branch: only trained where real pixel masks exist (Magnetic Tile). The semantic masks are split into
instances by connected components (``export_coco0_instances``) - flagged as DERIVED instances. On NEU / GC10
the registry uses the box-only model ``cascade_rcnn_swin_t_boxonly`` (no mask head, no mask loss).
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from ...data.export import export_coco0, export_coco0_instances
from ...formats import make_det_record, postprocess_det_record, save_det_predictions, validate_det_predictions
from ...utils.io import dump_json
from ...weights import WEIGHTS_DIR, fetch
from .base import BaseAdapter

log = logging.getLogger(__name__)


def _allow_mmengine_checkpoint_globals():
    """MMEngine checkpoints pickle the logger state (HistoryBuffer + numpy arrays/scalars). Allow-list exactly
    these types so torch.load keeps weights_only=True (default since torch 2.6)."""
    import numpy as np
    import torch
    from mmengine.logging.history_buffer import HistoryBuffer
    torch.serialization.add_safe_globals([
        HistoryBuffer, np.ndarray, np.dtype, np.core.multiarray._reconstruct, np.core.multiarray.scalar,
        *{type(np.dtype(t)) for t in ("float64", "float32", "int64", "int32", "bool")}])


SWIN_T = dict(type="SwinTransformer", embed_dims=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
              mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2,
              patch_norm=True, out_indices=(0, 1, 2, 3), with_cp=False, convert_weights=True)


class MMDetAdapter(BaseAdapter):
    task = "bbox"

    def __init__(self, run_cfg):
        super().__init__(run_cfg)
        self.task = run_cfg["task"]
        self.with_mask = bool(self.mc["model"].get("with_mask", False))

    def _data(self):
        subset = {w: self.records(w) for w in ("train", "val", "test")} if self.mc.get("smoke") else None
        if self.with_mask:
            return export_coco0_instances(self.ds, self.manifest, self.split, subset)
        return export_coco0(self.ds, self.manifest, self.split, subset)

    def _converted_coco_ckpt(self) -> Path:
        """Official Swin-OD (mmdet 2.x) checkpoint -> mmdet 3.x key names (backbone via swin_converter)."""
        import torch
        from mmdet.models.backbones.swin import swin_converter
        src = fetch(self.mc["pretrained"]["url"])
        dst = WEIGHTS_DIR / (src.stem + "_mmdet3.pth")
        if not dst.exists():
            st = torch.load(src, map_location="cpu", weights_only=False)
            sd = st.get("state_dict", st)
            bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
            conv = swin_converter(bb)  # NOTE: swin_converter itself prefixes every key with "backbone."
            assert all(k.startswith("backbone.") for k in conv)
            new = {**conv, **{k: v for k, v in sd.items() if not k.startswith("backbone.")}}
            torch.save({"state_dict": new, "meta": {"converted_from": str(src), "converter": "mmdet swin_converter"}}, dst)
        return dst

    def _cfg(self, data_root: Path, resume: str | None):
        from mmengine.config import Config
        m, t = self.mc["model"], self.mc["train"]
        root = Path(os.environ.get("MMDET_ROOT", "/opt/mmdetection"))
        cfg = Config.fromfile(str(root / m["base_config"]))
        K = self.num_classes
        fw = self.run_dir / "framework"
        cfg.work_dir = str(fw)
        bb = dict(SWIN_T)
        if self.mc["pretrained"]["init"] == "imagenet":
            bb["init_cfg"] = dict(type="Pretrained", checkpoint=str(fetch(self.mc["pretrained"]["backbone_url"])))
        cfg.model.backbone = bb
        cfg.model.neck.in_channels = [96, 192, 384, 768]
        # the three 4conv1fc GIoU box heads of the official Swin-OD config (SyncBN -> BN on one GPU)
        stds = [[0.1, 0.1, 0.2, 0.2], [0.05, 0.05, 0.1, 0.1], [0.033, 0.033, 0.067, 0.067]]
        cfg.model.roi_head.bbox_head = [
            dict(type="ConvFCBBoxHead", num_shared_convs=4, num_shared_fcs=1, in_channels=256, conv_out_channels=256,
                 fc_out_channels=1024, roi_feat_size=7, num_classes=K,
                 bbox_coder=dict(type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=s),
                 reg_class_agnostic=False, reg_decoded_bbox=True, norm_cfg=dict(type="BN", requires_grad=True),
                 loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                 loss_bbox=dict(type="GIoULoss", loss_weight=10.0)) for s in stds]
        if self.with_mask:
            cfg.model.roi_head.mask_head.num_classes = K
        cfg.model.test_cfg.rcnn.score_thr = float(self.mc["inference"]["score_thresh"])
        cfg.model.test_cfg.rcnn.max_per_img = int(self.mc["inference"]["max_dets"])
        meta = dict(classes=tuple(self.classes))
        for dl, d in (("train_dataloader", "train"), ("val_dataloader", "valid"), ("test_dataloader", "valid")):
            ds = cfg[dl].dataset
            ds.data_root = ""
            ds.ann_file = str(data_root / d / "_annotations.coco.json")
            ds.data_prefix = dict(img=str(data_root / d) + "/")
            ds.metainfo = meta
            cfg[dl].num_workers = int(t.get("num_workers", 2))
        cfg.train_dataloader.batch_size = int(t["batch_size"])
        cfg.train_dataloader.dataset.filter_cfg = dict(filter_empty_gt=False, min_size=1)  # keep negatives
        cfg.val_dataloader.batch_size = 1
        cfg.val_evaluator = dict(type="CocoMetric", ann_file=str(data_root / "valid" / "_annotations.coco.json"),
                                 metric=["bbox"], format_only=False)
        cfg.test_evaluator = cfg.val_evaluator
        epochs = int(t["epochs"]) if not t.get("max_iters_per_epoch") else 1
        cfg.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=epochs, val_interval=1)
        cfg.param_scheduler = [
            dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=int(t.get("warmup_iters", 500))),
            dict(type="MultiStepLR", begin=0, end=epochs, by_epoch=True,
                 milestones=[max(1, int(epochs * f)) for f in t.get("milestones", [0.67, 0.89])], gamma=0.1)]
        cfg.optim_wrapper = dict(
            type="AmpOptimWrapper" if t.get("amp", True) else "OptimWrapper",
            accumulative_counts=int(t.get("grad_accum", 1)),
            optimizer=dict(type="AdamW", lr=float(t["lr"]), betas=(0.9, 0.999), weight_decay=float(t["weight_decay"])),
            paramwise_cfg=dict(custom_keys={"absolute_pos_embed": dict(decay_mult=0.0),
                                            "relative_position_bias_table": dict(decay_mult=0.0),
                                            "norm": dict(decay_mult=0.0)}))
        cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=1, max_keep_ckpts=2,
                                            save_best="coco/bbox_mAP", rule="greater")
        cfg.custom_hooks = [dict(type="EarlyStoppingHook", monitor="coco/bbox_mAP", rule="greater",
                                 patience=int(t.get("patience", 10)))]
        cfg.randomness = dict(seed=int(self.run_cfg["seed"]))
        if t.get("max_iters_per_epoch"):
            cfg.train_dataloader.batch_size = 2
        if resume:
            cfg.resume = True
            cfg.load_from = resume
        elif self.mc["pretrained"]["init"] == "coco":
            cfg.load_from = str(self._converted_coco_ckpt())
        return cfg

    def train(self, resume: str | None = None) -> dict:
        import torch
        from mmengine.runner import Runner
        from mmdet.utils import register_all_modules
        register_all_modules()
        _allow_mmengine_checkpoint_globals()
        (self.run_dir / "framework").mkdir(parents=True, exist_ok=True)
        data_root = self._data()
        cfg = self._cfg(data_root, resume)
        cfg.dump(str(self.run_dir / "framework" / "config.py"))
        runner = Runner.from_cfg(cfg)
        if not resume and self.mc["pretrained"]["init"] == "coco":
            runner.load_or_resume()  # loads cfg.load_from now; runner.train() will not load again
            self._verify_init(runner.model, cfg.load_from)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        runner.train()
        summary = {"train_time_s": round(time.time() - t0, 1),
                   "peak_gpu_mem_allocated_gb": (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
                                                 if torch.cuda.is_available() else None),
                   "with_mask": self.with_mask, "instances": "derived (connected components)" if self.with_mask else None,
                   "init": self.mc["pretrained"]["init"], "load_from": cfg.get("load_from"),
                   "parameters": {"total": int(sum(p.numel() for p in runner.model.parameters()))},
                   "val_loss": "not computed (MMEngine val loop has no loss); train loss/val mAP in framework/*/vis_data"}
        self._convert_logs()
        dump_json(summary, self.run_dir / "train_summary.json")
        return summary

    def _verify_init(self, model, ckpt_path):
        """Fail if backbone weights of the COCO checkpoint were not loaded (non-strict loading would hide it)."""
        import torch
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        cur = model.state_dict()
        skip = ("relative_position_index", "attn_mask", "num_batches_tracked")
        missing = [k for k in cur if k not in sd and not k.endswith(skip)]
        mismatch = [k for k in cur if k in sd and sd[k].shape != cur[k].shape]
        loaded = [k for k in cur if k in sd and sd[k].shape == cur[k].shape]
        dump_json({"checkpoint": str(ckpt_path), "n_model_tensors": len(cur), "n_loaded": len(loaded),
                   "missing": missing, "shape_mismatch": mismatch},
                  self.run_dir / "framework" / "init_load_report.json")
        bad = [k for k in missing + mismatch if k.startswith("backbone.")]
        if bad:
            raise RuntimeError(f"COCO checkpoint backbone not fully loaded ({len(bad)} tensors, e.g. {bad[:3]})")
        log.info("init load: %d/%d tensors from %s; not loaded: %s", len(loaded), len(cur), ckpt_path,
                 (missing + mismatch)[:12])

    def _convert_logs(self):
        import csv
        import json
        fw = self.run_dir / "framework"
        files = sorted(fw.rglob("vis_data/scalars.json"))
        if not files:
            return
        rows = {}
        for line in files[-1].read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if "coco/bbox_mAP" in d:
                rows.setdefault(int(d["step"]), {})["val_metric"] = d["coco/bbox_mAP"]
            elif "loss" in d and "epoch" in d:
                r = rows.setdefault(int(d["epoch"]), {})
                r.setdefault("_l", []).append(d["loss"])
        out = []
        for e in sorted(rows):
            r = rows[e]
            out.append({"epoch": e, "train_loss": (sum(r["_l"]) / len(r["_l"])) if r.get("_l") else "",
                        "val_metric": r.get("val_metric", "")})
        with open(self.run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_metric"])
            w.writeheader()
            w.writerows(out)

    def _load(self, checkpoint):
        from mmdet.apis import init_detector
        from mmdet.utils import register_all_modules
        register_all_modules()
        _allow_mmengine_checkpoint_globals()
        fw = self.run_dir / "framework"
        if checkpoint:
            ck = Path(checkpoint)
        else:
            best = sorted(fw.glob("best_coco_bbox_mAP_epoch_*.pth"))
            ck = best[-1] if best else sorted(fw.glob("epoch_*.pth"))[-1]
        dev = "cuda:0" if self.run_cfg.get("device", "cuda") != "cpu" else "cpu"
        model = init_detector(str(fw / "config.py"), str(ck), device=dev)
        if list(model.dataset_meta["classes"]) != list(self.classes):
            raise RuntimeError(f"class mapping mismatch {model.dataset_meta['classes']}")
        return model, ck

    def predict(self, which: str, checkpoint: str | None = None) -> Path:
        from mmdet.apis import inference_detector
        model, ck = self._load(checkpoint)
        inf = self.mc["inference"]
        recs = self.records(which)
        outs = []
        t0 = time.time()
        for r in recs:
            res = inference_detector(model, str(self.raw_root / r["file"]))
            p = res.pred_instances.cpu()
            rec = make_det_record(r["image_id"], p.bboxes.numpy(), p.scores.numpy(), p.labels.numpy().astype(int))
            outs.append(postprocess_det_record(rec, r["width"], r["height"], self.num_classes,
                                               float(inf["score_thresh"]), int(inf["max_dets"])))
        out = self.pred_path(which)
        save_det_predictions(out, outs, dataset=self.ds["name"], split=which, model=self.mc["name"],
                             class_names=self.classes, score_thr=float(inf["score_thresh"]), max_dets=int(inf["max_dets"]),
                             extra={"checkpoint": str(ck), "seconds": round(time.time() - t0, 2),
                                    "coordinates": "mmdet rescales predictions to the original image (rescale=True)"})
        from ...formats import load_det_predictions
        probs = validate_det_predictions(load_det_predictions(out), {r["image_id"]: (r["width"], r["height"]) for r in recs})
        if probs:
            raise RuntimeError(f"invalid predictions: {probs[:5]}")
        return out

    def benchmark_model(self, checkpoint=None):
        from mmdet.apis import inference_detector
        model, _ = self._load(checkpoint)

        def pre(rec):
            return str(self.raw_root / rec["file"])

        def fwd(path):
            return inference_detector(model, path)

        def post(res, rec):
            p = res.pred_instances.cpu()
            return make_det_record(rec["image_id"], p.bboxes.numpy(), p.scores.numpy(), p.labels.numpy())

        return fwd, pre, post, "model() = mmdet inference_detector (load + test pipeline + forward + NMS)"
