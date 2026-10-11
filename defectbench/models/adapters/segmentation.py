"""In-process segmentation adapters: U-Net, Attention U-Net, TransUNet, Swin-Unet.

All four share the data pipeline, loss (0.5 CE + 0.5 Dice, as in the official
TransUNet/Swin-Unet training code), early stopping and inference so that the only
difference between them is the network (and the family-specific optimizer settings
from the model config). Networks come from their official/maintained implementations:

* U-Net          segmentation_models_pytorch 0.5.0 ``smp.Unet`` (MIT), ResNet-34 encoder, ImageNet
* Attention U-Net MONAI 1.6.1 ``monai.networks.nets.AttentionUnet`` (Apache-2.0), Oktay et al. 2018;
                  no public pretrained 2D weights exist -> trained from scratch (explicit)
* TransUNet      Beckschen/TransUNet @ 02ef001 ``networks.vit_seg_modeling.VisionTransformer`` (Apache-2.0),
                  R50+ViT-B/16 ImageNet-21k weights (Google ``R50+ViT-B_16.npz``)
* Swin-Unet      HuCaoFighting/Swin-Unet @ f48f623 ``networks.vision_transformer.SwinUnet`` with the repo's
                  ``swin_tiny_patch4_window7_224_lite.yaml``; Swin-T ImageNet-1k weights (Microsoft)

Inference: the image is resized (stretch) to the network input, the probability map is
resized back to the ORIGINAL resolution with bilinear interpolation, and saved as
float16 in the unified ``defectbench.seg.v1`` format.
"""
from __future__ import annotations

import logging
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ...data.torch_datasets import NORMALIZATIONS, SegmentationDataset, load_image
from ...formats import save_seg_index, save_seg_prediction
from ...training.engine import Trainer, TrainerHooks
from ...utils.io import dump_json
from ...utils.seed import seed_everything, worker_init_fn
from ...weights import fetch, import_from_repo, third_party_dir
from .base import BaseAdapter

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- builders
class _SlashKeys:
    """Mapping view over an NpzFile that accepts Windows-style keys (``a\\b`` -> ``a/b``)."""

    def __init__(self, npz):
        self.npz = npz

    def __getitem__(self, k):
        return self.npz[k.replace("\\", "/")]

    def __contains__(self, k):
        return k.replace("\\", "/") in self.npz.files


def build_network(mc: dict, n_out: int, input_size) -> torch.nn.Module:
    arch = mc["model"]["arch"]
    pre = mc["pretrained"]
    if pre.get("init") == "none" and not pre.get("from_scratch_explicit"):
        raise ValueError("from-scratch training must be explicit (pretrained.from_scratch_explicit: true)")
    if arch == "smp_unet":
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name=mc["model"]["encoder"],
                        encoder_weights="imagenet" if pre["init"] == "imagenet" else None,
                        in_channels=3, classes=n_out)
    if arch == "monai_attention_unet":
        from monai.networks.nets import AttentionUnet
        m = mc["model"]
        return AttentionUnet(spatial_dims=2, in_channels=3, out_channels=n_out, channels=tuple(m["channels"]),
                             strides=tuple(m["strides"]), dropout=float(m.get("dropout", 0.0)))
    if arch == "transunet":
        repo = third_party_dir("TransUNet")
        vsm = import_from_repo(repo, "networks.vit_seg_modeling")
        cfg = vsm.CONFIGS[mc["model"]["vit_name"]]
        cfg.n_classes = n_out
        cfg.n_skip = int(mc["model"].get("n_skip", 3))
        size = int(input_size[0])
        if "R50" in mc["model"]["vit_name"]:
            cfg.patches.grid = (size // 16, size // 16)
        net = vsm.VisionTransformer(cfg, img_size=size, num_classes=n_out)
        if pre["init"] == "imagenet21k":
            npz = fetch(pre["url"])
            # the official load_from builds npz keys with os.path.join -> backslashes on Windows;
            # a key-normalising view keeps the upstream code unmodified
            net.load_from(weights=_SlashKeys(np.load(npz)))
        return net
    if arch == "swin_unet":
        repo = third_party_dir("Swin-Unet")
        vt = import_from_repo(repo, "networks.vision_transformer")
        conf = import_from_repo(repo, "config")
        ck = fetch(pre["url"]) if pre["init"] == "imagenet" else None
        args = Namespace(cfg=str(repo / mc["model"]["repo_cfg"]), opts=["MODEL.PRETRAIN_CKPT", str(ck)] if ck else None,
                         batch_size=None, zip=False, cache_mode=None, resume=None, accumulation_steps=None,
                         use_checkpoint=False, amp_opt_level=None, tag=None, eval=False, throughput=False)
        config = conf.get_config(args)
        if int(config.DATA.IMG_SIZE) != int(input_size[0]):
            raise ValueError(f"Swin-Unet repo config expects {config.DATA.IMG_SIZE}px input")
        net = vt.SwinUnet(config, img_size=config.DATA.IMG_SIZE, num_classes=n_out)
        if ck:
            net.load_from(config)
        return net
    raise ValueError(arch)


def backbone_parameters(model: torch.nn.Module, arch: str):
    """Pretrained encoder parameters (used by train.freeze_backbone_epochs)."""
    if arch == "smp_unet":
        return list(model.encoder.parameters())
    if arch == "transunet":
        return list(model.transformer.parameters())
    if arch == "swin_unet":
        s = model.swin_unet
        return list(s.patch_embed.parameters()) + [p for l in s.layers for p in l.parameters()]
    return []  # MONAI Attention U-Net: trained from scratch, nothing pretrained to freeze


def dice_loss(logits, target, n_out: int, eps: float = 1e-5):
    """Soft Dice over classes (softmax) as in TransUNet's utils.DiceLoss; binary -> sigmoid."""
    if n_out == 1:
        p = torch.sigmoid(logits[:, 0])
        t = target.float()
        inter = (p * t).sum()
        return 1 - (2 * inter + eps) / (p.sum() + t.sum() + eps)
    p = torch.softmax(logits, 1)
    oh = F.one_hot(target, n_out).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return (1 - (2 * inter + eps) / (den + eps)).mean()


def seg_loss(logits, target, n_out):
    if n_out == 1:
        ce = F.binary_cross_entropy_with_logits(logits[:, 0], target.float())
    else:
        ce = F.cross_entropy(logits, target)
    d = dice_loss(logits.float(), target, n_out)
    return 0.5 * ce + 0.5 * d, {"ce": ce.detach(), "dice": d.detach()}


class SegmentationAdapter(BaseAdapter):
    task = "seg"

    def __init__(self, run_cfg: dict):
        super().__init__(run_cfg)
        dev = run_cfg.get("device", "cuda")
        self.device = torch.device(dev if (dev == "cpu" or torch.cuda.is_available()) else "cpu")
        self.mode = self.mc["model"].get("seg_mode", "multiclass")
        self.n_out = 1 if self.mode == "binary" else self.num_classes + 1
        self.input_size = tuple(self.mc["model"]["input_size"])
        self.normalize = self.mc["model"].get("normalize", "imagenet")
        self.thr_mask = int((self.ds.get("mask") or {}).get("binarize_threshold", 128))
        self.weak = bool(self.mc.get("weak_box_supervision", False))

    def build(self):
        seed_everything(int(self.run_cfg["seed"]), bool(self.mc["train"].get("deterministic", False)))
        return build_network(self.mc, self.n_out, self.input_size).to(self.device)

    def count_parameters(self, model) -> dict:
        return {"total": sum(p.numel() for p in model.parameters()),
                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}

    def _loader(self, which, train):
        t = self.mc["train"]
        ds = SegmentationDataset(self.records(which), self.raw_root, self.input_size, self.mode, self.thr_mask,
                                 self.normalize, aug=self.mc.get("augment") if train else None, train=train,
                                 channel_policy=self.run_cfg["input"]["channel_policy"], weak_boxes=self.weak)
        g = torch.Generator()
        g.manual_seed(int(self.run_cfg["seed"]))
        return torch.utils.data.DataLoader(ds, batch_size=int(t["batch_size"]) if train else int(t.get("eval_batch_size", 8)),
                                           shuffle=train, drop_last=train and len(ds) > int(t["batch_size"]),
                                           num_workers=int(t.get("num_workers", 2)), worker_init_fn=worker_init_fn,
                                           generator=g, pin_memory=self.device.type == "cuda")

    def train(self, resume: str | None = None) -> dict:
        model = self.build()
        params = self.count_parameters(model)
        dump_json(params, self.run_dir / "params.json")
        tl, vl = self._loader("train", True), self._loader("val", False)
        n_out = self.n_out

        def train_step(model, batch, dev):
            x, y, _ = batch
            logits = model(x.to(dev, non_blocking=True))
            return seg_loss(logits, y.to(dev, non_blocking=True), n_out)

        @torch.no_grad()
        def val_loss(model, loader, dev, amp_ctx):
            model.eval()
            tot, n = 0.0, 0
            for x, y, _ in loader:
                with amp_ctx():
                    l, _ = seg_loss(model(x.to(dev)), y.to(dev), n_out)
                tot += float(l)
                n += 1
            return tot / max(1, n)

        @torch.no_grad()
        def evaluate(model, dev):
            """Model-selection metric (threshold-free): pixel average precision of the defect map on the
            validation split at network resolution, from 1000-bin score histograms. The pooled Dice at
            t=0.5 is logged for information only (it is much noisier as a selection criterion)."""
            model.eval()
            nb = 1000
            pos = torch.zeros(nb, dtype=torch.float64, device=dev)
            neg = torch.zeros(nb, dtype=torch.float64, device=dev)
            tp = fp = fn = 0
            for x, y, _ in vl:
                p = self._prob(model(x.to(dev)).float())
                d = p[:, 0] if n_out == 1 else 1 - p[:, 0]
                g = y.to(dev) > 0
                b = (d.clamp(0, 1) * (nb - 1)).round().long()
                pos += torch.bincount(b[g], minlength=nb).double()
                neg += torch.bincount(b[~g], minlength=nb).double()
                h = d >= 0.5
                tp += int((h & g).sum()); fp += int((h & ~g).sum()); fn += int((~h & g).sum())
            ctp = pos.flip(0).cumsum(0)
            cfp = neg.flip(0).cumsum(0)
            npos = float(pos.sum())
            if npos == 0:
                ap = 0.0
            else:
                prec = ctp / (ctp + cfp).clamp(min=1)
                rec = ctp / npos
                ap = float(((rec - torch.cat([rec.new_zeros(1), rec[:-1]])) * prec).sum())
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            return {"val_metric": ap, "dice@0.5": dice}

        t = dict(self.mc["train"])
        t["_input_desc"] = f"{self.input_size}"
        arch = self.mc["model"]["arch"]
        bp = (lambda m: backbone_parameters(m, arch)) if int(t.get("freeze_backbone_epochs", 0)) > 0 else None
        trainer = Trainer(model, TrainerHooks(train_step, val_loss, evaluate, bp), tl, vl, t, self.run_dir,
                          self.device)
        if resume:
            trainer.resume(resume)
        s = trainer.fit()
        s["parameters"] = params
        s["model_selection_metric"] = "val pixel average precision of the defect map (1000-bin histogram), network resolution"
        dump_json(s, self.run_dir / "train_summary.json")
        return s

    def _prob(self, logits):
        return torch.sigmoid(logits) if self.n_out == 1 else torch.softmax(logits, 1)

    def load(self, checkpoint=None):
        ck = Path(checkpoint) if checkpoint else self.run_dir / "checkpoints" / "best.pt"
        mc = {**self.mc, "pretrained": {**self.mc["pretrained"], "init": "none", "from_scratch_explicit": True}}
        model = build_network(mc, self.n_out, self.input_size)
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        return model.to(self.device).eval(), ck

    def _preprocess(self, rec):
        img = load_image(self.raw_root / rec["file"], self.run_cfg["input"]["channel_policy"])
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255)[None].to(self.device)
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False, antialias=True)
        norm = NORMALIZATIONS[self.normalize]
        if norm:
            m = torch.tensor(norm[0], device=self.device).view(1, 3, 1, 1)
            s = torch.tensor(norm[1], device=self.device).view(1, 3, 1, 1)
            x = (x - m) / s
        return x

    def _postprocess(self, logits, rec):
        p = self._prob(logits.float())
        p = F.interpolate(p, size=(rec["height"], rec["width"]), mode="bilinear", align_corners=False)
        return p[0].cpu().numpy()

    @torch.no_grad()
    def predict(self, which: str, checkpoint: str | None = None) -> Path:
        model, ck = self.load(checkpoint)
        out = self.pred_path(which)
        files = {}
        t0 = time.time()
        for rec in self.records(which):
            prob = self._postprocess(model(self._preprocess(rec)), rec)
            files[rec["image_id"]] = save_seg_prediction(out, rec["image_id"], prob, self.mode)
        save_seg_index(out, files, dataset=self.ds["name"], split=which, model=self.mc["name"],
                       class_names=self.classes, mode=self.mode,
                       extra={"checkpoint": str(ck), "seconds": round(time.time() - t0, 2),
                              "input_size": list(self.input_size), "resize": "stretch; bilinear back to original",
                              "normalize": self.normalize})
        return out

    def benchmark_model(self, checkpoint=None):
        model, _ = self.load(checkpoint)
        return (lambda x: model(x)), self._preprocess, self._postprocess, \
            f"model() = network forward only at {self.input_size}; pre = load+resize+normalize; post = softmax+resize"
