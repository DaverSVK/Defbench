"""PyTorch datasets for the in-process adapters (core environment).

Augmentations use ``torchvision.transforms.v2`` so that boxes (``tv_tensors.BoundingBoxes``)
and masks (``tv_tensors.Mask``) are transformed synchronously with the image. The
default policy is conservative to preserve fine defects: flips and mild photometric
jitter only - no random crops, no mosaic, no strong blur (see configs).

Channel policy: all three datasets are grayscale (NEU is stored as RGB with three
identical channels - verified by the audit). Images are loaded as single channel and
REPLICATED to 3 channels to match ImageNet/COCO-pretrained RGB backbones; the policy
is written to every run config (``input.channel_policy``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NORMALIZATIONS = {"imagenet": (IMAGENET_MEAN, IMAGENET_STD), "half": ((0.5,) * 3, (0.5,) * 3), "none": None}


def load_image(path: Path, channel_policy: str = "replicate_gray_to_rgb") -> np.ndarray:
    with Image.open(path) as im:
        if channel_policy == "replicate_gray_to_rgb":
            g = np.asarray(im.convert("L"))
            return np.repeat(g[..., None], 3, axis=2)
        return np.asarray(im.convert("RGB"))


def load_mask(path: Path, threshold: int) -> np.ndarray:
    with Image.open(path) as im:
        m = np.asarray(im)
    if m.ndim == 3:
        m = m[..., 0]
    return m >= threshold


def gt_class_map(rec: dict, raw_root: Path, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """(binary defect mask, class map with 0=background and c+1 for class c) at original resolution."""
    h, w = rec["height"], rec["width"]
    if rec.get("mask_file"):
        m = load_mask(raw_root / rec["mask_file"], threshold)
    else:
        m = np.zeros((h, w), bool)
    cm = np.zeros((h, w), np.int64)
    if rec.get("mask_class") is not None:
        cm[m] = rec["mask_class"] + 1
    return m, cm


def weak_box_class_map(rec: dict) -> np.ndarray:
    """WEAK SUPERVISION ONLY: pseudo class map from filled bounding boxes (0=background, c+1 inside boxes).

    Used exclusively by the explicitly enabled weakly-supervised segmentation experiment on box-only datasets.
    These pseudo masks are never used as ground truth for any pixel metric. Larger boxes are drawn first so
    that smaller (nested) boxes keep their class.
    """
    h, w = rec["height"], rec["width"]
    cm = np.zeros((h, w), np.int64)
    order = sorted(range(len(rec["boxes"])), key=lambda i: -((rec["boxes"][i][2] - rec["boxes"][i][0]) *
                                                           (rec["boxes"][i][3] - rec["boxes"][i][1])))
    for i in order:
        x1, y1, x2, y2 = rec["boxes"][i]
        cm[int(np.floor(y1)):int(np.ceil(y2)), int(np.floor(x1)):int(np.ceil(x2))] = rec["labels"][i] + 1
    return cm


def build_det_transforms(aug: dict | None, train: bool):
    from torchvision.transforms import v2 as T
    ops = []  # the image is wrapped as tv_tensors.Image by the dataset (ToImage would also touch label tensors)
    if train and aug:
        if aug.get("hflip", 0) > 0:
            ops.append(T.RandomHorizontalFlip(aug["hflip"]))
        if aug.get("vflip", 0) > 0:
            ops.append(T.RandomVerticalFlip(aug["vflip"]))
        cj = aug.get("color_jitter")
        if cj:
            ops.append(T.ColorJitter(brightness=cj.get("brightness", 0), contrast=cj.get("contrast", 0)))
    ops += [T.ToDtype(torch.float32, scale=True), T.SanitizeBoundingBoxes(min_size=1)]
    return T.Compose(ops)


class DetectionDataset(Dataset):
    """Returns (image float[0,1] 3xHxW at ORIGINAL resolution, target dict).

    Resizing to the network resolution is done by the model itself (torchvision's
    GeneralizedRCNNTransform), which also maps predictions back to the original size.
    """

    def __init__(self, records, raw_root, label_offset: int, aug: dict | None = None, train: bool = False,
                 channel_policy: str = "replicate_gray_to_rgb"):
        self.records = records
        self.raw_root = Path(raw_root)
        self.label_offset = label_offset
        self.tf = build_det_transforms(aug, train)
        self.channel_policy = channel_policy

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        from torchvision import tv_tensors
        r = self.records[i]
        img = load_image(self.raw_root / r["file"], self.channel_policy)
        h, w = img.shape[:2]
        boxes = torch.tensor(r["boxes"], dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w)),
            "labels": torch.tensor([l + self.label_offset for l in r["labels"]], dtype=torch.int64),
        }
        img_t, target = self.tf(tv_tensors.Image(torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)), target)
        target["image_id"] = r["image_id"]
        target["boxes"] = target["boxes"].as_subclass(torch.Tensor)
        return img_t, target


def det_collate(batch):
    return tuple(zip(*batch))


class SegmentationDataset(Dataset):
    """Image resized (stretch) to ``input_size``; mask resized with nearest neighbour.

    For evaluation the prediction is resized back to the original resolution and
    compared with the ORIGINAL mask (see adapters/segmentation.py).
    ``weak_boxes=True`` (explicit weakly-supervised experiment on box-only datasets) uses box-filled pseudo
    masks; it is refused for records that have real masks.
    """

    def __init__(self, records, raw_root, input_size, mode: str, threshold: int, normalize: str,
                 aug: dict | None = None, train: bool = False, channel_policy: str = "replicate_gray_to_rgb",
                 weak_boxes: bool = False):
        if weak_boxes and any(r.get("mask_file") for r in records):
            raise ValueError("weak box supervision is only allowed on datasets WITHOUT real pixel masks")
        self.weak_boxes = weak_boxes
        self.records = records
        self.raw_root = Path(raw_root)
        self.size = tuple(input_size)
        self.mode = mode
        self.thr = threshold
        self.norm = NORMALIZATIONS[normalize]
        self.aug = aug or {}
        self.train = train
        self.channel_policy = channel_policy

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        from torchvision import tv_tensors
        from torchvision.transforms import v2 as T
        r = self.records[i]
        img = load_image(self.raw_root / r["file"], self.channel_policy)
        if self.weak_boxes:
            cm = weak_box_class_map(r)
        else:
            _, cm = gt_class_map(r, self.raw_root, self.thr)
        target = cm if self.mode == "multiclass" else (cm > 0).astype(np.int64)
        ops = []
        if self.train:
            if self.aug.get("hflip", 0) > 0:
                ops.append(T.RandomHorizontalFlip(self.aug["hflip"]))
            if self.aug.get("vflip", 0) > 0:
                ops.append(T.RandomVerticalFlip(self.aug["vflip"]))
            cj = self.aug.get("color_jitter")
            if cj:
                ops.append(T.ColorJitter(brightness=cj.get("brightness", 0), contrast=cj.get("contrast", 0)))
        ops += [T.Resize(self.size, antialias=True), T.ToDtype(torch.float32, scale=True)]
        if self.norm:
            ops.append(T.Normalize(*self.norm))
        tf = T.Compose(ops)
        # v2.Resize uses NEAREST for Mask tv_tensors automatically
        x, m = tf(tv_tensors.Image(torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)),
                  tv_tensors.Mask(torch.from_numpy(target)))
        return x, m.as_subclass(torch.Tensor).long(), i
