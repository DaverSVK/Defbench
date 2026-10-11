"""Box / coordinate conventions and exact inverse transforms.

Canonical convention used everywhere in the project
---------------------------------------------------
* Boxes are ``[x1, y1, x2, y2]`` in **continuous pixel-edge coordinates** of the
  ORIGINAL image: the pixel with column index ``c`` spans ``[c, c+1)``.
  A box that covers pixel columns ``c0..c1`` (inclusive) is ``x1=c0, x2=c1+1``.
  COCO ``[x, y, w, h]`` uses the same continuous system (``w = x2 - x1``).
* Pascal VOC XML stores 1-based inclusive pixel indices ``xmin..xmax``; therefore
  ``x1 = xmin - 1, x2 = xmax`` (see :func:`voc_to_xyxy`).
* Predictions from every adapter must be mapped back to this system before
  evaluation (removal of padding, reversal of resize / letterbox).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def voc_to_xyxy(xmin: float, ymin: float, xmax: float, ymax: float) -> list[float]:
    """1-based inclusive VOC pixel indices -> continuous 0-based pixel-edge box."""
    return [float(xmin) - 1.0, float(ymin) - 1.0, float(xmax), float(ymax)]


def xyxy_to_xywh(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)


def xywh_to_xyxy(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)


def cxcywh_norm_to_xyxy(b, width: float, height: float):
    """Normalised (cx, cy, w, h) in [0,1] (YOLO / DETR style) -> absolute xyxy."""
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    cx, cy, w, h = b[:, 0] * width, b[:, 1] * height, b[:, 2] * width, b[:, 3] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def xyxy_to_cxcywh_norm(b, width: float, height: float):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    return np.stack([(b[:, 0] + w / 2) / width, (b[:, 1] + h / 2) / height, w / width, h / height], axis=1)


def clip_boxes(b, width: float, height: float):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, width)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, height)
    return b


def box_area(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def box_iou(a, b):
    """Pairwise IoU matrix between (N,4) and (M,4) xyxy boxes (continuous coordinates)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


@dataclass(frozen=True)
class ResizeTransform:
    """Record of a (possibly aspect-preserving) resize + padding applied to an image.

    ``scale_x/scale_y`` multiply original coordinates; ``pad_x/pad_y`` are added
    afterwards (letterbox offset). Stretch resize: pad = 0, scale_x != scale_y.
    """
    orig_w: int
    orig_h: int
    scale_x: float
    scale_y: float
    pad_x: float = 0.0
    pad_y: float = 0.0

    @staticmethod
    def stretch(orig_w, orig_h, new_w, new_h) -> "ResizeTransform":
        return ResizeTransform(orig_w, orig_h, new_w / orig_w, new_h / orig_h)

    @staticmethod
    def letterbox(orig_w, orig_h, new_w, new_h, center: bool = True) -> "ResizeTransform":
        s = min(new_w / orig_w, new_h / orig_h)
        rw, rh = orig_w * s, orig_h * s
        px = (new_w - rw) / 2 if center else 0.0
        py = (new_h - rh) / 2 if center else 0.0
        return ResizeTransform(orig_w, orig_h, s, s, px, py)

    def forward_boxes(self, b):
        b = np.asarray(b, dtype=np.float64).reshape(-1, 4).copy()
        b[:, [0, 2]] = b[:, [0, 2]] * self.scale_x + self.pad_x
        b[:, [1, 3]] = b[:, [1, 3]] * self.scale_y + self.pad_y
        return b

    def inverse_boxes(self, b, clip: bool = True):
        """Map boxes from the network input frame back to the original image frame."""
        b = np.asarray(b, dtype=np.float64).reshape(-1, 4).copy()
        b[:, [0, 2]] = (b[:, [0, 2]] - self.pad_x) / self.scale_x
        b[:, [1, 3]] = (b[:, [1, 3]] - self.pad_y) / self.scale_y
        return clip_boxes(b, self.orig_w, self.orig_h) if clip else b


def mask_to_boxes(mask: np.ndarray, connectivity: int = 8, min_area: int = 1):
    """Connected components of a binary mask -> list of (xyxy box, area, component mask).

    Boxes follow the pixel-edge convention (``x2 = last_col + 1``). Components with
    fewer than ``min_area`` pixels are dropped. Used for DERIVED annotations only.
    """
    import cv2
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() == 0:
        return []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=connectivity)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out.append(([float(x), float(y), float(x + w), float(y + h)], int(area), lab == i))
    return out
