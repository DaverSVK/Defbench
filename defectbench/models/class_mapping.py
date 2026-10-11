"""Explicit class-id conventions of every framework and their inverse.

Canonical label = index into ``dataset.classes`` (0..K-1). Each adapter converts
its native label to the canonical one with :func:`to_canonical`; the tests assert
the round-trip for every framework.

framework          native label for canonical c      notes
-----------------  --------------------------------  -------------------------------------------
torchvision        c + 1                             0 is the implicit background class
coco_json          category_id = c + 1               used by our GT json and pycocotools
detr_family        c (category_id = c, 0-based json)  DEIMv2 / D-FINE / RT-DETRv2 with
                                                     remap_mscoco_category=False use category_id
                                                     directly as label -> we export 0-based ids
rfdetr             c (category_id = c in our export)  verified in adapters/rfdetr_runner.py at runtime
ultralytics        c                                 YOLO txt class index
detectron2         c                                 contiguous dataset id (thing_classes order)
mmdet              c                                 metainfo classes order
"""
from __future__ import annotations

OFFSETS = {
    "canonical": 0,
    "torchvision": 1,
    "coco_json": 1,
    "detr_family": 0,
    "rfdetr": 0,
    "ultralytics": 0,
    "detectron2": 0,
    "mmdet": 0,
}


def to_native(framework: str, canonical: int) -> int:
    return int(canonical) + OFFSETS[framework]


def to_canonical(framework: str, native: int) -> int:
    return int(native) - OFFSETS[framework]
