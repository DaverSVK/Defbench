"""Class-id mapping between frameworks and the canonical labels."""
import pytest

from defectbench.evaluation.detection import build_coco_gt
from defectbench.models.class_mapping import OFFSETS, to_canonical, to_native


@pytest.mark.parametrize("fw", sorted(OFFSETS))
def test_roundtrip(fw):
    for c in range(10):
        assert to_canonical(fw, to_native(fw, c)) == c


def test_torchvision_background_offset():
    assert to_native("torchvision", 0) == 1  # label 0 is reserved for background
    assert to_canonical("torchvision", 1) == 0


def test_coco_gt_category_ids_match_mapping():
    recs = [{"image_id": "a", "width": 10, "height": 10, "boxes": [[0, 0, 5, 5], [1, 1, 3, 3]], "labels": [0, 2]}]
    gt, _ = build_coco_gt(recs, ["x", "y", "z"])
    assert [a["category_id"] for a in gt["annotations"]] == [to_native("coco_json", 0), to_native("coco_json", 2)]
    assert [c["name"] for c in sorted(gt["categories"], key=lambda c: c["id"])] == ["x", "y", "z"]
