"""Unified prediction validation and segmentation metrics with empty masks."""
import numpy as np
import pytest

from defectbench.evaluation.segmentation import (seg_to_detections, segmentation_metrics,
                                                 select_binarization_threshold)
from defectbench.formats import make_det_record, postprocess_det_record, validate_det_predictions


def test_validation_detects_unreverted_boxes_and_missing_images():
    payload = {"class_names": ["a"], "predictions": [make_det_record("x", [[0, 0, 700, 700]], [0.5], [0])]}
    probs = validate_det_predictions(payload, {"x": (200, 200), "y": (200, 200)})
    assert any("outside original image" in p for p in probs)
    assert any("no prediction record" in p for p in probs)


def test_postprocess_rejects_bad_label_and_applies_topk():
    rec = make_det_record("x", [[0, 0, 5, 5]] * 5, [0.1, 0.5, 0.0005, 0.9, 0.3], [0] * 5)
    out = postprocess_det_record(rec, 10, 10, 1, score_thr=0.001, max_dets=3)
    assert out["scores"] == [0.9, 0.5, 0.3]
    with pytest.raises(ValueError):
        postprocess_det_record(make_det_record("x", [[0, 0, 1, 1]], [0.5], [3]), 10, 10, 2)


def _bin_prob(p):
    return np.asarray(p, np.float32)[None]


def test_segmentation_empty_rules():
    gt = [np.zeros((4, 4), bool), np.zeros((4, 4), bool), np.ones((4, 4), bool)]
    pr = [_bin_prob(np.zeros((4, 4))), _bin_prob(np.ones((4, 4))), _bin_prob(np.zeros((4, 4)))]
    m = segmentation_metrics(gt, None, pr, "binary", 0.5, ["d"])
    # image1: both empty -> 1, image2: FP only -> 0, image3: FN only -> 0
    assert m["per_image_mean"]["dice"] == pytest.approx(1 / 3)
    assert m["pooled"]["tp"] == 0 and m["pooled"]["fp"] == 16 and m["pooled"]["fn"] == 16
    assert m["per_image_counts"]["empty_prediction"] == 2


def test_perfect_segmentation_and_threshold_selection():
    g = np.zeros((10, 10), bool); g[2:5, 2:5] = True
    p = _bin_prob(g * 0.8 + 0.1)
    sel = select_binarization_threshold([g], [p[0]])
    assert 0.1 < sel["threshold"] <= 0.9
    m = segmentation_metrics([g], None, [p], "binary", sel["threshold"], ["d"])
    assert m["pooled"]["dice"] == 1.0 and m["pooled"]["iou"] == 1.0


def test_multiclass_per_class_metrics():
    g = np.zeros((6, 6), bool); g[0:2, 0:2] = True
    gc = np.zeros((6, 6), int); gc[g] = 2          # class index 1 (+1)
    prob = np.zeros((3, 6, 6), np.float32); prob[0] = 1
    prob[:, 0:2, 0:2] = [[[0.1]], [[0.2]], [[0.7]]]
    m = segmentation_metrics([g], [gc], [prob], "multiclass", 0.5, ["c0", "c1"])
    assert m["per_class"]["c1"]["dice"] == 1.0
    assert m["per_class"]["c0"]["dice"] is None


def test_weak_box_pseudo_masks_only_for_box_datasets():
    from defectbench.data.torch_datasets import SegmentationDataset, weak_box_class_map
    rec = {"height": 10, "width": 10, "boxes": [[0, 0, 8, 8], [2, 2, 4, 4]], "labels": [0, 1]}
    cm = weak_box_class_map(rec)
    assert cm[0, 0] == 1 and cm[3, 3] == 2 and cm[9, 9] == 0  # nested smaller box keeps its class
    assert cm[7, 7] == 1 and cm[8, 8] == 0                     # pixel-edge convention: x2=8 excludes col 8
    with pytest.raises(ValueError):
        SegmentationDataset([{"mask_file": "m.png"}], ".", (32, 32), "multiclass", 128, "imagenet", weak_boxes=True)


def test_seg_to_detections():
    prob = np.zeros((1, 20, 20), np.float32)
    prob[0, 2:6, 3:9] = 0.9
    prob[0, 15, 15] = 0.8
    d = seg_to_detections(prob, "binary", 0.5, min_area=2, score_rule="max")
    assert d["boxes_xyxy"] == [[3.0, 2.0, 9.0, 6.0]] and d["labels"] == [0]
    assert d["scores"][0] == pytest.approx(0.9)
