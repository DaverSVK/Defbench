"""Common detection evaluator: perfect / wrong / duplicate / empty predictions."""
import numpy as np
import pytest

from defectbench.evaluation.detection import (coco_evaluate, curve_from_matches, evaluate_detection,
                                              match_predictions, operating_point, select_threshold)
from defectbench.formats import make_det_record

CLASSES = ["a", "b"]


def _recs():
    return [
        {"image_id": "i1", "width": 100, "height": 100, "boxes": [[10, 10, 40, 40], [50, 50, 90, 90]], "labels": [0, 1]},
        {"image_id": "i2", "width": 100, "height": 100, "boxes": [[5, 5, 30, 30]], "labels": [0]},
        {"image_id": "i3", "width": 100, "height": 100, "boxes": [], "labels": []},  # negative image
    ]


def _payload(preds):
    return {"format": "defectbench.det.v1", "predictions": preds, "class_names": CLASSES,
            "max_detections_per_image": 100}


def _perfect():
    return _payload([make_det_record(r["image_id"], r["boxes"], [0.9] * len(r["boxes"]), r["labels"]) for r in _recs()])


def test_perfect_predictions():
    res = coco_evaluate(_recs(), _perfect(), CLASSES)
    assert res["coco"]["AP"] == pytest.approx(1.0)
    assert res["coco"]["AP50"] == pytest.approx(1.0)
    assert res["coco"]["AR100"] == pytest.approx(1.0)
    assert res["per_class"]["a"]["AP"] == pytest.approx(1.0)
    m = match_predictions(_recs(), _perfect())
    op = operating_point(m, 0.5, CLASSES)
    assert (op["tp"], op["fp"], op["fn"]) == (3, 0, 0)
    assert op["precision"] == op["recall"] == op["f1"] == 1.0


def test_wrong_class_gives_zero_ap_and_classification_errors():
    preds = [make_det_record(r["image_id"], r["boxes"], [0.9] * len(r["boxes"]), [1 - l for l in r["labels"]])
             for r in _recs()]
    res = coco_evaluate(_recs(), _payload(preds), CLASSES)
    assert res["coco"]["AP"] == pytest.approx(0.0)
    op = operating_point(match_predictions(_recs(), _payload(preds)), 0.5, CLASSES)
    assert op["tp"] == 0 and op["fp"] == 3 and op["fn"] == 3
    assert op["fp_by_type"] == {"classification": 3}


def test_duplicates_are_false_positives():
    r = _recs()
    preds = [make_det_record("i1", [[10, 10, 40, 40], [11, 10, 40, 40], [50, 50, 90, 90]], [0.9, 0.8, 0.7], [0, 0, 1]),
             make_det_record("i2", [[5, 5, 30, 30]], [0.9], [0]),
             make_det_record("i3", [], [], [])]
    m = match_predictions(r, _payload(preds))
    op = operating_point(m, 0.0, CLASSES)
    assert (op["tp"], op["fp"], op["fn"]) == (3, 1, 0)
    assert op["fp_by_type"] == {"duplicate": 1}
    # the higher-scoring box is the one matched
    tps = [d for d in m["dets"] if d[1]]
    assert all(d[0] in (0.9, 0.7) for d in tps)


def test_localization_error_and_background():
    r = _recs()
    preds = [make_det_record("i1", [[25, 25, 55, 55]], [0.9], [0]),   # IoU ~0.13 with GT a -> localization
             make_det_record("i2", [], [], []),
             make_det_record("i3", [[0, 0, 10, 10]], [0.8], [0])]      # negative image -> background FP
    m = match_predictions(r, _payload(preds))
    op = operating_point(m, 0.0, CLASSES)
    assert op["fp_by_type"] == {"background": 1, "localization": 1}
    assert op["tp"] == 0 and op["fn"] == 3


def test_empty_predictions_and_empty_gt():
    empty = _payload([make_det_record(r["image_id"], [], [], []) for r in _recs()])
    res = coco_evaluate(_recs(), empty, CLASSES)
    assert res["coco"]["AP"] == 0.0 and res["n_detections"] == 0
    op = operating_point(match_predictions(_recs(), empty), 0.5, CLASSES)
    assert op["tp"] == 0 and op["fn"] == 3 and op["precision"] == 0.0 and op["recall"] == 0.0
    # split without any GT box: COCO AP undefined -> None, not 0
    neg = [{"image_id": "n", "width": 10, "height": 10, "boxes": [], "labels": []}]
    res2 = coco_evaluate(neg, _payload([make_det_record("n", [[0, 0, 5, 5]], [0.5], [0])]), CLASSES)
    assert res2["coco"]["AP"] is None


def test_class_missing_in_split_is_none():
    recs = [{"image_id": "i1", "width": 50, "height": 50, "boxes": [[0, 0, 10, 10]], "labels": [0]}]
    res = coco_evaluate(recs, _payload([make_det_record("i1", [[0, 0, 10, 10]], [0.9], [0])]), CLASSES)
    assert res["per_class"]["b"]["AP"] is None and res["per_class"]["b"]["n_gt"] == 0


def test_ap_uses_score_ranking_not_single_threshold():
    """A low-scoring TP ranked below an FP lowers AP but still counts - AP is not binarised."""
    recs = [{"image_id": "i", "width": 100, "height": 100, "boxes": [[0, 0, 10, 10]], "labels": [0]}]
    good = _payload([make_det_record("i", [[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.1], [0, 0])])
    bad = _payload([make_det_record("i", [[0, 0, 10, 10], [50, 50, 60, 60]], [0.1, 0.9], [0, 0])])
    ap_good = coco_evaluate(recs, good, CLASSES)["coco"]["AP50"]
    ap_bad = coco_evaluate(recs, bad, CLASSES)["coco"]["AP50"]
    assert ap_good == pytest.approx(1.0) and 0.4 < ap_bad < 0.6


def test_threshold_selected_on_validation_and_frozen():
    recs = _recs()
    val = _payload([make_det_record("i1", [[10, 10, 40, 40], [60, 0, 90, 20]], [0.8, 0.3], [0, 0]),
                    make_det_record("i2", [[5, 5, 30, 30]], [0.7], [0]),
                    make_det_record("i3", [], [], [])])
    sel = select_threshold(match_predictions(recs, val))
    assert sel["threshold"] == pytest.approx(0.7)
    res = evaluate_detection(recs, val, recs, _perfect(), CLASSES)
    assert res["threshold_selection"]["threshold"] == pytest.approx(0.7)
    assert res["working_point"]["f1"] == 1.0
    c = curve_from_matches(match_predictions(recs, val))
    assert np.all(np.diff(c["thresholds"]) < 0)
