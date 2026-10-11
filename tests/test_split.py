"""Split isolation (groups, duplicates, near duplicates) and reproducibility."""
import numpy as np
import pytest

from defectbench.data.split import build_groups, iterative_group_stratification, make_split


def _records(n=120, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        labels = [int(rng.integers(0, 3))]
        if rng.random() < 0.2:
            labels.append(int(rng.integers(0, 3)))
        recs.append({"image_id": f"d/{i}", "labels": labels, "boxes": [[0, 0, 1, 1]] * len(labels),
                     "mask_class": None, "is_normal": False, "group_key": f"g{i // 4}",
                     "pixel_hash": f"p{i}", "sha256": f"s{i}", "original_split": None})
    # exact duplicates in different source groups
    if n > 77:
        recs[5]["pixel_hash"] = recs[77]["pixel_hash"] = "dup"
    # normal images
    for r in recs[int(n * 0.83):]:
        r["labels"], r["boxes"], r["is_normal"] = [], [], True
    return recs


CFG = {"name": "t", "classes": ["a", "b", "c"], "near_duplicates": {"use_for_grouping": True},
       "split": {"strategy": "stratified_group", "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
                 "split_seed": 123}, "group": {"method": "regex"}}


def test_groups_and_duplicates_never_cross_splits():
    recs = _records()
    near = [{"a": "d/1", "b": "d/50"}]
    s = make_split(CFG, recs, near)
    a = s["assignments"]
    for g in {r["group_key"] for r in recs}:
        assert len({a[r["image_id"]] for r in recs if r["group_key"] == g}) == 1
    assert a["d/5"] == a["d/77"]
    assert a["d/1"] == a["d/50"]
    assert s["verification"]["groups_spanning_splits"] == 0
    assert set(a.values()) == {"train", "val", "test"}


def test_split_is_reproducible_and_seed_dependent():
    recs = _records()
    s1 = make_split(CFG, recs, [])
    s2 = make_split(CFG, recs, [])
    assert s1["split_hash"] == s2["split_hash"]
    cfg2 = {**CFG, "split": {**CFG["split"], "split_seed": 7}}
    assert make_split(cfg2, recs, [])["split_hash"] != s1["split_hash"]


def test_ratios_roughly_respected_and_all_labels_in_every_split():
    recs = _records(400, seed=3)
    for i, r in enumerate(recs):
        r["group_key"] = f"g{i // 2}"
    s = make_split(CFG, recs, [])
    rr = s["verification"]["realised_ratios"]
    assert abs(rr["train"] - 0.7) < 0.05 and abs(rr["test"] - 0.15) < 0.05
    for sp in ("train", "val", "test"):
        assert all(v > 0 for v in s["stats"][sp]["images_per_class"].values())
        assert s["stats"][sp]["normal_images"] > 0


def test_iterative_stratification_assigns_every_group():
    vecs = {f"g{i}": np.eye(3)[i % 3] for i in range(30)}
    a = iterative_group_stratification(vecs, {"train": 0.7, "val": 0.15, "test": 0.15}, 0)
    assert set(a) == set(vecs)


def test_provided_split_keeps_test_folder():
    recs = _records(60)
    for i, r in enumerate(recs):
        r["original_split"] = "validation" if i >= 48 else "train"
        r["group_key"] = None
        r["pixel_hash"] = f"u{i}"
    cfg = {**CFG, "split": {**CFG["split"], "strategy": "provided"}}
    s = make_split(cfg, recs, [])
    for i, r in enumerate(recs):
        if i >= 48:
            assert s["assignments"][r["image_id"]] == "test"
        else:
            assert s["assignments"][r["image_id"]] in ("train", "val")
