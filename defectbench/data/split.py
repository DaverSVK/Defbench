"""Leakage-free, reproducible train/val/test splitting.

Algorithm (``strategy: stratified_group``)
1. Build *split groups* with union-find over valid images:
   - same source / sequence key (``group_key``; GC10 coil id, MT acquisition sequence),
   - pixel-identical images (exact duplicates),
   - suspicious near duplicates (pHash distance <= threshold) if ``use_for_grouping``.
2. Each group gets a label vector: number of images containing each defect class
   (images with several classes count for each) + number of normal images.
3. Multilabel *iterative stratification* over groups (Sechidis et al., 2011, adapted
   to groups): repeatedly take the label with the fewest remaining images, assign
   its groups (largest first) to the split with the highest remaining demand for
   that label (ties -> highest total remaining demand -> seeded random).
4. The split is done on whole images BEFORE any augmentation / cropping.

The split seed is fixed in the dataset config and is independent of the training
seeds, so seeds 17/42/73 all use the same manifest.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..utils.io import dump_json, load_json, stable_hash

log = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")


class UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def image_label_set(r: dict) -> set:
    s = set(r["labels"])
    if r.get("mask_class") is not None:
        s.add(r["mask_class"])
    return s


def build_groups(records: list[dict], near_pairs: list[dict] | None, use_near: bool) -> dict[str, list[str]]:
    ids = [r["image_id"] for r in records]
    uf = UnionFind(ids)
    by_key, by_pix = defaultdict(list), defaultdict(list)
    for r in records:
        if r.get("group_key"):
            by_key[r["group_key"]].append(r["image_id"])
        by_pix[r["pixel_hash"]].append(r["image_id"])
    for members in list(by_key.values()) + list(by_pix.values()):
        for m in members[1:]:
            uf.union(members[0], m)
    idset = set(ids)
    if use_near and near_pairs:
        for p in near_pairs:
            if p["a"] in idset and p["b"] in idset:
                uf.union(p["a"], p["b"])
    groups = defaultdict(list)
    for i in ids:
        groups[uf.find(i)].append(i)
    return dict(groups)


def iterative_group_stratification(group_vecs: dict[str, np.ndarray], ratios: dict[str, float],
                                   seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    names = list(ratios)
    r = np.array([ratios[n] for n in names], dtype=float)
    r = r / r.sum()
    gids = sorted(group_vecs)
    V = np.stack([group_vecs[g] for g in gids]).astype(float)       # (G, L)
    sizes = V.sum(1)
    total = V.sum(0)                                               # per label
    demand_lab = np.outer(r, total)                                # (S, L)
    demand_tot = r * V.sum()
    assigned: dict[str, str] = {}
    remaining = set(range(len(gids)))
    while remaining:
        rem = sorted(remaining)
        lab_left = V[rem].sum(0)
        cand_labels = np.nonzero(lab_left > 0)[0]
        if len(cand_labels) == 0:
            # groups without any label (should not happen) -> by total demand
            lab = None
            members = rem
        else:
            lab = cand_labels[np.argmin(lab_left[cand_labels])]
            members = [g for g in rem if V[g, lab] > 0]
        # largest groups first (hardest to place), seeded tie-break
        members = sorted(members, key=lambda g: (-sizes[g], rng.random()))
        for g in members:
            if lab is not None:
                d = demand_lab[:, lab]
                best = np.flatnonzero(d == d.max())
            else:
                best = np.arange(len(names))
            if len(best) > 1:
                dt = demand_tot[best]
                best = best[dt == dt.max()]
            s = int(best[rng.integers(len(best))]) if len(best) > 1 else int(best[0])
            assigned[gids[g]] = names[s]
            demand_lab[s] -= V[g]
            demand_tot[s] -= V[g].sum()
            remaining.discard(g)
    return assigned


def make_split(cfg: dict, records: list[dict], near_pairs: list[dict] | None) -> dict:
    scfg = cfg["split"]
    strategy = scfg.get("strategy", "stratified_group")
    ratios = scfg.get("ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    seed = int(scfg.get("split_seed", 0))
    classes = cfg["classes"]
    K = len(classes)
    use_near = bool((cfg.get("near_duplicates") or {}).get("use_for_grouping", True))
    recs = {r["image_id"]: r for r in records}
    groups = build_groups(records, near_pairs, use_near)

    def vec(members):
        v = np.zeros(K + 1)
        for m in members:
            ls = image_label_set(recs[m])
            if recs[m]["is_normal"] or not ls:
                v[K] += 1
            for c in ls:
                v[c] += 1
        return v

    assignment: dict[str, str] = {}
    provenance = {}
    if strategy == "stratified_group":
        gv = {g: vec(m) for g, m in groups.items()}
        ga = iterative_group_stratification(gv, ratios, seed)
        for g, m in groups.items():
            for i in m:
                assignment[i] = ga[g]
        provenance = {"origin": "generated", "method": "multilabel iterative stratification over leakage groups"}
    elif strategy == "provided":
        # keep the archive split: its non-train part becomes the test set; val is carved from train
        train_groups = {g: m for g, m in groups.items()
                        if all(recs[i].get("original_split") == "train" for i in m)}
        other = {g: m for g, m in groups.items() if g not in train_groups}
        crossing = [g for g, m in other.items() if any(recs[i].get("original_split") == "train" for i in m)]
        for g, m in other.items():
            for i in m:
                assignment[i] = "test" if recs[i].get("original_split") != "train" else "train"
        vr = ratios["val"] / (ratios["train"] + ratios["val"])
        gv = {g: vec(m) for g, m in train_groups.items()}
        ga = iterative_group_stratification(gv, {"train": 1 - vr, "val": vr}, seed)
        for g, m in train_groups.items():
            for i in m:
                assignment[i] = ga[g]
        provenance = {"origin": "archive folders (train/validation)", "validation_folder_used_as": "test",
                      "val": "carved from train folder by group stratification",
                      "groups_crossing_provided_split": len(crossing),
                      "warning": ("leakage groups cross the provided split" if crossing else None)}
    else:
        raise ValueError(f"unknown split strategy {strategy}")

    # ------------------------------------------------------------ verification
    group_of = {i: g for g, m in groups.items() for i in m}
    split_of_group = defaultdict(set)
    for i, s in assignment.items():
        split_of_group[group_of[i]].add(s)
    violations = [g for g, ss in split_of_group.items() if len(ss) > 1]
    pix = defaultdict(set)
    for i, s in assignment.items():
        pix[recs[i]["pixel_hash"]].add(s)
    dup_cross = sum(1 for v in pix.values() if len(v) > 1)
    near_cross = 0
    for p in near_pairs or []:
        if p["a"] in assignment and p["b"] in assignment and assignment[p["a"]] != assignment[p["b"]]:
            near_cross += 1

    stats = {}
    for s in SPLITS:
        ids = [i for i, x in assignment.items() if x == s]
        img_cls = Counter()
        obj_cls = Counter()
        for i in ids:
            for c in image_label_set(recs[i]):
                img_cls[classes[c]] += 1
            for c in recs[i]["labels"]:
                obj_cls[classes[c]] += 1
        stats[s] = {"images": len(ids), "normal_images": sum(1 for i in ids if recs[i]["is_normal"]),
                    "objects": sum(len(recs[i]["labels"]) for i in ids),
                    "images_per_class": {c: img_cls.get(c, 0) for c in classes},
                    "objects_per_class": {c: obj_cls.get(c, 0) for c in classes},
                    "groups": len({group_of[i] for i in ids})}
    n = len(assignment)
    split = {
        "dataset": cfg["name"], "strategy": strategy, "ratios": ratios, "split_seed": seed,
        "provenance": provenance,
        "grouping": {"group_key_method": (cfg.get("group") or {}).get("method", "none"),
                     "near_duplicates_used": use_near, "n_groups": len(groups),
                     "largest_group": max(len(m) for m in groups.values())},
        "assignments": dict(sorted(assignment.items())),
        "image_sha256": {i: recs[i]["sha256"] for i in sorted(assignment)},
        "verification": {"groups_spanning_splits": len(violations),
                         "pixel_duplicates_across_splits": dup_cross,
                         "near_duplicate_pairs_across_splits": near_cross,
                         "realised_ratios": {s: round(stats[s]["images"] / n, 4) for s in SPLITS}},
        "stats": stats,
    }
    split["split_hash"] = stable_hash({"a": split["assignments"], "h": split["image_sha256"]})
    if violations or dup_cross:
        raise RuntimeError(f"split verification failed: {len(violations)} groups span splits, "
                           f"{dup_cross} duplicates cross splits")
    return split


# ------------------------------------------------------------------ manifest I/O
def manifest_record(r: dict) -> dict:
    keep = ["image_id", "file", "alt_files", "width", "height", "mode", "channels", "sha256", "pixel_hash",
            "phash", "boxes", "labels", "box_source", "mask_file", "mask_source", "mask_class", "is_normal",
            "original_split", "group_key", "annotation_file", "exclude_from_detection", "rgb_channels_identical"]
    return {k: r.get(k) for k in keep}


def write_manifest(cfg: dict, records: list[dict], audit_report: dict) -> dict:
    valid = [manifest_record(r) for r in records if not r["excluded"]]
    manifest = {
        "dataset": cfg["name"], "classes": cfg["classes"], "raw_root": cfg["raw_root"],
        "coordinate_system": "xyxy continuous pixel-edge coordinates of the original image",
        "tasks": cfg.get("tasks", {}),
        "mask": cfg.get("mask"), "derived_boxes": cfg.get("derived_boxes"),
        "input": cfg.get("input"),
        "n_images": len(valid),
        "excluded": [{"image_id": r["image_id"], "file": r["file"], "reason": r["exclude_reason"]}
                     for r in records if r["excluded"]],
        "images": valid,
    }
    manifest["manifest_hash"] = stable_hash(valid)
    dump_json(manifest, Path(cfg["processed_root_abs"]) / "manifest.json", indent=None)
    return manifest


def load_manifest(cfg: dict) -> dict:
    p = Path(cfg["processed_root_abs"]) / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `defectbench prepare-data --dataset {cfg['name']}` first")
    return load_json(p)


def load_split(cfg: dict, name: str = "default") -> dict:
    p = Path(cfg["processed_root_abs"]) / "splits" / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run `defectbench prepare-data --dataset {cfg['name']}` first")
    return load_json(p)


def split_records(manifest: dict, split: dict, which: str, task: str | None = None) -> list[dict]:
    """Records of one split. ``task='bbox'`` drops images flagged exclude_from_detection."""
    a = split["assignments"]
    out = [r for r in manifest["images"] if a.get(r["image_id"]) == which]
    if task == "bbox":
        out = [r for r in out if not r.get("exclude_from_detection")]
    return out
