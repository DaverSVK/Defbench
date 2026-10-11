"""Data audit: structure, statistics, integrity, invalid annotations and duplicates.

The audit decodes every image (and mask) once; per-image results are cached in
``<processed_root>/audit_cache.json`` keyed by file SHA-256, so a re-run is fast.
Nothing is invented: every number in the report is computed from the files.
"""
from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..utils.io import dump_json, load_json, sha256_file
from .geometry import mask_to_boxes
from .parsers import parse_dataset

log = logging.getLogger(__name__)


def _phash64(gray: np.ndarray) -> int:
    """64-bit DCT perceptual hash (same algorithm as imagehash.phash, no extra dependency)."""
    import cv2
    small = cv2.resize(gray.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small)[:8, :8]
    med = np.median(dct)
    bits = (dct > med).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def _pixel_hash(arr: np.ndarray) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _inspect_image(path: Path) -> dict:
    from PIL import Image
    out: dict = {}
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            out["mode"] = im.mode
            out["width"], out["height"] = im.size
            arr = np.asarray(im)
    except Exception as e:
        return {"corrupt": True, "error": f"{type(e).__name__}: {e}"}
    out["corrupt"] = False
    out["channels"] = 1 if arr.ndim == 2 else arr.shape[2]
    gray = arr if arr.ndim == 2 else arr[..., :3].mean(axis=2)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        out["rgb_channels_identical"] = bool(np.array_equal(arr[..., 0], arr[..., 1]) and
                                             np.array_equal(arr[..., 1], arr[..., 2]))
    out["pixel_hash"] = _pixel_hash(arr)
    out["phash"] = format(_phash64(gray), "016x")
    out["mean_intensity"] = float(gray.mean())
    return out


def _inspect_mask(path: Path, threshold: int, connectivity: int, min_area: int, exp_hw) -> dict:
    from PIL import Image
    try:
        with Image.open(path) as im:
            m = np.asarray(im)
            mode = im.mode
    except Exception as e:
        return {"corrupt": True, "error": f"{type(e).__name__}: {e}"}
    out = {"corrupt": False, "mode": mode, "shape": list(m.shape)}
    if m.ndim == 3:
        out["multichannel"] = True
        m = m[..., 0]
    out["size_matches_image"] = tuple(m.shape[:2]) == tuple(exp_hw)
    vals = np.unique(m)
    out["n_unique_values"] = int(len(vals))
    out["binary_0_255"] = bool(set(vals.tolist()) <= {0, 255})
    out["n_soft_pixels"] = int(((m > 0) & (m < 255)).sum())
    b = m >= threshold
    out["defect_pixels"] = int(b.sum())
    comps = mask_to_boxes(b, connectivity=connectivity, min_area=1)
    out["n_components_all"] = len(comps)
    out["n_components_small"] = int(sum(1 for c in comps if c[1] < min_area))
    out["derived_boxes"] = [c[0] for c in comps if c[1] >= min_area]
    out["derived_box_areas_px"] = [c[1] for c in comps if c[1] >= min_area]
    return out


def near_duplicate_pairs(phashes: list[int], max_distance: int) -> list[tuple[int, int, int]]:
    """All pairs (i, j, d) with Hamming distance d <= max_distance (vectorised)."""
    if len(phashes) < 2:
        return []
    h = np.array(phashes, dtype=np.uint64)
    out = []
    # popcount via byte lookup table
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    for i in range(len(h) - 1):
        x = np.bitwise_xor(h[i], h[i + 1:])
        d = lut[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)
        for j in np.nonzero(d <= max_distance)[0]:
            out.append((i, int(i + 1 + j), int(d[j])))
    return out


def run_audit(cfg: dict, write: bool = True) -> dict:
    t0 = time.time()
    raw_root = Path(cfg["raw_root_abs"])
    proc = Path(cfg["processed_root_abs"])
    records, parse_info = parse_dataset(cfg, raw_root)
    classes = cfg["classes"]
    cache_path = proc / "audit_cache.json"
    cache = load_json(cache_path) if cache_path.exists() else {}
    mcfg = cfg.get("mask", {})
    dcfg = cfg.get("derived_boxes", {})
    thr = int(mcfg.get("binarize_threshold", 128))
    conn = int(dcfg.get("connectivity", 8))
    min_area = int(dcfg.get("min_area", 1))

    # ---------------------------------------------------------------- per-image inspection
    for i, r in enumerate(records):
        p = raw_root / r["file"]
        try:
            fh = sha256_file(p)
        except OSError as e:
            r["file_error"] = str(e)
            r["excluded"] = True
            r["exclude_reason"] = "unreadable file"
            continue
        r["sha256"] = fh
        key = f"{fh}|{r.get('mask_file')}|{thr}|{conn}|{min_area}"
        if key not in cache:
            info = _inspect_image(p)
            if r.get("mask_file") and not info.get("corrupt"):
                info["mask"] = _inspect_mask(raw_root / r["mask_file"], thr, conn, min_area,
                                             (info["height"], info["width"]))
            cache[key] = info
        r["_img"] = cache[key]
        # alternative copies (same stem in several folders)
        alt_info = []
        for a in r.get("alt_files", []):
            ai = _inspect_image(raw_root / a)
            alt_info.append({"file": a, "sha256": sha256_file(raw_root / a), "pixel_hash": ai.get("pixel_hash")})
        r["_alt"] = alt_info
        if (i + 1) % 500 == 0:
            log.info("audited %d/%d images", i + 1, len(records))
    if write:
        dump_json(cache, cache_path, indent=None)

    # ---------------------------------------------------------------- validation of annotations
    problems = defaultdict(list)
    for r in records:
        info = r.get("_img", {})
        if info.get("corrupt"):
            problems["corrupt_image"].append({"image_id": r["image_id"], "error": info.get("error")})
            r["excluded"] = True
            r["exclude_reason"] = "corrupt image"
            continue
        if not info:
            continue
        w, h = info["width"], info["height"]
        r["width"], r["height"], r["mode"], r["channels"] = w, h, info["mode"], info["channels"]
        r["pixel_hash"], r["phash"] = info["pixel_hash"], info["phash"]
        if "rgb_channels_identical" in info:
            r["rgb_channels_identical"] = info["rgb_channels_identical"]
        xs = r.get("xml_size")
        if xs and xs[0] and xs[1] and (xs[0] != w or xs[1] != h):
            problems["xml_size_mismatch"].append({"image_id": r["image_id"], "xml": xs[:2], "image": [w, h]})
        # alternative copies must be pixel identical to be collapsed
        for a in r.get("_alt", []):
            if a["pixel_hash"] != info["pixel_hash"]:
                problems["same_stem_different_pixels"].append({"image_id": r["image_id"], "alt": a["file"]})
                r["excluded"] = True
                r["exclude_reason"] = "several different images share the stem of one XML (ambiguous pairing)"
        # box checks (true annotations)
        if r.get("box_source") == "annotation" and r["boxes"]:
            keep_b, keep_l = [], []
            for b, l in zip(r["boxes"], r["labels"]):
                x1, y1, x2, y2 = b
                if not (x2 > x1 and y2 > y1):
                    problems["degenerate_box"].append({"image_id": r["image_id"], "box": b})
                    continue
                if x1 < -0.5 or y1 < -0.5 or x2 > w + 0.5 or y2 > h + 0.5:
                    problems["box_out_of_bounds"].append({"image_id": r["image_id"], "box": b, "size": [w, h]})
                    b = [max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)]
                    if not (b[2] > b[0] and b[3] > b[1]):
                        continue
                keep_b.append(b)
                keep_l.append(l)
            r["boxes"], r["labels"] = keep_b, keep_l
            # exact duplicate boxes
            seen = Counter((tuple(round(v, 2) for v in b), l) for b, l in zip(r["boxes"], r["labels"]))
            for k, c in seen.items():
                if c > 1:
                    problems["duplicate_box"].append({"image_id": r["image_id"], "box": list(k[0]), "count": c})
        # masks
        m = info.get("mask")
        if r.get("mask_file"):
            if m is None or m.get("corrupt"):
                problems["corrupt_mask"].append({"image_id": r["image_id"]})
                r["excluded"] = True
                r["exclude_reason"] = "corrupt mask"
                continue
            if not m["size_matches_image"]:
                problems["mask_size_mismatch"].append({"image_id": r["image_id"], "mask": m["shape"], "image": [h, w]})
                r["excluded"] = True
                r["exclude_reason"] = "mask size differs from image"
            if not m["binary_0_255"]:
                problems["mask_soft_values"].append({"image_id": r["image_id"], "n_soft_pixels": m["n_soft_pixels"]})
            r["mask_defect_pixels"] = m["defect_pixels"]
            if r["is_normal"] and m["defect_pixels"] > 0:
                problems["normal_image_with_mask_pixels"].append({"image_id": r["image_id"],
                                                                  "pixels": m["defect_pixels"]})
                r["excluded"] = True
                r["exclude_reason"] = "normal image has defect pixels"
            if not r["is_normal"] and r.get("mask_class") is not None:
                if m["defect_pixels"] == 0:
                    problems["defect_image_with_empty_mask"].append({"image_id": r["image_id"]})
                    if mcfg.get("empty_defect_mask_policy", "exclude") == "exclude":
                        r["excluded"] = True
                        r["exclude_reason"] = "defect-class image with empty mask"
                r["boxes"] = m["derived_boxes"]
                r["labels"] = [r["mask_class"]] * len(m["derived_boxes"])
                r["box_source"] = "derived_from_mask"
                if m["n_components_small"]:
                    problems["small_mask_components_dropped_from_derived_boxes"].append(
                        {"image_id": r["image_id"], "n": m["n_components_small"]})
                if m["defect_pixels"] > 0 and not m["derived_boxes"]:
                    # would silently become a negative detection sample -> keep for segmentation only
                    problems["defect_image_without_derived_box"].append({"image_id": r["image_id"]})
                    r["exclude_from_detection"] = True

    # ---------------------------------------------------------------- duplicates
    active = [r for r in records if r.get("pixel_hash")]
    by_bytes, by_pix = defaultdict(list), defaultdict(list)
    for r in active:
        by_bytes[r["sha256"]].append(r["image_id"])
        by_pix[r["pixel_hash"]].append(r["image_id"])
    exact_bytes = [v for v in by_bytes.values() if len(v) > 1]
    exact_pixels = [v for v in by_pix.values() if len(v) > 1]
    nd_cfg = cfg.get("near_duplicates", {}) or {}
    max_d = int(nd_cfg.get("max_distance", 4))
    ph = [int(r["phash"], 16) for r in active]
    pairs = near_duplicate_pairs(ph, max_d)
    near = []
    for i, j, d in pairs:
        a, b = active[i], active[j]
        if a["pixel_hash"] == b["pixel_hash"]:
            continue
        near.append({"a": a["image_id"], "b": b["image_id"], "hamming": d,
                     "same_class": sorted(set(a["labels"])) == sorted(set(b["labels"])) and a["is_normal"] == b["is_normal"]})

    # ---------------------------------------------------------------- statistics
    valid = [r for r in records if not r["excluded"]]
    img_per_class, obj_per_class = Counter(), Counter()
    for r in valid:
        img_classes = set(r["labels"])
        if r.get("mask_class") is not None:
            img_classes.add(r["mask_class"])
        for c in img_classes:
            img_per_class[classes[c]] += 1
        for c in r["labels"]:
            obj_per_class[classes[c]] += 1
    resolutions = Counter(f"{r['width']}x{r['height']}" for r in valid if "width" in r)
    widths = [r["width"] for r in valid if "width" in r]
    heights = [r["height"] for r in valid if "height" in r]
    box_w = [b[2] - b[0] for r in valid for b in r["boxes"]]
    box_h = [b[3] - b[1] for r in valid for b in r["boxes"]]
    rel_area = [((b[2] - b[0]) * (b[3] - b[1])) / (r["width"] * r["height"]) for r in valid for b in r["boxes"]]
    n_multi = sum(1 for r in valid if len(set(r["labels"])) > 1)
    orig_split = Counter(r.get("original_split") or "none" for r in records)
    orig_split_valid = Counter(r.get("original_split") or "none" for r in valid)
    issue_counts = Counter(i["type"] for r in records for i in r.get("issues", []))
    excl = Counter(r["exclude_reason"] for r in records if r["excluded"])
    groups = Counter(r.get("group_key") for r in valid if r.get("group_key"))

    def q(v):
        if not v:
            return None
        v = np.asarray(v, dtype=float)
        return {"min": float(v.min()), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)),
                "p95": float(np.percentile(v, 95)), "max": float(v.max()), "mean": float(v.mean())}

    report = {
        "dataset": cfg["name"],
        "raw_root": cfg["raw_root"],
        "audit_performed_on_real_data": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parse_info": parse_info,
        "counts": {
            "image_files_found": len(records) + sum(len(r.get("alt_files", [])) for r in records),
            "images_after_collapsing_identical_copies": len(records),
            "images_valid": len(valid),
            "images_excluded": len(records) - len(valid),
            "classes": len(classes),
            "annotations_valid_images": int(sum(len(r["boxes"]) for r in valid)),
            "annotation_type": {
                "bbox_annotated": sum(1 for r in valid if r.get("box_source") == "annotation" and not r["is_normal"]),
                "bbox_derived_from_mask": sum(1 for r in valid if r.get("box_source") == "derived_from_mask"),
                "with_pixel_mask": sum(1 for r in valid if r.get("mask_file")),
                "normal_images": sum(1 for r in valid if r["is_normal"]),
            },
            "images_with_multiple_classes": n_multi,
        },
        "per_class": {c: {"images": img_per_class.get(c, 0), "objects": obj_per_class.get(c, 0)} for c in classes},
        "resolutions": {"distinct": len(resolutions), "most_common": resolutions.most_common(10),
                        "width": q(widths), "height": q(heights)},
        "channels": dict(Counter(r.get("mode") for r in valid)),
        "rgb_images_with_identical_channels": sum(1 for r in valid if r.get("rgb_channels_identical")),
        "rgb_images_with_different_channels": sum(1 for r in valid if r.get("rgb_channels_identical") is False),
        "boxes": {"width_px": q(box_w), "height_px": q(box_h), "relative_area": q(rel_area),
                  "objects_per_image": dict(sorted(Counter(len(r["boxes"]) for r in valid).items()))},
        "availability": {
            "bounding_boxes": any(r.get("box_source") == "annotation" and r["boxes"] for r in valid),
            "derived_boxes": any(r.get("box_source") == "derived_from_mask" for r in valid),
            "pixel_masks": any(r.get("mask_file") for r in valid),
            "normal_samples": any(r["is_normal"] for r in valid),
        },
        "problems": {k: {"count": len(v), "examples": v[:20]} for k, v in problems.items()},
        "issues_by_type": dict(issue_counts),
        "excluded_by_reason": dict(excl),
        "duplicates": {
            "exact_byte_duplicate_groups": exact_bytes,
            "exact_pixel_duplicate_groups": exact_pixels,
            "collapsed_identical_copies": [{"image_id": r["image_id"], "copies": r["alt_files"]}
                                           for r in records if r.get("alt_files")],
            "near_duplicate_rule": f"64-bit pHash Hamming distance <= {max_d} (pixel-identical pairs excluded)",
            "near_duplicate_pairs": len(near),
            "near_duplicate_pairs_cross_class": sum(1 for n in near if not n["same_class"]),
            "near_duplicate_examples": near[:50],
        },
        "existing_split": {"all_records": dict(orig_split), "valid_records": dict(orig_split_valid),
                           "origin": "folder names in the downloaded archive" if len(orig_split) > 1 or "none" not in orig_split else None},
        "groups": {"method": (cfg.get("group") or {}).get("method", "none"), "n_groups": len(groups),
                   "largest": groups.most_common(5),
                   "size_histogram": dict(sorted(Counter(groups.values()).items()))},
        "elapsed_s": round(time.time() - t0, 1),
    }
    # strip private fields for the record dump
    for r in records:
        r.pop("_img", None)
        r.pop("_alt", None)
    report["_near_pairs_full"] = near
    return {"report": report, "records": records}


def write_audit_outputs(result: dict, cfg: dict) -> Path:
    proc = Path(cfg["processed_root_abs"])
    rep = dict(result["report"])
    near_full = rep.pop("_near_pairs_full", [])
    dump_json(rep, proc / "audit" / "audit_report.json")
    dump_json(near_full, proc / "audit" / "near_duplicate_pairs.json")
    dump_json(result["records"], proc / "audit" / "records_all.json", indent=None)
    (proc / "audit" / "audit_report.md").write_text(render_audit_markdown(rep), encoding="utf-8")
    return proc / "audit"


def render_audit_markdown(rep: dict) -> str:
    c = rep["counts"]
    L = [f"# Data audit - {rep['dataset']}", "",
         f"Performed on real files in `{rep['raw_root']}` at {rep['timestamp']} ({rep['elapsed_s']} s).", "",
         "## Counts", "",
         f"- image files found: {c['image_files_found']}",
         f"- images after collapsing pixel-identical copies: {c['images_after_collapsing_identical_copies']}",
         f"- valid images: {c['images_valid']} (excluded: {c['images_excluded']})",
         f"- classes: {c['classes']}",
         f"- annotations (boxes) on valid images: {c['annotations_valid_images']}",
         f"- images with >1 class: {c['images_with_multiple_classes']}",
         f"- annotation types: {c['annotation_type']}", "",
         "## Per class", "", "| class | images | objects |", "|---|---:|---:|"]
    for k, v in rep["per_class"].items():
        L.append(f"| {k} | {v['images']} | {v['objects']} |")
    r = rep["resolutions"]
    L += ["", "## Resolutions", "", f"- distinct resolutions: {r['distinct']}",
          f"- most common: {r['most_common']}", f"- width: {r['width']}", f"- height: {r['height']}",
          f"- image modes: {rep['channels']}",
          f"- RGB files whose three channels are identical (effectively grayscale): "
          f"{rep['rgb_images_with_identical_channels']}; with different channels: "
          f"{rep['rgb_images_with_different_channels']}", "", "## Boxes", "",
          f"- width px: {rep['boxes']['width_px']}", f"- height px: {rep['boxes']['height_px']}",
          f"- relative area: {rep['boxes']['relative_area']}",
          f"- objects per image: {rep['boxes']['objects_per_image']}", "",
          "## Availability", "", f"{rep['availability']}", "", "## Problems", ""]
    if not rep["problems"]:
        L.append("none")
    for k, v in rep["problems"].items():
        L.append(f"- **{k}**: {v['count']} (examples: {v['examples'][:3]})")
    L += ["", f"Issues by type: {rep['issues_by_type']}", "", f"Excluded by reason: {rep['excluded_by_reason']}", "",
          "## Duplicates", "",
          f"- exact byte-duplicate groups: {len(rep['duplicates']['exact_byte_duplicate_groups'])}",
          f"- exact pixel-duplicate groups: {len(rep['duplicates']['exact_pixel_duplicate_groups'])}",
          f"- collapsed identical copies (same stem, several folders): {len(rep['duplicates']['collapsed_identical_copies'])}",
          f"- near-duplicate rule: {rep['duplicates']['near_duplicate_rule']}",
          f"- suspicious similar pairs: {rep['duplicates']['near_duplicate_pairs']} "
          f"(cross-class: {rep['duplicates']['near_duplicate_pairs_cross_class']})", "",
          "## Existing split", "", f"{rep['existing_split']}", "", "## Source groups", "", f"{rep['groups']}", ""]
    return "\n".join(L)
