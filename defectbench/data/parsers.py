"""Dataset parsers producing canonical image records.

No fixed nested directory names are assumed: images, VOC XML files and masks are
discovered recursively under ``raw_root`` and paired by file stem / folder rules
that are configured in ``configs/datasets/*.yaml``.

A canonical *record* (plain dict, JSON serialisable)::

    image_id         unique id "<dataset>/<stem>" (or relative path if stems collide)
    file             image path relative to raw_root (posix)
    alt_files        other pixel-identical copies of the same image (collapsed)
    boxes            [[x1,y1,x2,y2], ...] continuous pixel-edge coords (see geometry.py)
    labels           [int, ...] canonical class index
    box_source       "annotation" | "derived_from_mask" | None
    mask_file        relative path of the pixel mask or None
    mask_source      "annotation" | None
    mask_class       canonical class index of all mask pixels (MT: from folder) or None
    is_normal        True only for images that are defect-free BY ANNOTATION (e.g. MT_Free)
    original_split   split name found in the path (train/validation/test) or None
    group_key        source / sequence identifier used for leakage-free splitting
    annotation_file  relative path of the XML (VOC) or None
    issues           list of {"type":..., "detail":...}
    excluded         bool, exclude_reason str

Parsers only read annotation files; image decoding / hashing is done by the audit.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from .geometry import voc_to_xyxy

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_WORDS = {"train": "train", "training": "train", "val": "validation", "valid": "validation",
               "validation": "validation", "test": "test", "testing": "test"}


def _issue(rec: dict, typ: str, detail: str = "") -> None:
    rec.setdefault("issues", []).append({"type": typ, "detail": detail})


def _split_from_path(rel: Path) -> str | None:
    for part in rel.parts[:-1]:
        s = SPLIT_WORDS.get(part.lower())
        if s:
            return s
    return None


def _new_record(dataset: str, rel: Path) -> dict:
    return {
        "image_id": None, "file": rel.as_posix(), "alt_files": [], "boxes": [], "labels": [],
        "box_source": None, "mask_file": None, "mask_source": None, "mask_class": None,
        "is_normal": False, "original_split": _split_from_path(rel), "group_key": None,
        "annotation_file": None, "issues": [], "excluded": False, "exclude_reason": None,
    }


def _exclude(rec: dict, reason: str) -> None:
    rec["excluded"] = True
    rec["exclude_reason"] = reason if not rec.get("exclude_reason") else rec["exclude_reason"] + "; " + reason


def discover_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _assign_image_ids(dataset: str, records: list[dict]) -> None:
    stems = defaultdict(int)
    for r in records:
        stems[Path(r["file"]).stem] += 1
    for r in records:
        stem = Path(r["file"]).stem
        if stems[stem] == 1:
            r["image_id"] = f"{dataset}/{stem}"
        else:
            r["image_id"] = f"{dataset}/{Path(r['file']).with_suffix('').as_posix()}"


# ----------------------------------------------------------------------------- VOC
def parse_voc_xml(path: Path) -> dict:
    root = ET.parse(path).getroot()
    size = root.find("size")
    out = {
        "filename": (root.findtext("filename") or "").strip(),
        "folder": (root.findtext("folder") or "").strip(),
        "width": int(float(size.findtext("width"))) if size is not None and size.findtext("width") else None,
        "height": int(float(size.findtext("height"))) if size is not None and size.findtext("height") else None,
        "depth": int(float(size.findtext("depth"))) if size is not None and size.findtext("depth") else None,
        "objects": [],
    }
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        o = {"name": (obj.findtext("name") or "").strip(), "difficult": int(obj.findtext("difficult") or 0),
             "bbox_voc": None}
        if bb is not None:
            try:
                o["bbox_voc"] = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            except (TypeError, ValueError):
                o["bbox_voc"] = None
        out["objects"].append(o)
    return out


def parse_voc_dataset(cfg: dict, raw_root: Path) -> tuple[list[dict], dict]:
    """Generic Pascal-VOC parser (NEU-DET, GC10-DET).

    Pairing: XML <-> image by file stem (the XML <filename> is checked for
    consistency). If several image files share a stem they are compared later by
    the audit (pixel hash); the copy in the folder named by XML <folder> is kept
    as primary and the others are recorded in ``alt_files``.
    """
    dataset = cfg["name"]
    classes = cfg["classes"]
    cls_index = {c: i for i, c in enumerate(classes)}
    aliases = {k: v for k, v in (cfg.get("label_aliases") or {}).items()}
    images = discover_images(raw_root)
    xmls = sorted(raw_root.rglob("*.xml"))
    info = {"n_image_files": len(images), "n_xml_files": len(xmls), "label_names_raw": defaultdict(int),
            "unmapped_labels": defaultdict(int), "detection_annotations_found": len(xmls) > 0}
    if not xmls:
        raise RuntimeError(
            f"{dataset}: no VOC XML files found under {raw_root}. This looks like a classification-only "
            "version of the dataset; bounding-box annotations are required for detection.")

    by_stem: dict[str, list[Path]] = defaultdict(list)
    for p in images:
        by_stem[p.stem].append(p)
    xml_by_stem: dict[str, list[Path]] = defaultdict(list)
    for x in xmls:
        xml_by_stem[x.stem].append(x)

    records = []
    group_re = re.compile(cfg["group"]["regex"]) if cfg.get("group", {}).get("method") == "regex" else None
    for stem, paths in sorted(by_stem.items()):
        xml_list = xml_by_stem.get(stem, [])
        ann = parse_voc_xml(xml_list[0]) if xml_list else None
        # choose primary copy
        primary = paths[0]
        if len(paths) > 1 and ann and ann["folder"]:
            for p in paths:
                if p.parent.name == ann["folder"]:
                    primary = p
                    break
        rel = primary.relative_to(raw_root)
        rec = _new_record(dataset, rel)
        rec["alt_files"] = [p.relative_to(raw_root).as_posix() for p in paths if p != primary]
        if len(paths) > 1:
            _issue(rec, "same_stem_multiple_files", ", ".join(p.relative_to(raw_root).as_posix() for p in paths))
        if len(xml_list) > 1:
            _issue(rec, "multiple_xml_for_image", ", ".join(x.relative_to(raw_root).as_posix() for x in xml_list))
        if group_re:
            m = group_re.search(stem)
            rec["group_key"] = m.group("group") if m else None
            if not m:
                _issue(rec, "group_regex_no_match", stem)
        if ann is None:
            _issue(rec, "missing_annotation", "no XML with the same stem")
            if cfg.get("unannotated_image_policy", "exclude") == "exclude":
                _exclude(rec, "missing annotation file")
            records.append(rec)
            continue
        rec["annotation_file"] = xml_list[0].relative_to(raw_root).as_posix()
        rec["xml_size"] = [ann["width"], ann["height"], ann["depth"]]
        if ann["filename"] and Path(ann["filename"]).stem != stem:
            _issue(rec, "xml_filename_mismatch", ann["filename"])
        if ann["folder"] and ann["folder"] != primary.parent.name:
            _issue(rec, "xml_folder_differs_from_image_folder", f"xml={ann['folder']} image={primary.parent.name}")
        invalid_label = False
        for o in ann["objects"]:
            info["label_names_raw"][o["name"]] += 1
            name = aliases.get(o["name"], o["name"])
            if name not in cls_index:
                info["unmapped_labels"][o["name"]] += 1
                _issue(rec, "unknown_label", o["name"])
                invalid_label = True
                continue
            if o["bbox_voc"] is None:
                _issue(rec, "missing_bndbox", o["name"])
                invalid_label = True
                continue
            rec["boxes"].append(voc_to_xyxy(*o["bbox_voc"]))
            rec["labels"].append(cls_index[name])
            if o["difficult"]:
                _issue(rec, "difficult_flag", o["name"])
        rec["box_source"] = "annotation"
        if invalid_label and cfg.get("invalid_label_policy", "exclude_image") == "exclude_image":
            _exclude(rec, "annotation contains unknown label or missing box (a defect would stay unlabeled)")
        if not ann["objects"]:
            _issue(rec, "empty_annotation", "XML without objects")
            pol = cfg.get("empty_annotation_policy", "exclude")
            if pol == "exclude":
                _exclude(rec, "empty XML in a defect dataset - not trusted as a normal sample")
            elif pol == "normal":
                rec["is_normal"] = True
        records.append(rec)

    orphan_xml = sorted(set(xml_by_stem) - set(by_stem))
    info["xml_without_image"] = orphan_xml
    _assign_image_ids(dataset, records)
    info["label_names_raw"] = dict(info["label_names_raw"])
    info["unmapped_labels"] = dict(info["unmapped_labels"])
    return records, info


# ----------------------------------------------------------------------------- mask folders
def parse_mask_folder_dataset(cfg: dict, raw_root: Path) -> tuple[list[dict], dict]:
    """Datasets where every image has a same-stem mask next to it and the class is
    given by an ancestor folder (Magnetic Tile: ``MT_<Class>/Imgs/<stem>.jpg + .png``).

    Masks are *semantic* (one binary mask per image); the pixel class is the folder
    class. Images in ``normal_class_dirs`` are defect-free by annotation.
    """
    dataset = cfg["name"]
    classes = cfg["classes"]
    cls_index = {c: i for i, c in enumerate(classes)}
    cls_re = re.compile(cfg["class_dir_regex"])
    normal_dirs = set(cfg.get("normal_class_dirs", []))
    mcfg = cfg["mask"]
    img_exts = {e.lower() for e in cfg.get("image_exts", [".jpg", ".jpeg", ".bmp"])}
    mask_suffix = mcfg.get("suffix", ".png")

    files = [p for p in raw_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    stems_by_dir = defaultdict(dict)
    for p in files:
        stems_by_dir[p.parent][p.suffix.lower()] = stems_by_dir[p.parent].get(p.suffix.lower(), set()) | {p.stem}
    info = {"n_files": len(files), "class_dirs": defaultdict(int), "images_without_mask": [],
            "masks_without_image": []}
    records = []
    for p in sorted(files):
        if p.suffix.lower() not in img_exts:
            continue
        # class from nearest ancestor dir matching regex
        cls_name = None
        for anc in p.relative_to(raw_root).parents:
            m = cls_re.match(anc.name) if anc.name else None
            if m:
                cls_name = m.group(1)
                break
        rel = p.relative_to(raw_root)
        rec = _new_record(dataset, rel)
        if cls_name is None:
            _issue(rec, "no_class_folder", rel.as_posix())
            _exclude(rec, "class folder not found")
            records.append(rec)
            continue
        info["class_dirs"][cls_name] += 1
        mask_path = p.with_suffix(mask_suffix)
        if mask_path.exists():
            rec["mask_file"] = mask_path.relative_to(raw_root).as_posix()
            rec["mask_source"] = "annotation"
        else:
            info["images_without_mask"].append(rel.as_posix())
            _issue(rec, "missing_mask")
            _exclude(rec, "mask missing")
        if cls_name in normal_dirs:
            rec["is_normal"] = True
            rec["box_source"] = "annotation"  # empty box list is a real (negative) annotation
        elif cls_name in cls_index:
            rec["mask_class"] = cls_index[cls_name]
            rec["box_source"] = "derived_from_mask"  # boxes are filled in by the audit (needs pixels)
        else:
            _issue(rec, "unknown_class_folder", cls_name)
            _exclude(rec, f"class folder {cls_name} not in config")
        records.append(rec)
    # masks without images
    for d, by_ext in stems_by_dir.items():
        mstems = by_ext.get(mask_suffix, set())
        istems = set().union(*[v for k, v in by_ext.items() if k in img_exts]) if by_ext else set()
        for s in sorted(mstems - istems):
            info["masks_without_image"].append((d / (s + mask_suffix)).relative_to(raw_root).as_posix())
    _assign_image_ids(dataset, records)
    info["class_dirs"] = dict(info["class_dirs"])
    return records, info


def assign_sequence_groups(records: list[dict], gcfg: dict) -> None:
    """Group key from a numeric acquisition counter (Magnetic Tile).

    Images are sorted by the counter ``key`` (optionally per class folder) and a new
    group starts whenever the gap to the previous image exceeds ``max_gap``. For MT
    this merges the six captures (exp1..exp6) of the same tile ROI - see README
    "Data leakage" for the evidence and the chosen threshold.
    """
    rx = re.compile(gcfg["regex"])
    key = gcfg.get("key", "num")
    scope = gcfg.get("scope", "global")
    max_gap = float(gcfg.get("max_gap", 300))
    buckets = defaultdict(list)
    for r in records:
        m = rx.search(Path(r["file"]).stem)
        if not m:
            _issue(r, "group_regex_no_match", r["file"])
            r["group_key"] = "nomatch:" + r["file"]
            continue
        b = "all" if scope == "global" else Path(r["file"]).parts[0]
        buckets[b].append((float(m.group(key)), r))
    for b, items in buckets.items():
        items.sort(key=lambda t: t[0])
        gid, prev = 0, None
        for v, r in items:
            if prev is not None and v - prev > max_gap:
                gid += 1
            r["group_key"] = f"{b}:seq{gid:04d}"
            prev = v


PARSERS = {"voc": parse_voc_dataset, "mask_folders": parse_mask_folder_dataset}


def parse_dataset(cfg: dict, raw_root: Path) -> tuple[list[dict], dict]:
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(
            f"Raw data for {cfg['name']} not found at {raw_root}. Download with "
            f"`python -m defectbench.cli prepare-data --dataset {cfg['name']} --download` or extract the "
            f"Kaggle archive there manually.")
    fmt = cfg["format"]
    if fmt not in PARSERS:
        raise ValueError(f"unknown dataset format {fmt}")
    records, info = PARSERS[fmt](cfg, raw_root)
    g = cfg.get("group", {}) or {}
    if g.get("method") == "sequence_gap":
        assign_sequence_groups(records, g)
    return records, info
