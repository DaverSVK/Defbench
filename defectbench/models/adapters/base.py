"""Adapter interface shared by in-process adapters and subprocess runners.

An adapter is constructed from a resolved ``run_cfg`` (see runs.create_run) and must
implement ``train`` and ``predict``. Predictions are ALWAYS written in the unified
formats of ``defectbench.formats`` (original-image coordinates, canonical labels).
"""
from __future__ import annotations

import logging
from pathlib import Path

from ...data.split import load_manifest, load_split, split_records

log = logging.getLogger(__name__)


class BaseAdapter:
    task: str = "bbox"

    def __init__(self, run_cfg: dict):
        self.run_cfg = run_cfg
        self.ds = run_cfg["dataset_cfg"]
        self.mc = run_cfg["model_cfg"]
        self.run_dir = Path(run_cfg["run_dir"])
        self.classes = self.ds["classes"]
        self.num_classes = len(self.classes)
        self.raw_root = Path(self.ds["raw_root_abs"])
        self._manifest = None
        self._split = None

    # ----------------------------------------------------------- data access
    @property
    def manifest(self):
        if self._manifest is None:
            self._manifest = load_manifest(self.ds)
        return self._manifest

    @property
    def split(self):
        if self._split is None:
            self._split = load_split(self.ds, self.run_cfg.get("split_name", "default"))
        return self._split

    def records(self, which: str) -> list[dict]:
        task = "bbox" if self.task in ("bbox", "instance_seg") else None
        recs = split_records(self.manifest, self.split, which, task=task)
        sm = self.mc.get("smoke")
        if sm:
            n = sm.get(f"n_{which}", 4)
            # deterministic small subset with defects AND (if present) normal images
            pos = [r for r in recs if r["labels"] or r.get("mask_class") is not None]
            neg = [r for r in recs if not (r["labels"] or r.get("mask_class") is not None)]
            recs = pos[: max(1, n - min(len(neg), n // 4))] + neg[: n // 4]
        return recs

    def pred_path(self, which: str) -> Path:
        return self.run_dir / "predictions" / (f"{which}_det.json" if self.task != "seg" else f"{which}_seg")

    # ----------------------------------------------------------- to implement
    def train(self, resume: str | None = None) -> dict:
        raise NotImplementedError

    def predict(self, which: str, checkpoint: str | None = None) -> Path:
        raise NotImplementedError

    def benchmark_model(self, checkpoint: str | None = None):
        """Return (model_fn, preprocess_fn, postprocess_fn, sample_records) for latency measurement."""
        raise NotImplementedError

    def count_parameters(self) -> dict | None:
        return None
