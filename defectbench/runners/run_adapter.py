"""Entry point executed INSIDE a framework-specific environment (venv or container).

    python -m defectbench.runners.run_adapter train   --run-dir runs/... [--resume ckpt]
    python -m defectbench.runners.run_adapter predict --run-dir runs/... [--checkpoint ckpt] [--splits val,test]
    python -m defectbench.runners.run_adapter benchmark --run-dir runs/... [--checkpoint ckpt]

Only the run directory is shared with the orchestrator.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["train", "predict", "benchmark"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--resume")
    ap.add_argument("--checkpoint")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--n-images", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from defectbench.pipeline import run_in_process
    from defectbench.training.engine import OOMAbort
    kw = {}
    if a.action == "train":
        kw["resume"] = a.resume
    elif a.action == "predict":
        kw.update(checkpoint=a.checkpoint, splits=a.splits)
    else:
        kw["checkpoint"] = a.checkpoint
        if a.n_images:
            kw["n_images"] = a.n_images
        if a.warmup is not None:
            kw["warmup"] = a.warmup
    try:
        run_in_process(a.action, Path(a.run_dir), **kw)
    except OOMAbort:
        sys.exit(3)


if __name__ == "__main__":
    main()
