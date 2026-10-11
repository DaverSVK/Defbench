"""Latency benchmark (batch size 1).

Protocol (identical for all adapters, run in the model's own environment):
* images: the first ``n_images`` test images (original files, real resolution distribution),
* batch size 1, ``warmup`` untimed iterations first (cuDNN autotuning, lazy init, caches),
* every timed stage is bracketed by ``torch.cuda.synchronize()`` so asynchronous kernels are counted,
* ``model_ms``   = time of ``model_fn`` only (network forward + whatever the framework fuses into its
                   forward call - described per adapter in ``model_scope``),
* ``end2end_ms`` = preprocess (read file + resize/normalise + H2D) + model + postprocess (D2H + conversion
                   to the unified record in original coordinates),
* reported: median and p95 over the timed images, plus mean,
* FPS definitions: ``fps_model = 1000 / median(model_ms)`` and ``fps_end2end = 1000 / median(end2end_ms)``
  (single-stream throughput at batch size 1; NOT a multi-batch throughput),
* precision / backend: PyTorch eager in the precision stated in ``precision`` (fp32 unless the adapter's
  inference config enables AMP); no TensorRT/ONNX export.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .utils.envinfo import gpu_info
from .utils.io import dump_json

log = logging.getLogger(__name__)


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def benchmark_adapter(adapter, run_cfg: dict, checkpoint: str | None = None, n_images: int = 200,
                      warmup: int = 20) -> dict:
    import torch
    model_fn, pre, post, scope = adapter.benchmark_model(checkpoint)
    recs = adapter.records("test")[:n_images]
    if not recs:
        raise RuntimeError("no test images")
    seq = [recs[i % len(recs)] for i in range(warmup)] + recs
    m_ms, e_ms = [], []
    with torch.inference_mode():
        for i, rec in enumerate(seq):
            _sync()
            t0 = time.perf_counter()
            x = pre(rec)
            _sync()
            t1 = time.perf_counter()
            out = model_fn(x)
            _sync()
            t2 = time.perf_counter()
            post(out, rec)
            _sync()
            t3 = time.perf_counter()
            if i >= warmup:
                m_ms.append((t2 - t1) * 1000)
                e_ms.append((t3 - t0) * 1000)
    m, e = np.array(m_ms), np.array(e_ms)
    gi = gpu_info()
    res = {
        "n_images": len(m), "warmup": warmup, "batch_size": 1,
        "model_ms": {"median": float(np.median(m)), "p95": float(np.percentile(m, 95)), "mean": float(m.mean())},
        "end2end_ms": {"median": float(np.median(e)), "p95": float(np.percentile(e, 95)), "mean": float(e.mean())},
        "fps_model": float(1000 / np.median(m)), "fps_end2end": float(1000 / np.median(e)),
        "fps_definition": "1000 / median latency at batch size 1 (single stream)",
        "model_scope": scope,
        "precision": "amp-fp16" if (run_cfg["model_cfg"].get("inference") or {}).get("amp") else "fp32",
        "backend": "PyTorch eager " + torch.__version__,
        "hardware": {"gpu": (gi.get("devices") or [{}])[0].get("name"), "nvidia_smi": gi.get("nvidia_smi"),
                     "cuda": gi.get("torch_cuda")},
        "device": str(getattr(adapter, "device", "cuda")),
    }
    out = Path(run_cfg["run_dir"]) / "benchmark" / "latency.json"
    dump_json(res, out)
    log.info("latency model %.2f ms (p95 %.2f), end2end %.2f ms (p95 %.2f)", res["model_ms"]["median"],
             res["model_ms"]["p95"], res["end2end_ms"]["median"], res["end2end_ms"]["p95"])
    return res
