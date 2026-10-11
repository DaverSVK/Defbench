"""run-matrix and smoke-test orchestration."""
from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

from .config import load_dataset_config, load_model_config
from .models.registry import load_registry, task_entry
from .pipeline import environments, env_python, execute
from .runs import create_run
from .utils.io import PROJECT_ROOT, dump_json, load_json

log = logging.getLogger(__name__)


def env_available(env_name: str) -> tuple[bool, str]:
    e = environments()[env_name]
    if e["type"] == "venv":
        py = env_python(env_name)
        return py.exists(), (f"{py} missing" if not py.exists() else "")
    import shutil
    import subprocess
    if not shutil.which("docker"):
        return False, "docker not available"
    r = subprocess.run(["docker", "image", "inspect", e["image"]], capture_output=True)
    return r.returncode == 0, ("" if r.returncode == 0 else f"docker image {e['image']} not built")


class _gpu_lock:
    """Cross-process GPU share for parallel run-matrix processes (one per dataset): models marked
    ``heavy`` take the GPU exclusively, the others share it. No-op where fcntl is unavailable (Windows)."""

    def __init__(self, exclusive: bool):
        self.exclusive, self.f = exclusive, None

    def __enter__(self):
        try:
            import fcntl
        except ImportError:
            return self
        p = PROJECT_ROOT / "reports" / ".gpu.lock"
        p.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(p, "a+")
        t0 = time.time()
        fcntl.flock(self.f, fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH)
        if time.time() - t0 > 1:
            log.info("waited %.0f s for the GPU (%s)", time.time() - t0, "exclusive" if self.exclusive else "shared")
        return self

    def __exit__(self, *exc):
        if self.f:
            self.f.close()  # releases the flock
        return False


def smoke_test(dataset: str = "neu_det", models: list[str] | None = None, device: str = "cuda",
               profile: str = "gpu8gb") -> bool:
    """Tiny end-to-end check per model: data loading, forward, backward, checkpoint save,
    inference (unified predictions validated) and evaluation. Results are NOT experimental
    results; they are written to reports/smoke/."""
    from .evaluation.run_eval import evaluate_run
    reg = load_registry()
    ds = load_dataset_config(dataset)
    if models is None:
        models = [n for n, m in reg.items() if m["environment"] == "core" and m.get("status") != "unsupported"]
    results = []
    all_ok = True
    for name in models:
        m = reg[name]
        for task in m["tasks"]:
            e = task_entry(m, ds, task)
            row = {"model": name, "dataset": dataset, "task": task, "environment": m["environment"]}
            if e["status"] != "run":
                row.update(status="skipped", reason=f"{e['status']}: {e.get('reason', '')}")
                results.append(row)
                continue
            ok_env, why = env_available(m["environment"])
            if not ok_env:
                row.update(status="skipped", reason=f"environment not installed: {why}")
                results.append(row)
                continue
            t0 = time.time()
            checks = {}
            try:
                rc = create_run(dataset, name, task, 0, profile=profile, device=device, smoke=True)
                rd = Path(rc["run_dir"])
                execute("train", rd)
                checks["forward_backward"] = True
                ck = [p for p in (rd / "checkpoints").glob("*") if p.is_file()] if (rd / "checkpoints").exists() else []
                checks["checkpoint_saved"] = bool(ck) or bool(list(rd.rglob("*.pt*")) or list(rd.rglob("*.pth")))
                execute("predict", rd)
                checks["inference"] = (rd / "predictions").exists()
                res = evaluate_run(rd)
                checks["evaluation"] = (rd / "eval" / "metrics.json").exists()
                row.update(status="passed" if all(checks.values()) else "failed", checks=checks,
                           seconds=round(time.time() - t0, 1), run_dir=str(rd))
            except Exception as ex:  # noqa: BLE001 - report every failure
                row.update(status="failed", checks=checks, error=f"{type(ex).__name__}: {ex}",
                           traceback=traceback.format_exc()[-3000:], seconds=round(time.time() - t0, 1))
            all_ok &= row["status"] != "failed"
            log.info("smoke %s/%s/%s: %s", name, dataset, task, row["status"])
            results.append(row)
    out = PROJECT_ROOT / "reports" / "smoke" / f"smoke_{dataset}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    dump_json({"note": "smoke tests on tiny subsets - NOT experimental results", "device": device,
               "results": results}, out)
    for r in results:
        print(f"{r['status']:8s} {r['model']:32s} {r['task']:12s} {r.get('reason', r.get('error', ''))}")
    print(f"smoke report: {out}")
    return all_ok


def run_matrix(datasets, models=None, seeds=(17, 42, 73), profile="default", device="cuda", dry_run=False,
               overrides=None, skip_existing=True):
    from .evaluation.run_eval import evaluate_run
    reg = load_registry()
    plan = []
    for dsn in datasets:
        ds = load_dataset_config(dsn)
        for name, m in reg.items():
            if models and name not in models:
                continue
            for task in m["tasks"]:
                e = task_entry(m, ds, task)
                for seed in seeds:
                    plan.append((dsn, name, task, seed, e))
    log.info("run-matrix: %d candidate runs", len(plan))
    status = []
    for dsn, name, task, seed, e in plan:
        row = {"dataset": dsn, "model": name, "task": task, "seed": seed, "matrix_status": e["status"]}
        if e["status"] != "run":
            reason = " ".join(str(e.get("reason", e["status"])).split())
            row["action"] = f"skip ({reason[:110] + '...' if len(reason) > 110 else reason})"
            row["reason_full"] = e.get("reason")
            status.append(row)
            continue
        ok_env, why = env_available(reg[name]["environment"])
        if not ok_env:
            row["action"] = f"skip (environment: {why})"
            status.append(row)
            continue
        if dry_run:
            row["action"] = "would run"
            status.append(row)
            continue
        try:
            rc = create_run(dsn, name, task, seed, profile=profile, device=device, overrides=overrides)
            rd = Path(rc["run_dir"])
            if skip_existing and (rd / "eval" / "metrics.json").exists():
                row["action"] = "exists"
                status.append(row)
                continue
            from .pipeline import find_resume_checkpoint
            gpu_lock = _gpu_lock(bool(reg[name].get("heavy")))
            # an interrupted run is resumed from the framework's own full-state checkpoint, never restarted;
            # transient failures (e.g. CUDA/cuDNN execution errors) get up to 2 automatic resume attempts
            with gpu_lock:
                for attempt in range(3):
                    if (rd / "train_summary.json").exists():
                        break
                    try:
                        execute("train", rd, resume=find_resume_checkpoint(rd))
                    except Exception as ex:  # noqa: BLE001
                        if attempt == 2 or "OOM" in str(ex) or "out of memory" in str(ex):  # OOM: never retry blindly
                            raise
                        log.warning("training failed (%s) - resuming, attempt %d", ex, attempt + 1)
                execute("predict", rd)
                evaluate_run(rd)
                execute("benchmark", rd)
            row["action"] = "done"
        except Exception as ex:  # noqa: BLE001
            row["action"] = f"FAILED: {type(ex).__name__}: {ex}"
            log.error("run failed: %s", row)
        status.append(row)
    out = PROJECT_ROOT / "reports" / "run_matrix_status.json"
    dump_json(status, out)
    for r in status:
        print(f"{r['dataset']:14s} {r['model']:32s} {r['task']:12s} seed={r['seed']:<3d} {r['action']}")
    return status
