"""Unified command line interface.

    python -m defectbench.cli <command> [options]

Commands: audit-data, prepare-data, train, evaluate, predict, benchmark, run-matrix,
summarize, smoke-test, task-matrix, list-models.
Common options: --dataset, --model, --task, --seed, --profile, --device, --checkpoint,
--set key=value (model config override, repeatable), --tag.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("defectbench")
DATASETS = ["neu_det", "magnetic_tile", "gc10_det"]


def _common(p, model=True, seed=True):
    p.add_argument("--dataset", required=True, choices=DATASETS)
    if model:
        p.add_argument("--model", required=True)
        p.add_argument("--task", default=None, help="bbox | seg | instance_seg (default: first task of the model)")
    if seed:
        p.add_argument("--seed", type=int, default=42, help="training seed (17, 42, 73 are the study seeds)")
    p.add_argument("--profile", default="default", help="hardware profile: default (12-16 GB) | gpu8gb")
    p.add_argument("--device", default="cuda", help="cuda | cpu")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a model config value, e.g. --set train.epochs=50")
    p.add_argument("--tag", default=None, help="suffix of the run directory")
    p.add_argument("--split-name", default="default")


def _run_cfg(a, smoke=False):
    from .config import load_model_config
    from .runs import create_run
    mc = load_model_config(a.model)
    task = a.task or mc["tasks"][0]
    if task not in mc["tasks"]:
        raise SystemExit(f"model {a.model} supports tasks {mc['tasks']}, not {task}")
    weak = bool(getattr(a, "weak_box_supervision", False))
    e = _check_matrix(a.model, a.dataset, task, allow_weak=weak)
    overrides = list(a.overrides)
    tag = a.tag
    if weak:
        if e["status"] != "optional_weak":
            raise SystemExit("--weak-box-supervision is only valid for segmentation models on box-only datasets")
        overrides.append("weak_box_supervision=true")
        tag = tag or "weakbox"
        log.warning("WEAKLY SUPERVISED experiment: segmentation trained on box-filled pseudo masks (%s)", tag)
    return create_run(a.dataset, a.model, task, a.seed, profile=a.profile, overrides=overrides, tag=tag,
                      split_name=a.split_name, device=a.device, smoke=smoke)


def _check_matrix(model, dataset, task, allow_weak=False):
    from .config import load_dataset_config, load_model_config
    from .models.registry import task_entry
    e = task_entry(load_model_config(model), load_dataset_config(dataset), task, allow_weak=allow_weak)
    if e["status"] == "run":
        if e.get("flag"):
            log.warning("%s on %s (%s): %s", model, dataset, task, e["annotation"])
        return e
    if e["status"] == "optional_weak" and allow_weak:
        return e
    raise SystemExit(f"{model} x {dataset} x {task} is not a valid experiment: {e['status']} - "
                     f"{e.get('reason', '')}")


def cmd_audit(a):
    from .config import load_dataset_config
    from .data.audit import run_audit, write_audit_outputs
    for ds in ([a.dataset] if a.dataset != "all" else DATASETS):
        cfg = load_dataset_config(ds)
        if not Path(cfg["raw_root_abs"]).exists():
            print(f"[{ds}] raw data NOT found at {cfg['raw_root_abs']} - the real audit has NOT been performed. "
                  f"Download with `prepare-data --dataset {ds} --download` or extract the Kaggle archive there.")
            continue
        res = run_audit(cfg)
        out = write_audit_outputs(res, cfg)
        print((out / "audit_report.md").read_text(encoding="utf-8"))


def cmd_prepare(a):
    from .config import load_dataset_config
    from .data.download import download_kaggle_dataset
    from .data.prepare import prepare_dataset
    for ds in ([a.dataset] if a.dataset != "all" else DATASETS):
        cfg = load_dataset_config(ds)
        if a.download:
            download_kaggle_dataset(cfg["kaggle_slug"], Path(cfg["raw_root_abs"]), force=a.force_download)
        out = prepare_dataset(cfg, split_name=a.split_name, force_split=a.force_split)
        s = out["split"]
        print(f"[{ds}] manifest {out['manifest']['manifest_hash']} ({out['manifest']['n_images']} images, "
              f"{len(out['manifest']['excluded'])} excluded), split '{a.split_name}' {s['split_hash']} "
              f"ratios {s['verification']['realised_ratios']} verification {s['verification']}")


def cmd_train(a):
    from .pipeline import execute
    rc = _run_cfg(a)
    res = execute("train", Path(rc["run_dir"]), resume=a.resume)
    print(json.dumps(res, indent=1, default=str) if isinstance(res, dict) else f"exit {res}")
    if not a.no_eval:
        execute("predict", Path(rc["run_dir"]), checkpoint=None)
        from .evaluation.run_eval import evaluate_run
        _print_eval(evaluate_run(rc["run_dir"]))
    print(f"run directory: {rc['run_dir']}")


def _run_dir_from_args(a):
    from .runs import run_dir_for
    if getattr(a, "run_dir", None):
        return Path(a.run_dir)
    from .config import load_model_config
    task = a.task or load_model_config(a.model)["tasks"][0]
    return run_dir_for(a.dataset, a.model, task, a.seed, a.tag)


def cmd_predict(a):
    from .pipeline import execute
    rd = _run_dir_from_args(a)
    print(execute("predict", rd, checkpoint=a.checkpoint, splits=a.splits))


def cmd_evaluate(a):
    from .evaluation.run_eval import evaluate_run
    rd = _run_dir_from_args(a)
    if a.predict_first:
        from .pipeline import execute
        execute("predict", rd, checkpoint=a.checkpoint)
    _print_eval(evaluate_run(rd))


def _print_eval(res):
    if "test" in res and isinstance(res["test"], dict) and "coco" in res["test"]:
        print("COCO (test):", json.dumps(res["test"]["coco"]))
        wp = res["working_point"]
        print(f"working point thr={wp['threshold']:.4f}: P={wp['precision']} R={wp['recall']} F1={wp['f1']}")
    elif "test" in res:
        print("segmentation (test, pooled):", json.dumps(res["test"]["pooled"]))
    il = res.get("image_level", {})
    print("image-level AUROC/AUPRC:", il.get("auroc"), il.get("auprc"), il.get("reason", ""))
    pl = res.get("pixel_level", {})
    print("pixel-level AUROC/AUPRC:", pl.get("auroc"), pl.get("auprc"), pl.get("reason", ""))
    print("AUPRO:", res.get("aupro", {}).get("aupro"), res.get("aupro", {}).get("reason", ""))


def cmd_benchmark(a):
    from .pipeline import execute
    rd = _run_dir_from_args(a)
    print(execute("benchmark", rd, checkpoint=a.checkpoint, n_images=a.n_images, warmup=a.warmup))


def cmd_matrix(a):
    from .models.registry import build_task_matrix, matrix_markdown
    rows = build_task_matrix(DATASETS)
    print(matrix_markdown(rows))
    if a.output:
        from .utils.io import dump_json
        dump_json(rows, a.output)


def cmd_list(a):
    from .models.registry import load_registry
    for n, m in load_registry().items():
        print(f"{n:32s} {m.get('status', ''):12s} env={m['environment']:12s} tasks={m['tasks']}  {m.get('display_name', '')}")


def cmd_run_matrix(a):
    from .orchestrate import run_matrix
    run_matrix(datasets=a.datasets.split(","), models=a.models.split(",") if a.models else None,
               seeds=[int(s) for s in a.seeds.split(",")], profile=a.profile, device=a.device,
               dry_run=a.dry_run, overrides=a.overrides, skip_existing=not a.rerun)


def cmd_summarize(a):
    from .reporting.aggregate import summarize
    out = summarize(Path(a.runs_root), Path(a.output), include_smoke=a.include_smoke)
    print(f"summary written to {out}")


def cmd_smoke(a):
    from .orchestrate import smoke_test
    ok = smoke_test(dataset=a.dataset, models=a.models.split(",") if a.models else None, device=a.device,
                    profile=a.profile)
    sys.exit(0 if ok else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="defectbench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("audit-data", help="audit raw data (counts, resolutions, invalid boxes/masks, duplicates)")
    p.add_argument("--dataset", required=True, choices=DATASETS + ["all"])
    p.set_defaults(fn=cmd_audit)

    p = sub.add_parser("prepare-data", help="(download) + audit + manifest + leakage-free split")
    p.add_argument("--dataset", required=True, choices=DATASETS + ["all"])
    p.add_argument("--download", action="store_true", help="download from Kaggle via kagglehub")
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--split-name", default="default")
    p.add_argument("--force-split", action="store_true", help="overwrite an existing, different split")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("train", help="train one model on one dataset (then predict + evaluate)")
    _common(p)
    p.add_argument("--resume", default=None, help="'auto' (last full-state checkpoint of the framework) or a checkpoint path")
    p.add_argument("--weak-box-supervision", action="store_true",
                   help="EXPLICIT weakly-supervised experiment: train a segmenter on box-filled pseudo masks of a "
                        "box-only dataset (evaluated only as segmentation-derived detection)")
    p.add_argument("--no-eval", action="store_true")
    p.set_defaults(fn=cmd_train)

    for name, fn, h in (("predict", cmd_predict, "export unified predictions for val/test"),
                        ("evaluate", cmd_evaluate, "compute all metrics from saved predictions"),
                        ("benchmark", cmd_benchmark, "latency (batch 1, warmup, CUDA sync)")):
        p = sub.add_parser(name, help=h)
        _common(p)
        p.add_argument("--run-dir", default=None)
        p.add_argument("--checkpoint", default=None)
        if name == "predict":
            p.add_argument("--splits", default="val,test")
        if name == "evaluate":
            p.add_argument("--predict-first", action="store_true")
        if name == "benchmark":
            p.add_argument("--n-images", type=int, default=200)
            p.add_argument("--warmup", type=int, default=20)
        p.set_defaults(fn=fn)

    p = sub.add_parser("task-matrix", help="print the model x dataset x task matrix")
    p.add_argument("--output", default=None)
    p.set_defaults(fn=cmd_matrix)

    p = sub.add_parser("list-models", help="list the model registry")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("run-matrix", help="train+predict+evaluate all valid (model, dataset, task, seed)")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--models", default=None, help="comma list (default: all runnable)")
    p.add_argument("--seeds", default="17,42,73")
    p.add_argument("--profile", default="default")
    p.add_argument("--device", default="cuda")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rerun", action="store_true", help="re-run finished runs")
    p.set_defaults(fn=cmd_run_matrix)

    p = sub.add_parser("summarize", help="aggregate runs -> results.csv/json, Markdown report, plots")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--output", default="reports/generated")
    p.add_argument("--include-smoke", action="store_true")
    p.set_defaults(fn=cmd_summarize)

    p = sub.add_parser("smoke-test", help="tiny end-to-end check: data, forward, backward, ckpt, predict, eval")
    p.add_argument("--dataset", default="neu_det", choices=DATASETS)
    p.add_argument("--models", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--profile", default="gpu8gb")
    p.set_defaults(fn=cmd_smoke)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    a.fn(a)


if __name__ == "__main__":
    main()
