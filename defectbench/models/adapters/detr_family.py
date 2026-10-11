"""DEIMv2 / D-FINE / RT-DETRv2 adapter (runs inside envs/detr_family).

All three repositories share the RT-DETR "engine" design (YAML ``__include__`` configs,
``YAMLConfig``, ``TASKS['detection']`` solver, ``-t`` tuning from a checkpoint). The
adapter reproduces the repository ``train.py:main`` in-process (same YAMLConfig update
dict, same HGNetv2 handling), with the official config of the chosen variant included
verbatim and only these keys overridden in a generated config:

* dataset paths (our ``coco0`` export), ``num_classes`` = K, ``remap_mscoco_category: False``
  (label = category_id = canonical label),
* ``total_batch_size`` (+ LINEAR lr scaling of every param group, the rule documented in
  the D-FINE README "Customizing batch size"),
* the epoch schedule of the fine-tuning recipe (model config ``schedule_overrides``),
* ``output_dir``.

Added on top of the upstream engine (without modifying it):
* early stopping on val AP (patience in epochs) - implemented by wrapping ``evaluate``;
  the best checkpoint is still written by the upstream solver (best_stg*.pth / best.pth),
* validation loss per epoch (same criterion, model in train mode with BatchNorm frozen,
  target boxes converted to normalised cxcywh like the train pipeline),
* a tuning-load report: the run FAILS if any backbone weight of the pretrained
  checkpoint was not loaded (guards against silently training the backbone from scratch).
"""
from __future__ import annotations

import copy
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from ...data.export import export_coco0
from ...formats import make_det_record, postprocess_det_record, save_det_predictions, validate_det_predictions
from ...utils.io import dump_json, dump_yaml, load_json
from ...weights import WEIGHTS_DIR, fetch, third_party_dir
from .base import BaseAdapter

log = logging.getLogger(__name__)


class EarlyStop(Exception):
    pass


def _lr_scale(opt_cfg: dict, s: float) -> dict:
    o = copy.deepcopy(opt_cfg)
    if "lr" in o:
        o["lr"] = float(o["lr"]) * s
    for g in o.get("params", []) or []:
        if isinstance(g, dict) and "lr" in g:
            g["lr"] = float(g["lr"]) * s
    return o


class DetrFamilyAdapter(BaseAdapter):
    task = "bbox"

    # ------------------------------------------------------------------ repo plumbing
    @property
    def repo(self) -> Path:
        m = self.mc["model"]
        return third_party_dir(m["repo"]) / m.get("repo_subdir", "")

    def _enter_repo(self):
        repo = str(self.repo.resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        os.chdir(repo)  # upstream code uses cwd-relative paths (ckpts/, weight/)
        pkg = self.mc["model"]["package"]
        mods = (importlib.import_module(f"{pkg}.core"), importlib.import_module(f"{pkg}.solver"),
                importlib.import_module(f"{pkg}.misc.dist_utils"), importlib.import_module(f"{pkg}.core.yaml_utils"))
        self._torchvision_v2_shim(pkg)
        return mods

    @staticmethod
    def _torchvision_v2_shim(pkg: str) -> list[str]:
        """torchvision>=0.21 calls ``Transform.transform``/``make_params``; code written for 0.20 only defines
        ``_transform``/``_get_params``. RT-DETRv2 and D-FINE (same code lineage) ship exactly this alias upstream
        (``def transform(self, inpt, params): return self._transform(inpt, params)``); DEIMv2 (pinned to
        torchvision 0.20.1) does not. The identical alias is added at runtime - the repository is not modified."""
        import pkgutil
        from torchvision.transforms import v2 as T
        patched = []
        base = importlib.import_module(f"{pkg}.data.transforms")
        for mi in pkgutil.iter_modules(base.__path__):
            mod = importlib.import_module(f"{pkg}.data.transforms.{mi.name}")
            for obj in vars(mod).values():
                if isinstance(obj, type) and issubclass(obj, T.Transform) and obj.__module__ == mod.__name__:
                    if "_transform" in vars(obj) and "transform" not in vars(obj):
                        obj.transform = lambda self, inpt, params: self._transform(inpt, params)
                        patched.append(f"{obj.__name__}.transform")
                    if "_get_params" in vars(obj) and "make_params" not in vars(obj):
                        obj.make_params = lambda self, flat_inputs: self._get_params(flat_inputs)
                        patched.append(f"{obj.__name__}.make_params")
        if patched:
            log.info("torchvision v2 compatibility alias added for: %s", patched)
        return patched

    def _tuning_checkpoint(self) -> Path:
        p = self.mc["pretrained"]
        if p.get("hf_repo"):
            from huggingface_hub import hf_hub_download
            st = hf_hub_download(p["hf_repo"], p["hf_file"], revision=p.get("hf_revision"))
            out = WEIGHTS_DIR / f"{p['hf_repo'].replace('/', '__')}.pth"
            if not out.exists():
                import torch
                from safetensors.torch import load_file
                torch.save({"model": load_file(st), "source": f"hf://{p['hf_repo']}@{p.get('hf_revision')}"}, out)
            return out
        return fetch(p["url"])

    def _write_config(self, data_dir: Path, yaml_utils) -> Path:
        m, t = self.mc["model"], self.mc["train"]
        base = (self.repo / m["base_config"]).resolve()
        recipe = yaml_utils.load_config(str(base), {})  # merged official recipe (read-only)
        rec_bs = int(recipe["train_dataloader"]["total_batch_size"])
        bs = int(t["total_batch_size"])
        scale = bs / rec_bs
        fw = self.run_dir / "framework"
        cfg = {
            "__include__": [base.as_posix()],
            "output_dir": fw.as_posix(),
            "num_classes": self.num_classes,
            "remap_mscoco_category": False,
            "train_dataloader": {"total_batch_size": bs, "num_workers": int(t.get("num_workers", 2)),
                                 "dataset": {"img_folder": (data_dir / "train").as_posix(),
                                             "ann_file": (data_dir / "train" / "_annotations.coco.json").as_posix()}},
            "val_dataloader": {"total_batch_size": int(t.get("eval_batch_size", bs)),
                               "num_workers": int(t.get("num_workers", 2)),
                               "dataset": {"img_folder": (data_dir / "valid").as_posix(),
                                           "ann_file": (data_dir / "valid" / "_annotations.coco.json").as_posix()}},
            "optimizer": _lr_scale(recipe["optimizer"], scale),
            "checkpoint_freq": 1000,
            "print_freq": 50,
        }
        from ...utils.io import deep_update
        cfg = deep_update(cfg, copy.deepcopy(self.mc.get("schedule_overrides") or {}))
        if t.get("max_iters_per_epoch"):  # smoke test: single short epoch without stage switches
            cfg = deep_update(cfg, self.mc.get("smoke_overrides") or {})
        path = fw / "config.yml"
        dump_yaml(cfg, path)
        dump_json({"recipe_config": base.as_posix(), "recipe_total_batch_size": rec_bs, "total_batch_size": bs,
                   "lr_scale_linear": scale, "recipe_optimizer": recipe["optimizer"],
                   "used_optimizer": cfg["optimizer"]}, fw / "batch_lr_scaling.json")
        return path

    # ------------------------------------------------------------------ train
    def train(self, resume: str | None = None) -> dict:
        import torch
        subset = {w: self.records(w) for w in ("train", "val", "test")} if self.mc.get("smoke") else None
        data_dir = export_coco0(self.ds, self.manifest, self.split, subset)
        (self.run_dir / "framework").mkdir(parents=True, exist_ok=True)
        tuning = None if resume else str(self._tuning_checkpoint())
        cwd = os.getcwd()
        core, solver_mod, dist_utils, yaml_utils = self._enter_repo()
        try:
            cfg_path = self._write_config(data_dir, yaml_utils)
            seed = int(self.run_cfg["seed"])
            dist_utils.setup_distributed(0, "builtin", seed=seed)
            upd = {"config": str(cfg_path), "seed": seed, "use_amp": bool(self.mc["train"].get("amp", True)),
                   "output_dir": str(self.run_dir / "framework"), "test_only": False, "print_method": "builtin",
                   "print_rank": 0}
            if resume:
                upd["resume"] = resume
            if tuning:
                upd["tuning"] = tuning
            if self.run_cfg.get("device") == "cpu":
                upd["device"] = "cpu"
            cfg = core.YAMLConfig(str(cfg_path), **upd)
            if (resume or tuning) and "HGNetv2" in cfg.yaml_cfg:
                cfg.yaml_cfg["HGNetv2"]["pretrained"] = False  # identical to upstream train.py
            solver = solver_mod.TASKS[cfg.yaml_cfg["task"]](cfg)
            self._patch(solver)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            stopped_early = False
            try:
                solver.fit()
            except EarlyStop as e:
                stopped_early = True
                log.info("early stopping: %s", e)
            dt = time.time() - t0
            try:
                dist_utils.cleanup()
            except Exception:
                pass
        finally:
            os.chdir(cwd)
        fw = self.run_dir / "framework"
        self._merge_logs(getattr(self, "_hist", []))  # upstream writes log.txt after evaluate -> merge once more
        best = self._best_ckpt()
        summary = {"train_time_s": round(dt, 1), "framework": f"{self.mc['model']['repo']} (in-process main)",
                   "peak_gpu_mem_allocated_gb": (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
                                                 if torch.cuda.is_available() else None),
                   "stopped_early": stopped_early, "best_checkpoint": str(best) if best else None,
                   "early_stopping": self._es, "tuning_checkpoint": tuning,
                   "parameters": {"total": int(sum(p.numel() for p in solver.model.parameters()))}}
        dump_json(summary, self.run_dir / "train_summary.json")
        return summary

    def _patch(self, solver):
        """Wrap evaluate (early stopping + val loss + our train_log) and load_tuning_state (report)."""
        import torch
        pkg = self.mc["model"]["package"]
        ds_mod = importlib.import_module(f"{pkg}.solver.det_solver")
        orig_eval = ds_mod.evaluate
        t = self.mc["train"]
        self._es = {"monitor": "val coco AP (upstream evaluator)", "patience_epochs": int(t.get("patience", 15)),
                    "best": None, "best_epoch": -1, "bad": 0}
        hist = []
        self._hist = hist
        adapter = self

        def evaluate(model, criterion, postprocessor, data_loader, evaluator, device, *a, **k):
            test_stats, coco_evaluator = orig_eval(model, criterion, postprocessor, data_loader, evaluator, device, *a, **k)
            ap = float(test_stats["coco_eval_bbox"][0])
            ep = int(getattr(solver, "last_epoch", -1))
            vl = adapter._val_loss(model, criterion, data_loader, device)
            es = adapter._es
            if es["best"] is None or ap > es["best"]:
                es.update(best=ap, best_epoch=ep, bad=0)
            else:
                es["bad"] += 1
            hist.append({"epoch": ep, "val_metric": ap, "val_AP50": float(test_stats["coco_eval_bbox"][1]),
                         "val_loss": vl, "time": time.time()})
            adapter._merge_logs(hist)
            if es["patience_epochs"] > 0 and es["bad"] >= es["patience_epochs"]:
                raise EarlyStop(f"no val AP improvement for {es['bad']} evaluations (best {es['best']:.4f} @ {es['best_epoch']})")
            return test_stats, coco_evaluator

        ds_mod.evaluate = evaluate

        orig_tune = solver.load_tuning_state

        def load_tuning_state(path):
            orig_tune(path)
            st = torch.load(path, map_location="cpu", weights_only=False)
            pre = st["ema"]["module"] if "ema" in st else st["model"]
            cur = solver.model.state_dict() if not hasattr(solver.model, "module") else solver.model.module.state_dict()
            loaded = [k for k, v in cur.items() if k in pre and pre[k].shape == v.shape]
            missing = [k for k in cur if k not in pre]
            shape_mismatch = [k for k in cur if k in pre and pre[k].shape != cur[k].shape]
            rep = {"checkpoint": str(path), "n_model_tensors": len(cur), "n_loaded": len(loaded),
                   "missing": missing, "shape_mismatch": shape_mismatch}
            dump_json(rep, adapter.run_dir / "framework" / "tuning_load_report.json")
            bad = [k for k in missing + shape_mismatch if k.startswith("backbone.")]
            if bad:
                raise RuntimeError(f"pretrained backbone NOT fully loaded ({len(bad)} tensors, e.g. {bad[:3]}) - refusing "
                                   "to train a silently re-initialised backbone")
            log.info("tuning load: %d/%d tensors loaded; not loaded: %s", len(loaded), len(cur),
                     (missing + shape_mismatch)[:10])

        solver.load_tuning_state = load_tuning_state

    def _val_loss(self, model, criterion, data_loader, device):
        """Validation loss with the training criterion (train mode, BN frozen, no grad)."""
        import torch
        from torchvision.ops import box_convert
        was = model.training
        try:
            model.train()
            for mod in model.modules():
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    mod.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                for i, (samples, targets) in enumerate(data_loader):
                    samples = samples.to(device)
                    h, w = samples.shape[-2:]
                    tg = []
                    for t in targets:
                        b = t["boxes"].to(device).float()
                        b = box_convert(b, "xyxy", "cxcywh") / torch.tensor([w, h, w, h], device=device)
                        tg.append({**{k: (v.to(device) if hasattr(v, "to") else v) for k, v in t.items()}, "boxes": b})
                    out = model(samples, targets=tg)
                    metas = {"epoch": 0, "step": i, "global_step": i, "epoch_step": len(data_loader)}
                    try:
                        ld = criterion(out, tg, **metas)
                    except TypeError:
                        metas.pop("epoch_step")
                        ld = criterion(out, tg, **metas)
                    tot += float(sum(v for k, v in ld.items() if torch.is_tensor(v)))
                    n += 1
            return tot / max(1, n)
        except Exception as e:  # noqa: BLE001 - diagnostics only
            log.warning("val loss not computed: %s", e)
            return None
        finally:
            model.train(was)

    def _merge_logs(self, hist):
        """train loss from upstream log.txt + our val loss/metric -> train_log.csv."""
        import csv
        fw = self.run_dir / "framework"
        train = {}
        lp = fw / "log.txt"
        if lp.exists():
            for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "epoch" in d:
                    train[int(d["epoch"])] = {"train_loss": d.get("train_loss"), "lr": d.get("train_lr")}
        rows = []
        for h in hist:
            r = {"epoch": h["epoch"], **train.get(h["epoch"], {}), "val_loss": h["val_loss"],
                 "val_metric": h["val_metric"], "val_AP50": h["val_AP50"]}
            rows.append(r)
        keys = ["epoch", "train_loss", "val_loss", "val_metric", "val_AP50", "lr"]
        with open(self.run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    def _best_ckpt(self) -> Path | None:
        fw = self.run_dir / "framework"
        for n in ("best_stg2.pth", "best_stg1.pth", "best.pth", "last.pth"):
            if (fw / n).exists():
                return fw / n
        return None

    # ------------------------------------------------------------------ predict
    def _build_eval(self, checkpoint, ann_dir: Path):
        import torch
        cwd = os.getcwd()
        core, _, _, _ = self._enter_repo()
        try:
            cfg_path = self.run_dir / "framework" / "config.yml"
            upd = {"val_dataloader": {"dataset": {"img_folder": ann_dir.as_posix(),
                                                  "ann_file": (ann_dir / "_annotations.coco.json").as_posix()},
                                      "shuffle": False}}
            cfg = core.YAMLConfig(str(cfg_path), **upd)
            if "HGNetv2" in cfg.yaml_cfg:
                cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
            if "PResNet" in cfg.yaml_cfg:
                cfg.yaml_cfg["PResNet"]["pretrained"] = False
            ck = Path(checkpoint) if checkpoint else self._best_ckpt()
            st = torch.load(ck, map_location="cpu", weights_only=False)
            sd = st["ema"]["module"] if "ema" in st and st["ema"] else st["model"]
            model = cfg.model
            model.load_state_dict(sd)
            dev = torch.device("cuda" if torch.cuda.is_available() and self.run_cfg.get("device") != "cpu" else "cpu")
            model = model.deploy().to(dev) if hasattr(model, "deploy") else model.eval().to(dev)
            post = cfg.postprocessor.deploy() if hasattr(cfg.postprocessor, "deploy") else cfg.postprocessor
            post.deploy_mode = False
            loader = cfg.val_dataloader
        finally:
            os.chdir(cwd)
        return model, post, loader, dev, ck, "ema" if ("ema" in st and st["ema"]) else "model"

    def predict(self, which: str, checkpoint: str | None = None) -> Path:
        import torch
        subset = {w: self.records(w) for w in ("train", "val", "test")} if self.mc.get("smoke") else None
        data_dir = export_coco0(self.ds, self.manifest, self.split, subset)
        split_dir = data_dir / {"train": "train", "val": "valid", "test": "test"}[which]
        coco = load_json(split_dir / "_annotations.coco.json")
        id2iid = {im["id"]: im["defectbench_image_id"] for im in coco["images"]}
        model, post, loader, dev, ck, weights = self._build_eval(checkpoint, split_dir)
        inf = self.mc["inference"]
        recs = {r["image_id"]: r for r in self.records(which)}
        outs = {}
        t0 = time.time()
        with torch.no_grad():
            for samples, targets in loader:
                o = model(samples.to(dev))
                sizes = torch.stack([t["orig_size"] for t in targets]).to(dev)   # (w, h) of the ORIGINAL image
                res = post(o, sizes)
                for t, r in zip(targets, res):
                    iid = id2iid[int(t["image_id"])]
                    rr = recs[iid]
                    rec = make_det_record(iid, r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy(),
                                          r["labels"].cpu().numpy().astype(int))
                    outs[iid] = postprocess_det_record(rec, rr["width"], rr["height"], self.num_classes,
                                                       float(inf["score_thresh"]), int(inf["max_dets"]))
        missing = [i for i in recs if i not in outs]
        if missing:
            raise RuntimeError(f"{len(missing)} images were not predicted (dataloader drop_last?)")
        out = self.pred_path(which)
        save_det_predictions(out, [outs[i] for i in recs], dataset=self.ds["name"], split=which,
                             model=self.mc["name"], class_names=self.classes, score_thr=float(inf["score_thresh"]),
                             max_dets=int(inf["max_dets"]),
                             extra={"checkpoint": str(ck), "weights": weights, "seconds": round(time.time() - t0, 2),
                                    "coordinates": "upstream PostProcessor: normalised cxcywh x orig_size (w,h); "
                                                   "val transform = plain Resize(640x640) (no letterbox)",
                                    "label_mapping": "remap_mscoco_category=False, 0-based category_id = canonical"})
        from ...formats import load_det_predictions
        probs = validate_det_predictions(load_det_predictions(out), {r["image_id"]: (r["width"], r["height"])
                                                                     for r in recs.values()})
        if probs:
            raise RuntimeError(f"invalid predictions: {probs[:5]}")
        return out

    def benchmark_model(self, checkpoint=None):
        import torch
        from PIL import Image
        data_dir = export_coco0(self.ds, self.manifest, self.split,
                                {w: self.records(w) for w in ("train", "val", "test")} if self.mc.get("smoke") else None)
        model, post, loader, dev, _, _ = self._build_eval(checkpoint, data_dir / "test")
        size = int(self.mc["model"].get("input_size", 640))
        norm = self.mc["model"].get("normalize")

        def pre(rec):
            with Image.open(self.raw_root / rec["file"]) as im:
                x = torch.from_numpy(np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))).permute(2, 0, 1)
            x = x.float().div(255)[None].to(dev)
            if norm:
                m = torch.tensor(norm[0], device=dev).view(1, 3, 1, 1)
                s = torch.tensor(norm[1], device=dev).view(1, 3, 1, 1)
                x = (x - m) / s
            return x, torch.tensor([[rec["width"], rec["height"]]], device=dev)

        def fwd(inp):
            x, sz = inp
            return post(model(x), sz)

        def postf(res, rec):
            r = res[0]
            return make_det_record(rec["image_id"], r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(),
                                   r["labels"].cpu().numpy())

        return fwd, pre, postf, "model() = network + upstream PostProcessor (top-k); pre = PIL resize 640 (+normalize)"
