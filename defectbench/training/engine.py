"""Generic training engine for the in-process (core environment) adapters.

Features: CUDA + AMP (fp16 with GradScaler or bf16), gradient accumulation,
warmup + cosine / multistep / poly schedules (per iteration), optional frozen
backbone for the first N epochs, gradient clipping, validation loss + validation
metric every ``eval_interval`` epochs, early stopping on the validation metric,
``best.pt`` / ``last.pt`` checkpoints, true resume (model, optimizer, scheduler,
GradScaler, epoch/iteration counters, early-stopping state, RNG states, history),
wall-clock timing and peak GPU memory. OOM errors are caught and a concrete fix is
printed - the experiment configuration is never changed silently.
"""
from __future__ import annotations

import csv
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from ..utils.io import dump_json
from ..utils.seed import rng_state, set_rng_state

log = logging.getLogger(__name__)


class OOMAbort(RuntimeError):
    pass


OOM_HELP = """
CUDA out of memory in {where} (batch_size={bs}, grad_accum={acc}, amp={amp}, input={inp}).
The run was stopped WITHOUT changing the experiment. Options (choose explicitly and record the change):
  1. keep the effective batch size: --set train.batch_size={half} --set train.grad_accum={dbl}
  2. use the low-memory profile of this model:  --profile gpu8gb
  3. enable mixed precision if disabled:        --set train.amp=true
  4. reduce the input resolution (CHANGES THE METHOD - report it): e.g. --set model.min_size=640
  5. close other GPU processes (nvidia-smi) and retry.
"""


def build_optimizer(params, cfg: dict) -> torch.optim.Optimizer:
    t = cfg.get("type", "sgd").lower()
    lr = float(cfg["lr"])
    wd = float(cfg.get("weight_decay", 0.0))
    if t == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=float(cfg.get("momentum", 0.9)), weight_decay=wd,
                               nesterov=bool(cfg.get("nesterov", False)))
    if t == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=tuple(cfg.get("betas", (0.9, 0.999))))
    if t == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    raise ValueError(f"unknown optimizer {t}")


def build_scheduler(opt, cfg: dict, total_iters: int):
    t = cfg.get("type", "warmup_cosine")
    warm = int(cfg.get("warmup_iters", 0))
    wf = float(cfg.get("warmup_factor", 0.001))
    min_ratio = float(cfg.get("min_lr_ratio", 0.0))
    milestones = [int(m * total_iters) for m in cfg.get("milestones", [0.67, 0.89])]
    gamma = float(cfg.get("gamma", 0.1))
    power = float(cfg.get("power", 0.9))

    def f(it):
        w = 1.0
        if it < warm:
            a = it / max(1, warm)
            w = wf * (1 - a) + a
        if t == "warmup_cosine":
            p = min(1.0, it / max(1, total_iters))
            return w * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))
        if t == "warmup_multistep":
            return w * gamma ** sum(1 for m in milestones if it >= m)
        if t == "poly":
            return w * max(0.0, 1 - it / max(1, total_iters)) ** power
        if t == "constant":
            return w
        raise ValueError(f"unknown scheduler {t}")

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@dataclass
class EarlyStopping:
    monitor: str = "val_metric"
    mode: str = "max"
    patience: int = 10
    min_delta: float = 0.0
    best: float | None = None
    best_epoch: int = -1
    bad_evals: int = 0

    def step(self, value: float | None, epoch: int) -> bool:
        """Returns True if this is a new best."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            self.bad_evals += 1
            return False
        better = self.best is None or (value > self.best + self.min_delta if self.mode == "max"
                                       else value < self.best - self.min_delta)
        if better:
            self.best, self.best_epoch, self.bad_evals = value, epoch, 0
        else:
            self.bad_evals += 1
        return better

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad_evals >= self.patience


@dataclass
class TrainerHooks:
    """Model-family specific functions supplied by the adapter."""
    train_step: Callable            # (model, batch, device) -> (loss tensor, dict of floats)
    val_loss: Callable | None       # (model, loader, device, amp_ctx) -> float | None
    evaluate: Callable              # (model, device) -> dict with key "val_metric" (+ extras)
    backbone_params: Callable | None = None  # model -> iterable of params (for freezing)


@dataclass
class Trainer:
    model: torch.nn.Module
    hooks: TrainerHooks
    train_loader: object
    val_loader: object | None
    cfg: dict
    run_dir: Path
    device: torch.device
    history: list = field(default_factory=list)

    def __post_init__(self):
        t = self.cfg
        self.epochs = int(t["epochs"])
        self.accum = max(1, int(t.get("grad_accum", 1)))
        self.amp = bool(t.get("amp", True)) and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if t.get("amp_dtype", "fp16") == "bf16" else torch.float16
        self.clip = t.get("grad_clip")
        self.max_iters_per_epoch = t.get("max_iters_per_epoch")  # smoke tests only
        self.freeze_epochs = int(t.get("freeze_backbone_epochs", 0))
        self.eval_interval = int(t.get("eval_interval", 1))
        self.log_every = int(t.get("log_every", 100))
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = build_optimizer(params, t["optimizer"])
        n_iter = len(self.train_loader)
        if self.max_iters_per_epoch:
            n_iter = min(n_iter, int(self.max_iters_per_epoch))
        self.iters_per_epoch = max(1, n_iter // self.accum)
        self.sched = build_scheduler(self.opt, t.get("scheduler", {}), self.iters_per_epoch * self.epochs)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp and self.amp_dtype == torch.float16)
        es = t.get("early_stopping", {}) or {}
        self.es = EarlyStopping(monitor=es.get("monitor", "val_metric"), mode=es.get("mode", "max"),
                                patience=int(es.get("patience", 10)), min_delta=float(es.get("min_delta", 0.0)))
        self.start_epoch = 0
        self.global_step = 0
        self.train_time_s = 0.0
        self.peak_prev: dict = {}
        self.ckpt_dir = Path(self.run_dir) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- checkpointing
    def _state(self, epoch: int) -> dict:
        return {"model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
                "scheduler": self.sched.state_dict(), "scaler": self.scaler.state_dict(),
                "epoch": epoch, "global_step": self.global_step, "early_stopping": self.es.__dict__.copy(),
                "history": self.history, "rng": rng_state(), "train_time_s": self.train_time_s,
                "peak_mem_prev_gb": self._peak_gb(), "train_cfg": self.cfg}

    def _peak_gb(self, kind: str = "allocated") -> float | None:
        """Peak GPU memory over ALL sessions of this run (previous sessions are stored in the checkpoint)."""
        if self.device.type != "cuda":
            return None
        cur = (torch.cuda.max_memory_allocated(self.device) if kind == "allocated"
               else torch.cuda.max_memory_reserved(self.device)) / 1024 ** 3
        prev = self.peak_prev.get(kind) or 0.0
        return round(max(cur, prev), 3)

    def save(self, name: str, epoch: int) -> Path:
        p = self.ckpt_dir / name
        tmp = p.with_suffix(".tmp")
        torch.save(self._state(epoch), tmp)
        tmp.replace(p)
        return p

    def resume(self, path: str | Path) -> None:
        st = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["optimizer"])
        self.sched.load_state_dict(st["scheduler"])
        self.scaler.load_state_dict(st["scaler"])
        self.start_epoch = int(st["epoch"]) + 1
        self.global_step = int(st["global_step"])
        self.es.__dict__.update(st["early_stopping"])
        self.history = st.get("history", [])
        self.train_time_s = float(st.get("train_time_s", 0.0))
        pm = st.get("peak_mem_prev_gb")
        if isinstance(pm, (int, float)):
            self.peak_prev = {"allocated": float(pm)}
        # older checkpoints: fall back to the per-epoch peaks recorded in the history
        hist_peak = [h.get("peak_mem_gb") for h in self.history if isinstance(h.get("peak_mem_gb"), (int, float))]
        if hist_peak:
            self.peak_prev["allocated"] = max(self.peak_prev.get("allocated", 0.0), max(hist_peak))
        set_rng_state(st["rng"])
        log.info("resumed from %s at epoch %d (global step %d, best %s @ %d)", path, self.start_epoch,
                 self.global_step, self.es.best, self.es.best_epoch)

    # --------------------------------------------------------------- helpers
    def _set_backbone_trainable(self, flag: bool) -> None:
        if self.hooks.backbone_params is None:
            return
        for p in self.hooks.backbone_params(self.model):
            p.requires_grad_(flag)

    def _oom(self, where: str):
        bs = getattr(self.train_loader, "batch_size", "?")
        msg = OOM_HELP.format(where=where, bs=bs, acc=self.accum, amp=self.amp, inp=self.cfg.get("_input_desc", "?"),
                              half=max(1, (bs if isinstance(bs, int) else 2) // 2), dbl=self.accum * 2)
        log.error(msg)
        dump_json({"status": "oom", "where": where, "message": msg}, Path(self.run_dir) / "oom.json")
        raise OOMAbort(msg)

    def _write_log(self):
        p = Path(self.run_dir) / "train_log.csv"
        if not self.history:
            return
        keys = sorted({k for h in self.history for k in h})
        order = ["epoch", "train_loss", "val_loss", "val_metric", "lr", "epoch_time_s", "peak_mem_gb"]
        keys = [k for k in order if k in keys] + [k for k in keys if k not in order]
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for h in self.history:
                w.writerow(h)

    # --------------------------------------------------------------- main loop
    def fit(self) -> dict:
        dev = self.device
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        stopped_early = False
        for epoch in range(self.start_epoch, self.epochs):
            if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            self._set_backbone_trainable(epoch >= self.freeze_epochs)
            self.model.train()
            t0 = time.time()
            run_loss, n_batches, parts = 0.0, 0, {}
            self.opt.zero_grad(set_to_none=True)
            try:
                for i, batch in enumerate(self.train_loader):
                    if self.max_iters_per_epoch and i >= int(self.max_iters_per_epoch):
                        break
                    with torch.autocast(device_type=dev.type, dtype=self.amp_dtype, enabled=self.amp):
                        loss, loss_parts = self.hooks.train_step(self.model, batch, dev)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss {loss.item()} at epoch {epoch} iter {i}: "
                                                 f"{loss_parts}. Lower the learning rate or check the data.")
                    self.scaler.scale(loss / self.accum).backward()
                    if (i + 1) % self.accum == 0:
                        if self.clip:
                            self.scaler.unscale_(self.opt)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.clip))
                        self.scaler.step(self.opt)
                        self.scaler.update()
                        self.opt.zero_grad(set_to_none=True)
                        self.sched.step()
                        self.global_step += 1
                    run_loss += float(loss.detach())
                    n_batches += 1
                    if self.log_every and n_batches % self.log_every == 0:
                        el = time.time() - t0
                        tot = self.max_iters_per_epoch or len(self.train_loader)
                        log.info("  epoch %d iter %d/%d | loss %.4f | %.2f it/s | epoch ETA %.0fs", epoch, n_batches,
                                 tot, run_loss / n_batches, n_batches / el, el / n_batches * (int(tot) - n_batches))
                    for k, v in loss_parts.items():
                        parts[k] = parts.get(k, 0.0) + float(v)
            except torch.OutOfMemoryError:
                self._oom(f"training epoch {epoch}")
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            ep_time = time.time() - t0
            self.train_time_s += ep_time
            rec = {"epoch": epoch, "train_loss": run_loss / max(1, n_batches),
                   "lr": self.opt.param_groups[0]["lr"], "epoch_time_s": round(ep_time, 2)}
            rec.update({f"train_{k}": v / max(1, n_batches) for k, v in parts.items()})
            is_best = False
            if (epoch + 1) % self.eval_interval == 0 or epoch == self.epochs - 1:
                try:
                    amp_ctx = lambda: torch.autocast(device_type=dev.type, dtype=self.amp_dtype, enabled=self.amp)
                    if self.hooks.val_loss is not None and self.val_loader is not None:
                        rec["val_loss"] = self.hooks.val_loss(self.model, self.val_loader, dev, amp_ctx)
                    ev = self.hooks.evaluate(self.model, dev)
                except torch.OutOfMemoryError:
                    self._oom(f"validation epoch {epoch}")
                rec["val_metric"] = ev.get("val_metric")
                rec.update({f"val_{k}": v for k, v in ev.items() if k != "val_metric" and isinstance(v, (int, float))})
                is_best = self.es.step(rec["val_metric"], epoch)
            if dev.type == "cuda":
                rec["peak_mem_gb"] = self._peak_gb()
            self.history.append(rec)
            self._write_log()
            self.save("last.pt", epoch)
            if is_best:
                self.save("best.pt", epoch)
            log.info("epoch %d | train %.4f | val_loss %s | val_metric %s | best %s@%d | %.1fs", epoch,
                     rec["train_loss"], _fmt(rec.get("val_loss")), _fmt(rec.get("val_metric")), _fmt(self.es.best),
                     self.es.best_epoch, ep_time)
            if self.es.should_stop:
                log.info("early stopping at epoch %d (no improvement of %s for %d evaluations)", epoch,
                         self.es.monitor, self.es.patience)
                stopped_early = True
                break
        summary = {
            "epochs_completed": (self.history[-1]["epoch"] + 1) if self.history else 0,
            "best_epoch": self.es.best_epoch, "best_val_metric": self.es.best,
            "monitor": self.es.monitor, "stopped_early": stopped_early,
            "train_time_s": round(self.train_time_s, 1),
            "peak_gpu_mem_allocated_gb": self._peak_gb("allocated"),
            "peak_gpu_mem_reserved_gb": self._peak_gb("reserved"),
            "peak_mem_note": "maximum over all sessions of this run (resume-aware)",
            "amp": self.amp, "amp_dtype": str(self.amp_dtype), "grad_accum": self.accum,
            "batch_size": getattr(self.train_loader, "batch_size", None),
            "effective_batch_size": (getattr(self.train_loader, "batch_size", 0) or 0) * self.accum,
            "iters_per_epoch": self.iters_per_epoch,
        }
        if not (self.ckpt_dir / "best.pt").exists():
            # no validation improvement recorded (e.g. metric None) - keep last as best, flagged
            self.save("best.pt", self.history[-1]["epoch"] if self.history else 0)
            summary["best_is_last_fallback"] = True
        dump_json(summary, Path(self.run_dir) / "train_summary.json")
        return summary


def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


def run_guarded(fn, *a, **k):
    """Run a training function and turn OOMAbort into exit code 3 for the orchestrator."""
    try:
        return fn(*a, **k)
    except OOMAbort:
        sys.exit(3)
