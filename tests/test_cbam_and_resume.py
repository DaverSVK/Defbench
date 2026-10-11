"""CBAM variant = reference + CBAM only; the training engine resumes exactly."""
import copy

import pytest
import torch

from defectbench.models.adapters.torchvision_frcnn import build_frcnn
from defectbench.training.engine import Trainer, TrainerHooks

MC = {
    "model": {"variant": "v1", "trainable_backbone_layers": 3, "min_size": 128, "max_size": 128, "cbam": None},
    "pretrained": {"init": "none", "from_scratch_explicit": True},
    "inference": {"score_thresh": 0.001, "max_dets": 100, "nms_iou": 0.5},
}


def test_cbam_adds_only_cbam_parameters_and_keeps_shared_init():
    torch.manual_seed(0)
    ref = build_frcnn(MC, 3)
    mc = copy.deepcopy(MC)
    mc["model"]["cbam"] = {"levels": ["P2", "P3", "P4", "P5"], "reduction_ratio": 16}
    torch.manual_seed(0)
    cb = build_frcnn(mc, 3)
    ref_names = {n for n, _ in ref.named_parameters()}
    cb_names = {n.replace("backbone.inner.", "backbone.") for n, _ in cb.named_parameters()
                if not n.startswith("backbone.cbam.")}
    assert ref_names == cb_names
    new = [n for n, _ in cb.named_parameters() if n.startswith("backbone.cbam.")]
    assert {n.split(".")[2] for n in new} == {"0", "1", "2", "3"}  # one CBAM per FPN level P2..P5
    # shared parameters identical for the same seed (CBAM is created last)
    sd_ref = ref.state_dict()
    for n, v in cb.state_dict().items():
        if n.startswith("backbone.cbam."):
            continue
        assert torch.equal(v, sd_ref[n.replace("backbone.inner.", "backbone.")]), n


def test_cbam_forward_levels_and_p6_recomputed():
    mc = copy.deepcopy(MC)
    mc["model"]["cbam"] = {"levels": ["P2", "P3", "P4", "P5"]}
    m = build_frcnn(mc, 2).eval()
    feats = m.backbone(torch.rand(1, 3, 128, 128))
    assert list(feats) == ["0", "1", "2", "3", "pool"]
    assert torch.allclose(feats["pool"], torch.nn.functional.max_pool2d(feats["3"], 1, 2, 0))


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.l(x)


class _Crash(Exception):
    pass


def _trainer(tmp, epochs, crash_at_eval=None):
    torch.manual_seed(1)
    data = [(torch.randn(8, 4), torch.randn(8, 1)) for _ in range(5)]
    model = _Toy()
    calls = {"n": 0}

    def evaluate(m, d):
        calls["n"] += 1
        if crash_at_eval is not None and calls["n"] == crash_at_eval:
            raise _Crash("simulated interruption")
        return {"val_metric": -float(m.l.weight.detach().abs().sum())}

    hooks = TrainerHooks(train_step=lambda m, b, d: (torch.nn.functional.mse_loss(m(b[0]), b[1]), {}),
                         val_loss=None, evaluate=evaluate)
    cfg = {"epochs": epochs, "grad_accum": 2, "amp": False,
           "optimizer": {"type": "adamw", "lr": 0.01, "weight_decay": 0.01},
           "scheduler": {"type": "warmup_cosine", "warmup_iters": 2}, "early_stopping": {"patience": 0}}
    return Trainer(model, hooks, data, None, cfg, tmp, torch.device("cpu"))


def test_resume_reproduces_uninterrupted_training(tmp_path):
    full = _trainer(tmp_path / "full", 4)
    full.fit()
    # same 4-epoch run, interrupted during epoch index 2 (after last.pt of epoch 1 was written)
    part = _trainer(tmp_path / "part", 4, crash_at_eval=3)
    with pytest.raises(_Crash):
        part.fit()
    res = _trainer(tmp_path / "part", 4)
    res.resume(tmp_path / "part" / "checkpoints" / "last.pt")
    assert res.start_epoch == 2
    res.fit()
    for a, b in zip(full.model.parameters(), res.model.parameters()):
        assert torch.allclose(a, b, atol=1e-6)
    assert res.sched.last_epoch == full.sched.last_epoch
    assert [h["epoch"] for h in res.history] == [0, 1, 2, 3]
