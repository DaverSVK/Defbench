"""CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018).

The module code below is copied verbatim (apart from formatting and the removal of
unused imports) from the authors' official implementation:
  https://github.com/Jongchan/attention-module/blob/459efad0e05ee7dde50c41ca10a3d0800bc3792a/MODELS/cbam.py
  License: MIT (c) Jongchan Park, Sanghyun Woo

``FPNWithCBAM`` is our integration into torchvision's Faster R-CNN ResNet-50-FPN:

* CBAM is inserted AFTER the FPN output 3x3 convolutions, i.e. on the feature maps
  that feed the RPN and the RoI heads.
* Default levels: P2, P3, P4, P5 (torchvision keys '0','1','2','3'; strides 4, 8, 16, 32).
  One independent CBAM per level (no weight sharing), gate_channels = 256,
  reduction_ratio = 16, pool_types = ['avg', 'max'], spatial gate with 7x7 conv + BN.
* P6 ('pool', stride 64, used only by the RPN) is recomputed from the CBAM-refined P5
  with the same operation torchvision uses (max_pool2d(kernel=1, stride=2)), so the
  RPN sees refined features on all levels. CBAM itself is NOT applied to P6
  (configurable via ``levels``).
* Everything else (ResNet-50 backbone, FPN, RPN, RoI heads, COCO initialisation,
  hyper-parameters, seeds, data) is identical to the reference model -> controlled
  comparison. The CBAM parameters are the only newly initialised weights (PyTorch
  default init, as in the official code); ``load_state_dict`` reports exactly these
  as missing keys (checked in the adapter).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=("avg", "max")):
        super().__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == "lp":
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == "lse":
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)
            channel_att_sum = channel_att_raw if channel_att_sum is None else channel_att_sum + channel_att_raw
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    return s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=("avg", "max"), no_spatial=False):
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


# ------------------------------------------------------------------------ integration
FPN_LEVEL_NAMES = {"P2": "0", "P3": "1", "P4": "2", "P5": "3", "P6": "pool"}


class BackboneWithFPNAndCBAM(nn.Module):
    """Wraps torchvision ``BackboneWithFPN``; applies CBAM to selected FPN outputs."""

    def __init__(self, backbone_with_fpn: nn.Module, levels=("P2", "P3", "P4", "P5"), reduction_ratio: int = 16,
                 recompute_p6_from_refined_p5: bool = True):
        super().__init__()
        self.inner = backbone_with_fpn
        self.out_channels = backbone_with_fpn.out_channels
        self.levels = list(levels)
        self.keys = [FPN_LEVEL_NAMES[l] for l in self.levels]
        self.cbam = nn.ModuleDict({k: CBAM(self.out_channels, reduction_ratio) for k in self.keys})
        self.recompute_p6 = recompute_p6_from_refined_p5 and "pool" not in self.keys

    def forward(self, x):
        feats = self.inner(x)
        out = type(feats)()
        for k, v in feats.items():
            out[k] = self.cbam[k](v) if k in self.cbam else v
        if self.recompute_p6 and "pool" in out and "3" in self.cbam:
            out["pool"] = F.max_pool2d(out["3"], 1, 2, 0)  # identical op to torchvision LastLevelMaxPool
        return out

    def describe(self) -> dict:
        return {"cbam_levels": self.levels, "torchvision_keys": self.keys, "placement": "after FPN output convs",
                "p6": "recomputed from refined P5 via max_pool2d(1,2)" if self.recompute_p6 else "unchanged",
                "reduction_ratio": next(iter(self.cbam.values())).ChannelGate.mlp[1].in_features //
                max(1, next(iter(self.cbam.values())).ChannelGate.mlp[1].out_features),
                "weight_sharing": False,
                "params": sum(p.numel() for p in self.cbam.parameters())}
