"""AUROC/AUPRC availability and AUPRO correctness on synthetic maps."""
import numpy as np
import pytest

from defectbench.evaluation.anomaly import (aupro, binary_auroc_auprc, compute_pro_curve, image_level_metrics,
                                            pixel_level_metrics, trapezoid_limited)


def test_single_class_returns_none_with_reason():
    r = binary_auroc_auprc([1, 1, 1], [0.2, 0.5, 0.9])
    assert r["auroc"] is None and r["auprc"] is None and "only one class" in r["reason"]
    r = image_level_metrics([False, False], [0.1, 0.2], "max")
    assert r["auroc"] is None


def test_auroc_auprc_values():
    r = binary_auroc_auprc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert r["auroc"] == pytest.approx(0.75)
    # AP (step-wise): ranks 0.8(P) 0.4(N) 0.35(P) 0.1(N) -> 0.5*1 + 0.5*(2/3)
    assert r["auprc"] == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert r["auprc_trapezoid"] is not None and r["auprc_trapezoid"] != r["auprc"]


def test_pixel_metrics_na_without_masks():
    assert pixel_level_metrics(None, [np.zeros((2, 2))])["auroc"] is None
    assert aupro(None, [np.zeros((2, 2))])["aupro"] is None


def test_aupro_perfect_map_is_one():
    gt = np.zeros((20, 20), bool); gt[2:6, 2:6] = True; gt[10:12, 14:19] = True
    res = aupro([gt, np.zeros((20, 20), bool)], [gt.astype(float), np.zeros((20, 20))])
    assert res["aupro"] == pytest.approx(1.0)
    assert res["n_regions"] == 2


def test_aupro_inverted_map_is_zero():
    gt = np.zeros((10, 10), bool); gt[2:5, 2:5] = True
    res = aupro([gt], [(~gt).astype(float)])
    assert res["aupro"] == pytest.approx(0.0, abs=1e-12)


def test_aupro_region_weighting_small_region_counts_equally():
    """Two regions (big and small); the map detects only the big one perfectly.
    PRO = mean over regions -> 0.5 at FPR 0 regardless of the area ratio."""
    gt = np.zeros((40, 40), bool)
    gt[0:20, 0:20] = True     # 400 px
    gt[30:32, 30:32] = True   # 4 px
    s = np.zeros((40, 40)); s[0:20, 0:20] = 1.0
    fpr, pro, thr, n, _ = compute_pro_curve([gt], [s])
    assert n == 2
    # at threshold 1.0: FPR 0, PRO 0.5
    i = np.nonzero(thr == 1.0)[0][0]
    assert fpr[i] == 0.0 and pro[i] == pytest.approx(0.5)
    # at threshold 0 everything positive: PRO 1, FPR 1 -> area up to 0.3 with linear interpolation:
    # curve goes (0,0)->(0,0.5)->(1,1); on [0,0.3] y = 0.5 + 0.5x -> area = 0.15 + 0.0225 = 0.1725, /0.3
    res = aupro([gt], [s])
    assert res["aupro"] == pytest.approx((0.5 * 0.3 + 0.5 * 0.3 ** 2 / 2) / 0.3)


def test_aupro_matches_bruteforce():
    rng = np.random.default_rng(0)
    gts, maps = [], []
    for _ in range(3):
        g = np.zeros((16, 16), bool)
        x, y = rng.integers(0, 12, 2)
        g[y:y + 4, x:x + 3] = True
        gts.append(g)
        maps.append(np.clip(g * 0.6 + rng.random((16, 16)) * 0.5, 0, 1).round(2))
    fpr, pro, thr, n, _ = compute_pro_curve(gts, maps)
    import cv2
    for t in [0.2, 0.45, 0.7, 0.9]:
        negs = np.concatenate([m[~g] for g, m in zip(gts, maps)])
        fp_rate = (negs >= t).mean()
        overl = []
        for g, m in zip(gts, maps):
            k, lab = cv2.connectedComponents(g.astype(np.uint8), connectivity=8)
            for r in range(1, k):
                overl.append((m[lab == r] >= t).mean())
        i = np.nonzero(thr <= t)[0]
        # curve point at the smallest threshold >= t equals the brute force point at t
        j = np.nonzero(thr >= t)[0][-1]
        assert fpr[j] == pytest.approx(fp_rate) and pro[j] == pytest.approx(np.mean(overl))


def test_trapezoid_limited_interpolation():
    x = np.array([0.0, 0.2, 0.6, 1.0]); y = np.array([0.0, 0.4, 0.8, 1.0])
    # area on [0,0.3]: 0.5*0.2*0.4 + (0.4+0.5)/2*0.1 = 0.04 + 0.045
    assert trapezoid_limited(x, y, 0.3) == pytest.approx(0.085)


def test_monotone_transform_invariance():
    gt = np.zeros((12, 12), bool); gt[3:7, 3:7] = True
    s = np.random.default_rng(1).random((12, 12)) + gt * 0.3
    a = aupro([gt], [s])["aupro"]
    b = aupro([gt], [np.exp(3 * s)])["aupro"]
    assert a == pytest.approx(b)
