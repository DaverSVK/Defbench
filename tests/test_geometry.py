"""Box / coordinate conversions and mask-to-box derivation."""
import numpy as np
import pytest

from defectbench.data.geometry import (ResizeTransform, box_iou, cxcywh_norm_to_xyxy, mask_to_boxes, voc_to_xyxy,
                                       xywh_to_xyxy, xyxy_to_cxcywh_norm, xyxy_to_xywh)


def test_voc_to_xyxy_pixel_edges():
    # VOC 1-based inclusive pixels 1..10 -> 10 px wide box starting at edge 0
    assert voc_to_xyxy(1, 1, 10, 20) == [0.0, 0.0, 10.0, 20.0]


def test_xywh_roundtrip():
    b = np.array([[3.0, 4.0, 13.0, 24.0]])
    assert np.allclose(xywh_to_xyxy(xyxy_to_xywh(b)), b)
    assert np.allclose(xyxy_to_xywh(b), [[3, 4, 10, 20]])


def test_cxcywh_norm_roundtrip():
    b = np.array([[10.0, 20.0, 50.0, 60.0], [0, 0, 200, 100]])
    n = xyxy_to_cxcywh_norm(b, 200, 100)
    assert np.allclose(cxcywh_norm_to_xyxy(n, 200, 100), b)
    assert np.allclose(n[1], [0.5, 0.5, 1.0, 1.0])


@pytest.mark.parametrize("ow,oh,nw,nh", [(2048, 1000, 640, 640), (200, 200, 800, 800), (123, 286, 224, 224)])
def test_letterbox_inverse(ow, oh, nw, nh):
    t = ResizeTransform.letterbox(ow, oh, nw, nh)
    b = np.array([[5.0, 7.0, ow - 3.0, oh - 1.0], [0, 0, ow, oh]])
    fwd = t.forward_boxes(b)
    # letterboxed boxes live inside the padded canvas
    assert fwd[:, [0, 2]].min() >= t.pad_x - 1e-6 and fwd[:, [1, 3]].min() >= t.pad_y - 1e-6
    assert np.allclose(t.inverse_boxes(fwd), b)


def test_stretch_inverse_and_clipping():
    t = ResizeTransform.stretch(2048, 1000, 640, 640)
    b = np.array([[100.0, 50.0, 300.0, 900.0]])
    assert np.allclose(t.inverse_boxes(t.forward_boxes(b)), b)
    # a prediction partially outside the network canvas is clipped to the original image
    out = t.inverse_boxes(np.array([[-10.0, -10.0, 700.0, 700.0]]))
    assert np.allclose(out, [[0, 0, 2048, 1000]])


def test_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert np.allclose(box_iou(a, b), [[1.0, 50 / 150, 0.0]])
    assert box_iou(np.zeros((0, 4)), b).shape == (0, 3)


def test_mask_to_boxes_components():
    m = np.zeros((20, 30), np.uint8)
    m[2:5, 3:8] = 255          # rows 2..4, cols 3..7 -> box [3,2,8,5]
    m[10:12, 20:21] = 1        # 2 pixels
    comps = mask_to_boxes(m, connectivity=8, min_area=1)
    boxes = sorted(c[0] for c in comps)
    assert boxes == [[3.0, 2.0, 8.0, 5.0], [20.0, 10.0, 21.0, 12.0]]
    assert sorted(c[1] for c in comps) == [2, 15]
    assert len(mask_to_boxes(m, min_area=3)) == 1
    assert mask_to_boxes(np.zeros((5, 5))) == []


def test_mask_to_boxes_connectivity():
    m = np.zeros((4, 4), np.uint8)
    m[0, 0] = m[1, 1] = 1      # diagonal neighbours
    assert len(mask_to_boxes(m, connectivity=8)) == 1
    assert len(mask_to_boxes(m, connectivity=4)) == 2
