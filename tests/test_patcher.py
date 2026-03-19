"""Tests for tiling/patcher coordinate conversions."""

from rcp_detector.tiling.patcher import _xywh_to_xyxy, _xyxy_to_xywh


def test_xywh_to_xyxy_roundtrip():
    lines = ["0 0.5 0.5 0.1 0.2"]
    labels = _xywh_to_xyxy(lines, 1000, 1000)
    assert len(labels) == 1
    cls, x1, y1, x2, y2 = labels[0]
    assert cls == 0
    # Center (500, 500), size (100, 200) → (450, 400, 550, 600)
    assert x1 == 450
    assert y1 == 400
    assert x2 == 550
    assert y2 == 600


def test_xyxy_to_xywh():
    label = [0, 450, 400, 550, 600]
    result = _xyxy_to_xywh(label, 1000, 1000)
    assert result[0] == 0
    assert abs(result[1] - 0.5) < 0.001
    assert abs(result[2] - 0.5) < 0.001
    assert abs(result[3] - 0.1) < 0.001
    assert abs(result[4] - 0.2) < 0.001
