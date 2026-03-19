"""Tests for NMS logic."""

import numpy as np
from rcp_detector.detection.nms import non_max_suppression


def test_nms_empty():
    result = non_max_suppression(np.empty((0, 6)), 0.3)
    assert result.shape == (0, 6)


def test_nms_no_overlap():
    boxes = np.array([
        [0, 0, 10, 10, 0.9, 0],
        [100, 100, 110, 110, 0.8, 0],
    ])
    result = non_max_suppression(boxes, 0.3)
    assert len(result) == 2


def test_nms_high_overlap():
    boxes = np.array([
        [0, 0, 10, 10, 0.9, 0],
        [1, 1, 11, 11, 0.7, 0],
    ])
    result = non_max_suppression(boxes, 0.3)
    assert len(result) == 1
    assert result[0][4] == 0  # kept the higher-confidence box (int-cast of 0.9)
