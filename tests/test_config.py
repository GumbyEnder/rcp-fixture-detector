"""Tests for config loading."""

from rcp_detector.config import load_config, load_class_config


def test_load_default_config():
    cfg = load_config()
    assert cfg["pdf"]["dpi"] == 300
    assert cfg["tiling"]["patch_width"] == 640
    assert cfg["detection"]["confidence_threshold"] == 0.25


def test_load_class_config():
    cc = load_class_config()
    assert cc["nc"] == 15
    assert cc["names"][0] == "RECESSED_DOWNLIGHT"
    assert cc["names"][14] == "UNKNOWN_FIXTURE"
