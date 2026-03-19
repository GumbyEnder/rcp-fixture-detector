"""YAML configuration loader with defaults merging."""

from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(user_config: str | Path | None = None) -> dict[str, Any]:
    """Load default config, optionally merged with a user override file."""
    with open(_DEFAULT_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    if user_config is not None:
        with open(user_config) as f:
            overrides = yaml.safe_load(f) or {}
        config = _deep_merge(config, overrides)

    return config


def load_class_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the RCP class definition YAML."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config" / "rcp_classes.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
