"""Read/write detector configs as plain JSON so settings can be tuned in a text editor."""

import json
import os
from dataclasses import asdict, fields
from typing import Optional

from src.screening.resistance import ResistanceConfig, ResistanceDetector
from src.screening.volume_profile import VolumeProfileConfig, VolumeProfileDetector

DEFAULT_CONFIG_FILE = os.path.join("data", "labels", "detector_config.json")
DETECTORS = {
    "atr_pivot": (ResistanceDetector, ResistanceConfig, DEFAULT_CONFIG_FILE),
    "volume_profile": (VolumeProfileDetector, VolumeProfileConfig,
                       os.path.join("data", "labels", "volume_profile_config.json")),
}


def load_config(path: Optional[str] = None, cls=ResistanceConfig, **overrides):
    """Load config from JSON (if path given), then apply non-None keyword overrides."""
    values = {}
    if path:
        with open(path, encoding="utf8") as f:
            values = json.load(f)
    values.update({k: v for k, v in overrides.items() if v is not None})

    valid = {f.name for f in fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown config keys {sorted(unknown)}; valid keys: {sorted(valid)}")
    if "ma_periods" in values:
        values["ma_periods"] = tuple(values["ma_periods"])
    return cls(**values)


def dump_config(path: Optional[str] = None, config=None) -> str:
    config = config or ResistanceConfig()
    if path is None:
        path = next(p for _, c, p in DETECTORS.values() if isinstance(config, c))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(asdict(config), f, indent=2)
    return path


def make_detector(name: str = "atr_pivot", config_path: Optional[str] = None, **overrides):
    """Build a detector by name from an optional JSON config plus non-None overrides."""
    if name not in DETECTORS:
        raise ValueError(f"Unknown detector {name!r}; choose from {sorted(DETECTORS)}")
    detector_cls, config_cls, _ = DETECTORS[name]
    return detector_cls(load_config(config_path, cls=config_cls, **overrides))
