"""Preset discovery and loading."""

import json
from pathlib import Path

from .schema import PresetConfig

_PRESETS_FILE = Path(__file__).parent / "presets.json"
_cache: list[PresetConfig] | None = None


def load_presets() -> list[PresetConfig]:
    """Load and validate all presets from presets.json."""
    global _cache
    if _cache is not None:
        return _cache

    with open(_PRESETS_FILE) as f:
        raw = json.load(f)

    _cache = [PresetConfig(**p) for p in raw]
    return _cache


def get_preset(name: str) -> PresetConfig | None:
    """Look up a preset by name (case-insensitive)."""
    for p in load_presets():
        if p.name.lower() == name.lower():
            return p
    return None


def list_presets() -> list[tuple[str, str, str, str]]:
    """Return (name, genre, category, description) tuples for GUI display."""
    return [(p.name, p.genre, p.category, p.description) for p in load_presets()]
