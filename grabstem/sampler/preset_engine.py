"""Preset loading, parameter interpolation, and application."""

from ..presets.registry import get_preset, list_presets, load_presets
from ..presets.schema import PresetConfig


class PresetEngine:
    """Loads presets and provides interpolated configurations."""

    def __init__(self):
        self._presets = load_presets()

    def get(self, name: str) -> PresetConfig | None:
        return get_preset(name)

    def list_all(self) -> list[tuple[str, str, str, str]]:
        return list_presets()

    def get_chop_config(self, preset: PresetConfig) -> dict:
        """Convert preset's ChopConfig to dict for the chopper."""
        return preset.chop.model_dump()

    def get_effects_config(self, preset: PresetConfig) -> dict[str, list[dict]]:
        """Convert preset's effects to dict for the EffectsPipeline."""
        result = {}
        for stem_type, chain in preset.effects.items():
            result[stem_type] = [e.model_dump() for e in chain]
        return result
