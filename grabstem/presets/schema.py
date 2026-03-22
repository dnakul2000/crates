"""Pydantic models for preset validation."""

from typing import Literal

from pydantic import BaseModel


class ChopConfig(BaseModel):
    mode: Literal["onset", "beat_grid", "phrase", "transient", "granular", "syllable", "random"]
    onset_sensitivity: float = 0.5
    beat_division: float = 1.0
    phrase_beats: int = 4
    grain_size_ms: float = 50.0
    min_chop_ms: float = 100.0
    max_chop_ms: float = 8000.0
    swing_amount: float = 0.0
    silence_threshold_db: float = -40.0
    prefer_downbeats: bool = False
    crossfade_ms: float = 5.0
    normalization: Literal["peak_individual", "peak_group", "lufs_group", "relative"] = "peak_group"


class EffectConfig(BaseModel):
    type: str
    params: dict[str, float] = {}
    enabled: bool = True


class PadMappingConfig(BaseModel):
    strategy: Literal["grouped", "chromatic", "sequential", "hybrid"]
    bank_assignments: dict[str, dict] = {}
    sort_by: Literal["pitch", "time", "energy", "duration", "random"] = "time"
    velocity_layers: int = 1


class PresetConfig(BaseModel):
    name: str
    genre: str
    category: str
    description: str
    tags: list[str] = []
    chop: ChopConfig
    effects: dict[str, list[EffectConfig]]  # stem_type -> effect chain
    pad_mapping: PadMappingConfig
