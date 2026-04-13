"""Shared audio I/O helpers."""

from enum import Enum
from pathlib import Path

import numpy as np


class NormalizationStrategy(Enum):
    """How to normalize samples in a pack."""

    PEAK_INDIVIDUAL = "peak_individual"  # Each sample to -1dBFS independently (legacy)
    PEAK_GROUP = "peak_group"  # Loudest in each classification group to -1dBFS, others scaled proportionally
    LUFS_GROUP = "lufs_group"  # Group-aware perceptual loudness (ITU-R BS.1770)
    RELATIVE = "relative"  # Loudest sample in entire pack to -1dBFS, everything else proportional


def load_audio(path: Path, sr: int = 44100) -> tuple[np.ndarray, int]:
    """Load an audio file, return (samples, sample_rate)."""
    import librosa

    y, loaded_sr = librosa.load(str(path), sr=sr, mono=False)
    # If stereo, keep as-is; librosa returns (channels, samples) for mono=False
    return y, loaded_sr


def save_audio(
    audio: np.ndarray,
    sr: int,
    path: Path,
    bit_depth: int = 24,
) -> None:
    """Save audio to WAV using pedalboard for bit-depth control."""
    from pedalboard.io import AudioFile

    # Ensure 2D: (channels, samples)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    # WAV files don't accept quality param — use soundfile for bit-depth control instead
    import soundfile as sf

    # soundfile expects (samples, channels)
    data = audio.T if audio.ndim > 1 else audio
    subtype = f"PCM_{bit_depth}"
    sf.write(str(path), data, sr, subtype=subtype)


def normalize(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Peak-normalize audio to target dBFS."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    target_linear = 10 ** (target_db / 20.0)
    return audio * (target_linear / peak)


def measure_lufs(audio: np.ndarray, sr: int) -> float:
    """Measure integrated loudness in LUFS (ITU-R BS.1770)."""
    import pyloudnorm

    meter = pyloudnorm.Meter(sr)
    # pyloudnorm expects (samples,) or (samples, channels)
    if audio.ndim == 1:
        return meter.integrated_loudness(audio)
    else:
        return meter.integrated_loudness(audio.T)


def normalize_group(
    audio_items: list[tuple[str, np.ndarray, int]],
    strategy: NormalizationStrategy,
    target_db: float = -1.0,
    target_lufs: float = -14.0,
) -> dict[str, np.ndarray]:
    """Normalize a group of audio samples according to the strategy.

    Args:
        audio_items: list of (key, audio_array, sample_rate) tuples.
            Key is used to identify each sample in the result.
        strategy: normalization approach.
        target_db: peak target for PEAK strategies.
        target_lufs: loudness target for LUFS strategy.

    Returns:
        dict mapping key -> normalized audio array.
    """
    if not audio_items:
        return {}

    if strategy == NormalizationStrategy.PEAK_INDIVIDUAL:
        return {key: normalize(audio, target_db) for key, audio, sr in audio_items}

    if strategy == NormalizationStrategy.RELATIVE:
        # Find the global peak across all samples
        global_peak = max(
            float(np.max(np.abs(audio))) for _, audio, _ in audio_items
        )
        if global_peak == 0:
            return {key: audio for key, audio, _ in audio_items}
        target_linear = 10 ** (target_db / 20.0)
        gain = target_linear / global_peak
        return {key: audio * gain for key, audio, _ in audio_items}

    if strategy == NormalizationStrategy.PEAK_GROUP:
        # Group by classification (embedded in key as "classification::id")
        groups: dict[str, list[tuple[str, np.ndarray, int]]] = {}
        for key, audio, sr in audio_items:
            cls = key.split("::")[0] if "::" in key else "default"
            groups.setdefault(cls, []).append((key, audio, sr))

        result = {}
        for cls, items in groups.items():
            group_peak = max(
                float(np.max(np.abs(audio))) for _, audio, _ in items
            )
            if group_peak == 0:
                for key, audio, _ in items:
                    result[key] = audio
                continue
            target_linear = 10 ** (target_db / 20.0)
            gain = target_linear / group_peak
            for key, audio, _ in items:
                result[key] = audio * gain
        return result

    if strategy == NormalizationStrategy.LUFS_GROUP:
        # Group by classification, normalize each group to target LUFS
        groups: dict[str, list[tuple[str, np.ndarray, int]]] = {}
        for key, audio, sr in audio_items:
            cls = key.split("::")[0] if "::" in key else "default"
            groups.setdefault(cls, []).append((key, audio, sr))

        result = {}
        for cls, items in groups.items():
            # Find the loudest sample in the group (by LUFS)
            loudest_lufs = -float("inf")
            for _, audio, sr in items:
                try:
                    lufs = measure_lufs(audio, sr)
                    if lufs > loudest_lufs:
                        loudest_lufs = lufs
                except Exception:
                    pass

            if loudest_lufs == -float("inf") or loudest_lufs < -70:
                # Too quiet to measure, fall back to peak normalization
                for key, audio, _ in items:
                    result[key] = normalize(audio, target_db)
                continue

            # Apply the same gain offset to all samples in the group
            gain_db = target_lufs - loudest_lufs
            gain_linear = 10 ** (gain_db / 20.0)
            for key, audio, _ in items:
                normalized = audio * gain_linear
                # Safety limiter: prevent clipping
                peak = float(np.max(np.abs(normalized)))
                if peak > 1.0:
                    normalized = normalized / peak
                result[key] = normalized
        return result

    # Fallback
    return {key: normalize(audio, target_db) for key, audio, sr in audio_items}


def trim_silence(
    audio: np.ndarray, sr: int, threshold_db: float = -40.0
) -> np.ndarray:
    """Trim leading and trailing silence."""
    import librosa

    if audio.ndim > 1:
        # Use first channel for detection, trim all
        trimmed, idx = librosa.effects.trim(audio[0], top_db=abs(threshold_db))
        return audio[:, idx[0] : idx[1]]
    trimmed, _ = librosa.effects.trim(audio, top_db=abs(threshold_db))
    return trimmed


def fade(
    audio: np.ndarray,
    sr: int,
    fade_in_ms: float = 5.0,
    fade_out_ms: float = 10.0,
) -> np.ndarray:
    """Apply cosine fade in/out to prevent clicks."""
    fade_in_samples = int(sr * fade_in_ms / 1000)
    fade_out_samples = int(sr * fade_out_ms / 1000)

    result = audio.copy()

    if audio.ndim == 1:
        n = len(result)
        if fade_in_samples > 0 and n > fade_in_samples:
            curve = np.cos(np.linspace(np.pi, 2 * np.pi, fade_in_samples)) * 0.5 + 0.5
            result[:fade_in_samples] *= curve
        if fade_out_samples > 0 and n > fade_out_samples:
            curve = np.cos(np.linspace(0, np.pi, fade_out_samples)) * 0.5 + 0.5
            result[-fade_out_samples:] *= curve
    else:
        n = result.shape[1]
        if fade_in_samples > 0 and n > fade_in_samples:
            curve = np.cos(np.linspace(np.pi, 2 * np.pi, fade_in_samples)) * 0.5 + 0.5
            result[:, :fade_in_samples] *= curve
        if fade_out_samples > 0 and n > fade_out_samples:
            curve = np.cos(np.linspace(0, np.pi, fade_out_samples)) * 0.5 + 0.5
            result[:, -fade_out_samples:] *= curve

    return result
