"""Audio analysis using librosa — BPM, key, onsets, beats, pitch, spectral features."""

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AnalysisResult:
    """Complete analysis of an audio file."""

    bpm: float
    beat_times: np.ndarray  # seconds
    onset_times: np.ndarray  # seconds
    onset_strengths: np.ndarray
    key: str  # e.g. "C major", "Am"
    pitch_track: np.ndarray  # Hz per frame
    rms_envelope: np.ndarray
    spectral_centroid: np.ndarray
    duration: float  # seconds
    sr: int


def estimate_key(y: np.ndarray, sr: int) -> str:
    """Estimate musical key using chroma features and Krumhansl-Schmuckler algorithm."""
    import librosa

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Krumhansl-Schmuckler major and minor profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    best_corr = -1.0
    best_key = "C major"

    for i in range(12):
        rotated = np.roll(chroma_mean, -i)
        corr_major = np.corrcoef(rotated, major_profile)[0, 1]
        corr_minor = np.corrcoef(rotated, minor_profile)[0, 1]

        if corr_major > best_corr:
            best_corr = corr_major
            best_key = f"{note_names[i]} major"
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = f"{note_names[i]} minor"

    return best_key


_analysis_cache: dict[str, tuple[float, AnalysisResult]] = {}
_cache_lock = threading.Lock()


def analyze(audio_path: Path, sr: int = 44100) -> AnalysisResult:
    """Perform full analysis on an audio file. Results are cached by path+mtime."""
    import librosa

    # Cache check: skip re-analysis if file unchanged
    cache_key = str(audio_path)
    mtime = audio_path.stat().st_mtime
    with _cache_lock:
        if cache_key in _analysis_cache:
            cached_mtime, cached_result = _analysis_cache[cache_key]
            if cached_mtime == mtime:
                return cached_result

    y, loaded_sr = librosa.load(str(audio_path), sr=sr, mono=True)
    duration = len(y) / sr

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_strengths = onset_env[onset_frames] if len(onset_frames) > 0 else np.array([])

    # Key estimation
    key = estimate_key(y, sr)

    # Pitch tracking
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        pitch_track = np.nan_to_num(f0, nan=0.0)
    except Exception:
        pitch_track = np.zeros(len(y) // 512)

    # RMS envelope
    rms = librosa.feature.rms(y=y)[0]

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    result = AnalysisResult(
        bpm=bpm,
        beat_times=beat_times,
        onset_times=onset_times,
        onset_strengths=onset_strengths,
        key=key,
        pitch_track=pitch_track,
        rms_envelope=rms,
        spectral_centroid=centroid,
        duration=duration,
        sr=sr,
    )

    # Cache the result
    with _cache_lock:
        _analysis_cache[cache_key] = (mtime, result)
    return result
