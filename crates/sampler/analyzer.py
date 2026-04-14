"""Audio analysis using librosa — BPM, key, onsets, beats, pitch, spectral features."""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor_profile = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )

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


def analyze(
    audio_path: Path,
    sr: int = 44100,
    progress_callback: Callable[[str, float], None] | None = None,
) -> AnalysisResult:
    """Perform full analysis on an audio file. Results are cached by path+mtime.

    Args:
        audio_path: Path to the audio file to analyze
        sr: Sample rate for analysis (default: 44100)
        progress_callback: Optional callback function(step_name: str, progress: float)
            Called with analysis step name and progress (0.0-1.0) for non-blocking UIs.

    Returns:
        AnalysisResult with all computed features
    """
    import librosa

    def _emit(step: str, progress: float) -> None:
        """Emit progress if callback provided."""
        if progress_callback:
            try:
                progress_callback(step, progress)
            except Exception:
                pass  # Don't let callback errors break analysis

    cache_key = str(audio_path)

    # Atomic cache check with file validation
    with _cache_lock:
        if cache_key in _analysis_cache:
            cached_mtime, cached_result = _analysis_cache[cache_key]
            try:
                current_mtime = audio_path.stat().st_mtime
                if cached_mtime == current_mtime:
                    _emit("cached", 1.0)
                    return cached_result
            except (OSError, FileNotFoundError):
                # File removed or inaccessible, invalidate cache
                del _analysis_cache[cache_key]

    # Load file outside lock to minimize contention
    _emit("loading", 0.0)
    try:
        y, loaded_sr = librosa.load(str(audio_path), sr=sr, mono=True)
        mtime = audio_path.stat().st_mtime
    except (OSError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to load audio file: {audio_path}") from e
    duration = len(y) / sr
    _emit("loading", 1.0)

    # Beat tracking
    _emit("beat_tracking", 0.0)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    _emit("beat_tracking", 1.0)

    # Onset detection
    _emit("onset_detection", 0.0)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_strengths = onset_env[onset_frames] if len(onset_frames) > 0 else np.array([])
    _emit("onset_detection", 1.0)

    # Key estimation
    _emit("key_estimation", 0.0)
    key = estimate_key(y, sr)
    _emit("key_estimation", 1.0)

    # Pitch tracking
    _emit("pitch_tracking", 0.0)
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        pitch_track = np.nan_to_num(f0, nan=0.0)
    except Exception:
        pitch_track = np.zeros(len(y) // 512)
    _emit("pitch_tracking", 1.0)

    # RMS envelope
    _emit("rms_envelope", 0.0)
    rms = librosa.feature.rms(y=y)[0]
    _emit("rms_envelope", 1.0)

    # Spectral centroid
    _emit("spectral_features", 0.0)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    _emit("spectral_features", 1.0)

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

    # Cache the result with fresh mtime check (file may have changed during analysis)
    with _cache_lock:
        try:
            current_mtime = audio_path.stat().st_mtime
            _analysis_cache[cache_key] = (current_mtime, result)
        except (OSError, FileNotFoundError):
            # Don't cache if file is now inaccessible
            pass
    _emit("complete", 1.0)
    return result
