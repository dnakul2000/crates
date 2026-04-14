"""7-mode chopping engine — the algorithmic heart of Crates.

Translates librosa analysis into preset-driven audio slicing.
"""

from collections.abc import Generator
from dataclasses import dataclass
from typing import Final

import numpy as np

from .analyzer import AnalysisResult
from .classifier import classify_chop as classify_chop_ml


# =============================================================================
# Constants
# =============================================================================

# Chop duration constraints (milliseconds)
DEFAULT_MIN_CHOP_MS: Final[float] = 100.0
DEFAULT_MAX_CHOP_MS: Final[float] = 8000.0
DEFAULT_GRAIN_SIZE_MS: Final[float] = 50.0

# Audio thresholds
DEFAULT_SILENCE_THRESHOLD_DB: Final[float] = -40.0
DEFAULT_CROSSFADE_MS: Final[float] = 5.0

# Zero-crossing detection window (seconds)
ZERO_CROSSING_WINDOW_S: Final[float] = 0.002  # 2ms snap window

# Classification thresholds (seconds)
MIN_CLASSIFICATION_DURATION_S: Final[float] = 0.1
MIN_PITCH_ESTIMATE_SAMPLES: Final[int] = 2048
MIN_CHOP_CLASSIFICATION_SAMPLES: Final[int] = 512

# Pitch detection range
PITCH_MIN_NOTE: Final[str] = "C2"
PITCH_MAX_NOTE: Final[str] = "C7"

# Feature extraction defaults
DEFAULT_N_FFT: Final[int] = 2048
MIN_N_FFT: Final[int] = 64
HOP_LENGTH: Final[int] = 512

# Swing calculation
BEAT_DIVISION_FOR_SWING: Final[int] = 4
SWING_SCALE_MIN: Final[float] = 0.5
SWING_SCALE_MAX: Final[float] = 1.0

# Quality gate relaxation
QUALITY_RELAXATION_DB: Final[float] = 6.0

# Adaptive fade minimums
MIN_FADE_SAMPLES: Final[int] = 8
MIN_SEGMENT_SAMPLES_FOR_FADE: Final[int] = 16
FADE_LENGTH_DIVISOR: Final[int] = 4

# Crossfade boundary constraints (seconds)
MIN_CHOP_GAP_S: Final[float] = 0.01

# MIDI conversion constants
MIDI_A4: Final[int] = 69
HZ_A4: Final[float] = 440.0
SEMITONES_PER_OCTAVE: Final[int] = 12
OCTAVE_OFFSET: Final[int] = 1
MIDI_NOTE_MIN: Final[int] = 0
MIDI_NOTE_MAX: Final[int] = 127

# Sensitivity scaling
RMS_PERCENTILE_MAX: Final[float] = 80.0
SYLLABLE_PERCENTILE: Final[float] = 70.0

# Classification feature thresholds
CENTROID_LOW_MAX: Final[float] = 1500.0
CENTROID_MID_MAX: Final[float] = 5000.0
RMS_DRUM_KICK_MIN: Final[float] = 0.05
RMS_DRUM_SNARE_MIN: Final[float] = 0.03
RMS_DRUM_HAT_MAX: Final[float] = 0.05
DURATION_CYMBAL_MIN: Final[float] = 0.3
DURATION_VOCAL_PHRASE_MIN: Final[float] = 2.0
DURATION_VOCAL_WORD_MIN: Final[float] = 0.5
DURATION_VOCAL_SYLLABLE_MIN: Final[float] = 0.1
RMS_VOCAL_SYLLABLE_MIN: Final[float] = 0.01
RMS_BREATH_MAX: Final[float] = 0.005
ZCR_BASS_SLIDE_MIN: Final[float] = 0.1
DURATION_BASS_RIFF_MIN: Final[float] = 1.0
DURATION_TEXTURE_MIN: Final[float] = 2.0
CENTROID_MELODY_MIN: Final[float] = 3000.0
DURATION_MELODY_STAB_MAX: Final[float] = 0.5
RMS_NOISE_MAX: Final[float] = 0.005

# Auto chop mode mapping
AUTO_CHOP_MODES: dict[str, str] = {
    "Drums": "onset",
    "Vocals": "syllable",
    "Bass": "beat_grid",
    "Guitar": "phrase",
    "Piano": "phrase",
    "Other": "onset",
}

DEFAULT_CHOP_MODE: Final[str] = "onset"
DEFAULT_BEAT_DIVISION: Final[float] = 1.0
DEFAULT_PHRASE_BEATS: Final[int] = 4
DEFAULT_ONSET_SENSITIVITY: Final[float] = 0.5
DEFAULT_SWING_AMOUNT: Final[float] = 0.0
DEFAULT_INTENSITY: Final[float] = 0.75
SENSITIVITY_SCALE_BASE: Final[float] = 0.5
SENSITIVITY_SCALE_RANGE: Final[float] = 0.5
MIN_TRIM_SAMPLES: Final[int] = 128
MIN_SEGMENT_LENGTH: Final[int] = 256

# Random chop defaults
DEFAULT_MIN_CHOP_RANDOM_MS: Final[float] = 100.0
DEFAULT_MAX_CHOP_RANDOM_MS: Final[float] = 2000.0


@dataclass
class ChopResult:
    """A single chopped sample."""

    audio: np.ndarray
    sr: int
    start_time: float
    end_time: float
    source_stem: str  # "Vocals", "Drums", "Bass", "Other"
    pitch_hz: float | None  # estimated fundamental frequency
    pitch_midi: int | None  # MIDI note number
    pitch_name: str | None  # e.g. "C4"
    classification: str  # e.g. "kick", "snare", "vocal_phrase"
    temporal_group: int = -1  # chops with overlapping time ranges share a group
    source_bpm: float | None = None  # BPM of the source track
    source_key: str | None = None  # key of the source track (e.g. "C major")
    beat_length: float | None = None  # duration in beats (e.g. 1.0 = quarter note)


def _hz_to_midi(hz: float) -> int:
    """Convert frequency in Hz to MIDI note number."""
    if hz <= 0 or not np.isfinite(hz):
        return MIDI_NOTE_MIN
    try:
        midi = int(round(MIDI_A4 + SEMITONES_PER_OCTAVE * np.log2(hz / HZ_A4)))
        # MIDI note range is 0-127
        return max(MIDI_NOTE_MIN, min(MIDI_NOTE_MAX, midi))
    except (ValueError, OverflowError):
        return MIDI_NOTE_MIN


def _midi_to_name(midi: int) -> str:
    """Convert MIDI note number to note name."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    if midi <= MIDI_NOTE_MIN or midi > MIDI_NOTE_MAX:
        return ""
    octave = (midi // SEMITONES_PER_OCTAVE) - OCTAVE_OFFSET
    note = names[midi % SEMITONES_PER_OCTAVE]
    return f"{note}{octave}"


def _get_chop_points_onset(
    analysis: AnalysisResult, sensitivity: float, duration: float
) -> list[float]:
    """Onset-based chop points with adjustable sensitivity."""
    import librosa

    if len(analysis.onset_strengths) == 0:
        return [0.0, duration]

    # Filter by strength threshold (lower sensitivity = fewer chops)
    threshold = np.percentile(analysis.onset_strengths, (1 - sensitivity) * 100)
    mask = analysis.onset_strengths >= threshold
    times = analysis.onset_times[: len(mask)][mask]
    points = [0.0] + list(times) + [duration]
    return sorted(set(points))


def _get_chop_points_beat_grid(
    analysis: AnalysisResult, beat_division: float, duration: float
) -> list[float]:
    """Beat-grid aligned chop points."""
    if len(analysis.beat_times) < 2:
        return [0.0, duration]

    beat_interval = np.median(np.diff(analysis.beat_times))
    sub_interval = beat_interval * beat_division

    points = [0.0]
    t = analysis.beat_times[0] if len(analysis.beat_times) > 0 else 0.0
    while t < duration:
        points.append(t)
        t += sub_interval
    points.append(duration)
    return sorted(set(points))


def _get_chop_points_phrase(
    analysis: AnalysisResult, phrase_beats: int, duration: float
) -> list[float]:
    """Group beats into N-beat phrases."""
    if len(analysis.beat_times) < 2:
        return [0.0, duration]

    points = [0.0]
    for i in range(0, len(analysis.beat_times), phrase_beats):
        points.append(analysis.beat_times[i])
    points.append(duration)
    return sorted(set(points))


def _get_chop_points_transient(
    analysis: AnalysisResult, sensitivity: float, duration: float
) -> list[float]:
    """Transient detection — onset with energy gating."""
    if len(analysis.onset_times) == 0:
        return [0.0, duration]

    # Use RMS envelope to gate: only keep onsets at high-energy moments
    rms_threshold = np.percentile(
        analysis.rms_envelope, (1 - sensitivity) * RMS_PERCENTILE_MAX
    )

    points = [0.0]
    for t in analysis.onset_times:
        frame = int(t * analysis.sr / HOP_LENGTH)
        if (
            frame < len(analysis.rms_envelope)
            and analysis.rms_envelope[frame] >= rms_threshold
        ):
            points.append(t)
    points.append(duration)
    return sorted(set(points))


def _get_chop_points_granular(grain_size_ms: float, duration: float) -> list[float]:
    """Uniform micro-slices."""
    grain_s = grain_size_ms / 1000.0
    points: list[float] = [float(x) for x in np.arange(0.0, duration, grain_s)]
    points.append(duration)
    return points


def _get_chop_points_syllable(
    analysis: AnalysisResult, sensitivity: float, duration: float
) -> list[float]:
    """Onset detection tuned for vocal syllable boundaries using spectral flux."""
    # Use onset detection with spectral flux for better vocal boundary detection
    if len(analysis.onset_times) == 0:
        return [0.0, duration]

    # Higher spectral centroid moments often correspond to consonant onsets (syllable starts)
    threshold = np.percentile(
        analysis.onset_strengths, (1 - sensitivity) * SYLLABLE_PERCENTILE
    )
    mask = analysis.onset_strengths >= threshold
    times = analysis.onset_times[: len(mask)][mask]
    points = [0.0] + list(times) + [duration]
    return sorted(set(points))


def _get_chop_points_random(
    min_ms: float, max_ms: float, duration: float, seed: int = 42
) -> list[float]:
    """Random slicing within min/max constraints."""
    rng = np.random.RandomState(seed)
    points = [0.0]
    t = 0.0
    while t < duration:
        gap = rng.uniform(min_ms / 1000, max_ms / 1000)
        t += gap
        if t < duration:
            points.append(t)
    points.append(duration)
    return points


def _apply_swing(
    points: list[float], swing_amount: float, intensity: float, beat_duration: float
) -> list[float]:
    """Apply MPC-style swing to chop points."""
    if swing_amount <= 0 or intensity <= 0:
        return points

    result = [points[0]]
    for i in range(1, len(points) - 1):
        offset = swing_amount * intensity * (beat_duration / BEAT_DIVISION_FOR_SWING)
        if i % 2 == 0:
            offset = -offset
        new_t = points[i] + offset
        # Ensure we don't go before previous or after next
        new_t = max(
            result[-1] + MIN_CHOP_GAP_S, min(new_t, points[i + 1] - MIN_CHOP_GAP_S)
        )
        result.append(new_t)
    result.append(points[-1])
    return result


def classify_chop(audio: np.ndarray, sr: int, stem_type: str, duration_s: float) -> str:
    """Classify a chop based on spectral features and stem type."""
    import librosa

    if len(audio) < MIN_CHOP_CLASSIFICATION_SAMPLES:
        return f"{stem_type.lower()}_fragment"

    # Use n_fft that fits the audio length
    n_fft = min(DEFAULT_N_FFT, len(audio))

    # Compute features
    centroid = float(
        np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft))
    )
    rms = float(np.mean(librosa.feature.rms(y=audio, frame_length=n_fft)))
    zcr = float(
        np.mean(librosa.feature.zero_crossing_rate(y=audio, frame_length=n_fft))
    )

    if stem_type == "Drums":
        # Low centroid + high energy = kick
        if centroid < CENTROID_LOW_MAX and rms > RMS_DRUM_KICK_MIN:
            return "kick"
        # Mid centroid + sharp transient = snare
        elif centroid < CENTROID_MID_MAX and rms > RMS_DRUM_SNARE_MIN:
            return "snare"
        # High centroid + low energy = hat
        elif centroid > CENTROID_MID_MAX and rms < RMS_DRUM_HAT_MAX:
            return "hihat"
        # High centroid + sustained = cymbal
        elif centroid > CENTROID_MID_MAX and duration_s > DURATION_CYMBAL_MIN:
            return "cymbal"
        else:
            return "percussion"

    elif stem_type == "Vocals":
        if duration_s > DURATION_VOCAL_PHRASE_MIN:
            return "vocal_phrase"
        elif duration_s > DURATION_VOCAL_WORD_MIN:
            return "vocal_word"
        elif duration_s > DURATION_VOCAL_SYLLABLE_MIN and rms > RMS_VOCAL_SYLLABLE_MIN:
            return "vocal_syllable"
        elif rms < RMS_BREATH_MAX:
            return "breath"
        else:
            return "vocal_chop"

    elif stem_type == "Bass":
        if duration_s > DURATION_BASS_RIFF_MIN:
            return "bass_riff"
        elif zcr > ZCR_BASS_SLIDE_MIN:
            return "bass_slide"
        else:
            return "bass_note"

    else:  # Other
        if centroid > CENTROID_MELODY_MIN and duration_s < DURATION_MELODY_STAB_MAX:
            return "melody_stab"
        elif centroid > CENTROID_MELODY_MIN:
            return "melody_phrase"
        elif duration_s > DURATION_TEXTURE_MIN:
            return "texture"
        elif rms < RMS_NOISE_MAX:
            return "noise"
        else:
            return "chord"


def estimate_chop_pitch(
    audio: np.ndarray, sr: int
) -> tuple[float | None, int | None, str | None]:
    """Estimate the fundamental pitch of a chop."""
    import librosa

    if len(audio) < MIN_PITCH_ESTIMATE_SAMPLES:
        return None, None, None

    try:
        f0, voiced, _ = librosa.pyin(
            audio,
            fmin=float(librosa.note_to_hz(PITCH_MIN_NOTE)),
            fmax=float(librosa.note_to_hz(PITCH_MAX_NOTE)),
            sr=sr,
            frame_length=min(DEFAULT_N_FFT, len(audio)),
        )
        # Take median of voiced frames
        voiced_f0 = f0[~np.isnan(f0)]
        if len(voiced_f0) == 0:
            return None, None, None
        hz = float(np.median(voiced_f0))
        midi = _hz_to_midi(hz)
        name = _midi_to_name(midi)
        return hz, midi, name
    except Exception:
        return None, None, None


def _get_auto_chop_mode(stem_type: str) -> str:
    """Select the best chop mode for a given stem type."""
    return AUTO_CHOP_MODES.get(stem_type, DEFAULT_CHOP_MODE)


def chop(
    audio: np.ndarray,
    sr: int,
    analysis: AnalysisResult,
    stem_type: str,
    chop_config: dict,
    intensity: float = DEFAULT_INTENSITY,
    yield_chops: bool = False,
) -> list[ChopResult] | Generator[ChopResult, None, None] | None:
    """Main chopping function. Takes audio + analysis + preset config, returns chops.

    chop_config keys:
        mode: str — one of onset, beat_grid, phrase, transient, granular, syllable, random
        onset_sensitivity: float (0-1)
        beat_division: float (1.0 = quarter, 0.5 = eighth, 0.25 = sixteenth)
        phrase_beats: int
        grain_size_ms: float
        min_chop_ms: float
        max_chop_ms: float
        swing_amount: float (0-1)
        silence_threshold_db: float
        crossfade_ms: float

    Args:
        yield_chops: If True, yields chops as generator for memory-efficient streaming.
                     If False (default), returns a list of all chops.
    """
    # Guard against empty audio
    if len(audio) == 0 or sr <= 0:
        if yield_chops:
            return None
        return []

    duration = len(audio) / sr
    mode = chop_config.get("mode", DEFAULT_CHOP_MODE)

    # Auto mode: select best chop mode per stem type
    if mode == "auto":
        mode = _get_auto_chop_mode(stem_type)

    # Scale sensitivity by intensity
    raw_sensitivity = chop_config.get("onset_sensitivity", DEFAULT_ONSET_SENSITIVITY)
    sensitivity = raw_sensitivity * (
        SENSITIVITY_SCALE_BASE + SENSITIVITY_SCALE_RANGE * intensity
    )

    # Get chop points based on mode
    if mode == "onset":
        points = _get_chop_points_onset(analysis, sensitivity, duration)
    elif mode == "beat_grid":
        points = _get_chop_points_beat_grid(
            analysis, chop_config.get("beat_division", DEFAULT_BEAT_DIVISION), duration
        )
    elif mode == "phrase":
        points = _get_chop_points_phrase(
            analysis, chop_config.get("phrase_beats", DEFAULT_PHRASE_BEATS), duration
        )
    elif mode == "transient":
        points = _get_chop_points_transient(analysis, sensitivity, duration)
    elif mode == "granular":
        points = _get_chop_points_granular(
            chop_config.get("grain_size_ms", DEFAULT_GRAIN_SIZE_MS), duration
        )
    elif mode == "syllable":
        points = _get_chop_points_syllable(analysis, sensitivity, duration)
    elif mode == "random":
        points = _get_chop_points_random(
            chop_config.get("min_chop_ms", DEFAULT_MIN_CHOP_RANDOM_MS),
            chop_config.get("max_chop_ms", DEFAULT_MAX_CHOP_RANDOM_MS),
            duration,
        )
    else:
        points = [0.0, duration]

    # Apply swing
    swing = chop_config.get("swing_amount", DEFAULT_SWING_AMOUNT)
    if swing > 0 and len(analysis.beat_times) >= 2:
        beat_dur = float(np.median(np.diff(analysis.beat_times)))
        points = _apply_swing(points, swing, intensity, beat_dur)

    # Filter by duration constraints
    min_ms = chop_config.get("min_chop_ms", DEFAULT_MIN_CHOP_MS)
    max_ms = chop_config.get("max_chop_ms", DEFAULT_MAX_CHOP_MS)
    min_s = min_ms / 1000.0
    max_s = max_ms / 1000.0

    # Slice audio and create ChopResults
    import librosa

    crossfade_ms = chop_config.get("crossfade_ms", DEFAULT_CROSSFADE_MS)
    crossfade_samples = int(sr * crossfade_ms / 1000)
    silence_db = chop_config.get("silence_threshold_db", DEFAULT_SILENCE_THRESHOLD_DB)

    # Generator for memory-efficient processing
    def _generate_chops():
        for i in range(len(points) - 1):
            start_t = points[i]
            end_t = points[i + 1]
            chop_duration = end_t - start_t

            if chop_duration < min_s or chop_duration > max_s:
                continue

            start_sample = int(start_t * sr)
            end_sample = int(end_t * sr)
            segment = audio[start_sample:end_sample].copy()

            if len(segment) < MIN_SEGMENT_LENGTH:
                continue

            # Trim silence
            try:
                trimmed, trim_idx = librosa.effects.trim(
                    segment, top_db=abs(silence_db)
                )
                if len(trimmed) < MIN_TRIM_SAMPLES:
                    continue
                segment = trimmed
            except Exception:
                pass

            # Snap boundaries to zero crossings and apply cosine crossfade
            if crossfade_samples > 0 and len(segment) > MIN_SEGMENT_SAMPLES_FOR_FADE:
                # Snap start to nearest zero crossing within ±2ms
                snap_window = min(int(sr * ZERO_CROSSING_WINDOW_S), len(segment) // 4)
                if snap_window > 1:
                    search_region = segment[:snap_window]
                    zero_crossings = np.where(np.diff(np.signbit(search_region)))[0]
                    if len(zero_crossings) > 0:
                        # Trim to nearest zero crossing at start
                        segment = segment[zero_crossings[0] :]

                # Snap end to nearest zero crossing within ±2ms
                if snap_window > 1 and len(segment) > snap_window:
                    search_region = segment[-snap_window:]
                    zero_crossings = np.where(np.diff(np.signbit(search_region)))[0]
                    if len(zero_crossings) > 0:
                        trim_point = len(segment) - snap_window + zero_crossings[-1] + 1
                        segment = segment[:trim_point]

                # Adaptive cosine (Hann) fade — reduce window for short segments
                effective_fade = crossfade_samples
                if len(segment) <= crossfade_samples * 2:
                    effective_fade = max(
                        MIN_FADE_SAMPLES, len(segment) // FADE_LENGTH_DIVISOR
                    )

                if effective_fade > 0 and len(segment) > effective_fade * 2:
                    fade_in = 0.5 * (
                        1 - np.cos(np.pi * np.arange(effective_fade) / effective_fade)
                    )
                    fade_out = 0.5 * (
                        1 + np.cos(np.pi * np.arange(effective_fade) / effective_fade)
                    )
                    segment[:effective_fade] *= fade_in
                    segment[-effective_fade:] *= fade_out

            # Classify using multi-feature analyzer (imported at module level)
            seg_duration = len(segment) / sr
            classification = classify_chop_ml(segment, sr, stem_type, seg_duration)

            # Estimate pitch: use per-chop pyin for chops >100ms, fall back to track
            if len(segment) > int(sr * MIN_CLASSIFICATION_DURATION_S):
                pitch_hz, pitch_midi, pitch_name = estimate_chop_pitch(segment, sr)
            else:
                pitch_hz, pitch_midi, pitch_name = _pitch_from_track(
                    analysis.pitch_track, start_t, end_t, analysis.duration, sr
                )

            # Calculate beat length from BPM
            chop_duration_s = end_t - start_t
            beat_length = None
            if analysis.bpm > 0:
                beat_duration_s = 60.0 / analysis.bpm
                beat_length = round(chop_duration_s / beat_duration_s, 2)

            yield ChopResult(
                audio=segment,
                sr=sr,
                start_time=start_t,
                end_time=end_t,
                source_stem=stem_type,
                pitch_hz=pitch_hz,
                pitch_midi=pitch_midi,
                pitch_name=pitch_name,
                classification=classification,
                source_bpm=analysis.bpm if analysis.bpm > 0 else None,
                source_key=analysis.key if hasattr(analysis, "key") else None,
                beat_length=beat_length,
            )

    # Stream mode: return generator for memory-efficient processing
    if yield_chops:
        return _generate_chops()

    # List mode: collect all results with quality gating
    results = list(_generate_chops())

    # Quality gate: filter out bad chops (tiered — strict then relaxed)
    from .quality import QualityGate

    gate = QualityGate(silence_threshold_db=silence_db)
    filtered = []
    for chop_result in results:
        score = gate.evaluate(chop_result.audio, chop_result.sr)
        if score.passed:
            filtered.append(chop_result)
    if filtered:
        return filtered

    # Relaxed pass: 6dB more lenient on silence threshold
    relaxed_gate = QualityGate(silence_threshold_db=silence_db - QUALITY_RELAXATION_DB)
    relaxed = []
    for chop_result in results:
        score = relaxed_gate.evaluate(chop_result.audio, chop_result.sr)
        if score.passed:
            relaxed.append(chop_result)
    return relaxed  # may be empty — caller should handle this


def _pitch_from_track(
    pitch_track: np.ndarray,
    start_t: float,
    end_t: float,
    total_duration: float,
    sr: int,
) -> tuple[float | None, int | None, str | None]:
    """Extract median pitch for a time range from the pre-computed pitch track."""
    if len(pitch_track) == 0:
        return None, None, None

    hop = HOP_LENGTH
    n_frames = len(pitch_track)
    start_frame = int(start_t * sr / hop)
    end_frame = int(end_t * sr / hop)
    start_frame = max(0, min(start_frame, n_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, n_frames))

    segment = pitch_track[start_frame:end_frame]
    voiced = segment[segment > 0]
    if len(voiced) == 0:
        return None, None, None

    hz = float(np.median(voiced))
    midi = _hz_to_midi(hz)
    name = _midi_to_name(midi)
    return hz, midi, name
