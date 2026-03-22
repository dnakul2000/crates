"""7-mode chopping engine — the algorithmic heart of GrabStem.

Translates librosa analysis into preset-driven audio slicing.
"""

from dataclasses import dataclass

import numpy as np

from .analyzer import AnalysisResult


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
    if hz <= 0:
        return 0
    return int(round(69 + 12 * np.log2(hz / 440.0)))


def _midi_to_name(midi: int) -> str:
    """Convert MIDI note number to note name."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    if midi <= 0:
        return ""
    octave = (midi // 12) - 1
    note = names[midi % 12]
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
    rms_threshold = np.percentile(analysis.rms_envelope, (1 - sensitivity) * 80)

    import librosa

    hop_length = 512
    points = [0.0]
    for t in analysis.onset_times:
        frame = int(t * analysis.sr / hop_length)
        if frame < len(analysis.rms_envelope) and analysis.rms_envelope[frame] >= rms_threshold:
            points.append(t)
    points.append(duration)
    return sorted(set(points))


def _get_chop_points_granular(grain_size_ms: float, duration: float) -> list[float]:
    """Uniform micro-slices."""
    grain_s = grain_size_ms / 1000.0
    points = list(np.arange(0.0, duration, grain_s))
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
    threshold = np.percentile(analysis.onset_strengths, (1 - sensitivity) * 70)
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
        offset = swing_amount * intensity * (beat_duration / 4)
        if i % 2 == 0:
            offset = -offset
        new_t = points[i] + offset
        # Ensure we don't go before previous or after next
        new_t = max(result[-1] + 0.01, min(new_t, points[i + 1] - 0.01))
        result.append(new_t)
    result.append(points[-1])
    return result


def classify_chop(
    audio: np.ndarray, sr: int, stem_type: str, duration_s: float
) -> str:
    """Classify a chop based on spectral features and stem type."""
    import librosa

    if len(audio) < 512:
        return f"{stem_type.lower()}_fragment"

    # Use n_fft that fits the audio length
    n_fft = min(2048, len(audio))

    # Compute features
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft)))
    rms = float(np.mean(librosa.feature.rms(y=audio, frame_length=n_fft)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio, frame_length=n_fft)))

    if stem_type == "Drums":
        # Low centroid + high energy = kick
        if centroid < 1500 and rms > 0.05:
            return "kick"
        # Mid centroid + sharp transient = snare
        elif centroid < 5000 and rms > 0.03:
            return "snare"
        # High centroid + low energy = hat
        elif centroid > 5000 and rms < 0.05:
            return "hihat"
        # High centroid + sustained = cymbal
        elif centroid > 5000 and duration_s > 0.3:
            return "cymbal"
        else:
            return "percussion"

    elif stem_type == "Vocals":
        if duration_s > 2.0:
            return "vocal_phrase"
        elif duration_s > 0.5:
            return "vocal_word"
        elif duration_s > 0.1 and rms > 0.01:
            return "vocal_syllable"
        elif rms < 0.005:
            return "breath"
        else:
            return "vocal_chop"

    elif stem_type == "Bass":
        if duration_s > 1.0:
            return "bass_riff"
        elif zcr > 0.1:
            return "bass_slide"
        else:
            return "bass_note"

    else:  # Other
        if centroid > 3000 and duration_s < 0.5:
            return "melody_stab"
        elif centroid > 3000:
            return "melody_phrase"
        elif duration_s > 2.0:
            return "texture"
        elif rms < 0.005:
            return "noise"
        else:
            return "chord"


def estimate_chop_pitch(audio: np.ndarray, sr: int) -> tuple[float | None, int | None, str | None]:
    """Estimate the fundamental pitch of a chop."""
    import librosa

    if len(audio) < 2048:
        return None, None, None

    try:
        f0, voiced, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=min(2048, len(audio)),
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


def chop(
    audio: np.ndarray,
    sr: int,
    analysis: AnalysisResult,
    stem_type: str,
    chop_config: dict,
    intensity: float = 0.75,
) -> list[ChopResult]:
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
    """
    duration = len(audio) / sr
    mode = chop_config.get("mode", "onset")

    # Scale sensitivity by intensity
    raw_sensitivity = chop_config.get("onset_sensitivity", 0.5)
    sensitivity = raw_sensitivity * (0.5 + 0.5 * intensity)  # range: 50%-100% of raw

    # Get chop points based on mode
    if mode == "onset":
        points = _get_chop_points_onset(analysis, sensitivity, duration)
    elif mode == "beat_grid":
        points = _get_chop_points_beat_grid(
            analysis, chop_config.get("beat_division", 1.0), duration
        )
    elif mode == "phrase":
        points = _get_chop_points_phrase(
            analysis, chop_config.get("phrase_beats", 4), duration
        )
    elif mode == "transient":
        points = _get_chop_points_transient(analysis, sensitivity, duration)
    elif mode == "granular":
        points = _get_chop_points_granular(
            chop_config.get("grain_size_ms", 50.0), duration
        )
    elif mode == "syllable":
        points = _get_chop_points_syllable(analysis, sensitivity, duration)
    elif mode == "random":
        points = _get_chop_points_random(
            chop_config.get("min_chop_ms", 100),
            chop_config.get("max_chop_ms", 2000),
            duration,
        )
    else:
        points = [0.0, duration]

    # Apply swing
    swing = chop_config.get("swing_amount", 0.0)
    if swing > 0 and len(analysis.beat_times) >= 2:
        beat_dur = float(np.median(np.diff(analysis.beat_times)))
        points = _apply_swing(points, swing, intensity, beat_dur)

    # Filter by duration constraints
    min_ms = chop_config.get("min_chop_ms", 100)
    max_ms = chop_config.get("max_chop_ms", 8000)
    min_s = min_ms / 1000.0
    max_s = max_ms / 1000.0

    # Slice audio and create ChopResults
    import librosa

    crossfade_ms = chop_config.get("crossfade_ms", 5.0)
    crossfade_samples = int(sr * crossfade_ms / 1000)
    silence_db = chop_config.get("silence_threshold_db", -40)

    results = []
    for i in range(len(points) - 1):
        start_t = points[i]
        end_t = points[i + 1]
        chop_duration = end_t - start_t

        if chop_duration < min_s or chop_duration > max_s:
            continue

        start_sample = int(start_t * sr)
        end_sample = int(end_t * sr)
        segment = audio[start_sample:end_sample].copy()

        if len(segment) < 256:
            continue

        # Trim silence
        try:
            trimmed, trim_idx = librosa.effects.trim(segment, top_db=abs(silence_db))
            if len(trimmed) < 128:
                continue
            segment = trimmed
        except Exception:
            pass

        # Snap boundaries to zero crossings and apply cosine crossfade
        if crossfade_samples > 0 and len(segment) > crossfade_samples * 2:
            # Snap start to nearest zero crossing within ±2ms
            snap_window = min(int(sr * 0.002), len(segment) // 4)
            if snap_window > 1:
                search_region = segment[:snap_window]
                zero_crossings = np.where(np.diff(np.signbit(search_region)))[0]
                if len(zero_crossings) > 0:
                    # Trim to nearest zero crossing at start
                    segment = segment[zero_crossings[0]:]

            # Snap end to nearest zero crossing within ±2ms
            if snap_window > 1 and len(segment) > snap_window:
                search_region = segment[-snap_window:]
                zero_crossings = np.where(np.diff(np.signbit(search_region)))[0]
                if len(zero_crossings) > 0:
                    trim_point = len(segment) - snap_window + zero_crossings[-1] + 1
                    segment = segment[:trim_point]

            # Cosine (Hann) fade curves instead of linear
            if len(segment) > crossfade_samples * 2:
                fade_in = 0.5 * (1 - np.cos(np.pi * np.arange(crossfade_samples) / crossfade_samples))
                fade_out = 0.5 * (1 + np.cos(np.pi * np.arange(crossfade_samples) / crossfade_samples))
                segment[:crossfade_samples] *= fade_in
                segment[-crossfade_samples:] *= fade_out

        # Classify using multi-feature analyzer
        from .classifier import classify_chop as classify_chop_ml
        seg_duration = len(segment) / sr
        classification = classify_chop_ml(segment, sr, stem_type, seg_duration)

        # Estimate pitch: use per-chop pyin for chops >100ms, fall back to track
        if len(segment) > int(sr * 0.1):
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

        results.append(ChopResult(
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
            source_key=analysis.key if hasattr(analysis, 'key') else None,
            beat_length=beat_length,
        ))

    # Quality gate: filter out bad chops
    from .quality import QualityGate
    gate = QualityGate(silence_threshold_db=silence_db)
    filtered = []
    for chop_result in results:
        score = gate.evaluate(chop_result.audio, chop_result.sr)
        if score.passed:
            filtered.append(chop_result)
    return filtered if filtered else results[:max(1, len(results) // 2)]  # never return empty


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

    hop = 512
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
