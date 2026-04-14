"""Multi-feature audio classifier with expanded taxonomy.

39 categories across 6 stem types. Genre-agnostic — classifies by
acoustic/timbral properties rather than genre conventions.

Stem types: Drums, Vocals, Bass, Guitar, Piano, Other
"""

import numpy as np


def _extract_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Extract a comprehensive feature vector from an audio segment.

    Optimized to compute STFT once and reuse for multiple features,
    reducing redundant FFT computations.
    """
    import librosa

    n_fft = min(2048, len(audio))
    if n_fft < 64:
        return {}

    features = {}

    # Compute STFT once and reuse for spectral features
    # This is the key optimization - librosa features were each recomputing FFT internally
    hop_length = n_fft // 4
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)

    # Spectral features - all using pre-computed magnitude spectrogram
    centroid = librosa.feature.spectral_centroid(S=mag, sr=sr)
    features["centroid_mean"] = float(np.mean(centroid))
    features["centroid_std"] = float(np.std(centroid))

    rolloff = librosa.feature.spectral_rolloff(S=mag, sr=sr)
    features["rolloff_mean"] = float(np.mean(rolloff))

    bandwidth = librosa.feature.spectral_bandwidth(S=mag, sr=sr)
    features["bandwidth_mean"] = float(np.mean(bandwidth))

    flatness = librosa.feature.spectral_flatness(S=mag)
    features["flatness_mean"] = float(np.mean(flatness))

    # Spectral contrast (energy distribution across frequency bands)
    try:
        contrast = librosa.feature.spectral_contrast(S=mag, sr=sr, n_bands=4)
        for i in range(min(5, contrast.shape[0])):
            features[f"contrast_{i}"] = float(np.mean(contrast[i]))
    except Exception:
        pass

    # MFCCs using pre-computed power spectrogram - more efficient path
    try:
        # Use melspectrogram from STFT then to MFCC for efficiency
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=128)
        mel_spectrogram = np.dot(mel_basis, mag**2)
        log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
        # Compute MFCCs from log-mel spectrogram
        mfccs = librosa.feature.mfcc(S=log_mel, n_mfcc=13)
        for i in range(13):
            features[f"mfcc_{i}"] = float(np.mean(mfccs[i]))
    except Exception:
        pass

    # Energy features using pre-computed spectrogram
    rms = librosa.feature.rms(S=mag)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))
    features["rms_max"] = float(np.max(rms))

    # Crest factor (peak-to-RMS ratio — high for transient/percussive material)
    peak = float(np.max(np.abs(audio)))
    rms_val = features["rms_mean"]
    features["crest_factor"] = peak / max(rms_val, 1e-10)

    # Zero-crossing rate (time-domain, no STFT reuse possible)
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=n_fft)
    features["zcr_mean"] = float(np.mean(zcr))

    # Onset strength using pre-computed spectrogram
    try:
        onset_env = librosa.onset.onset_strength(S=mag, sr=sr)
        features["onset_strength_max"] = (
            float(np.max(onset_env)) if len(onset_env) > 0 else 0
        )
        features["onset_strength_mean"] = (
            float(np.mean(onset_env)) if len(onset_env) > 0 else 0
        )
    except Exception:
        features["onset_strength_max"] = 0
        features["onset_strength_mean"] = 0

    # Temporal envelope shape (attack time estimation)
    abs_audio = np.abs(audio)
    peak_idx = np.argmax(abs_audio)
    features["attack_ratio"] = peak_idx / max(len(audio), 1)

    # Harmonic-to-noise ratio approximation using pre-computed STFT for HPSS
    try:
        # Apply HPSS filter in frequency domain using the computed STFT
        S_harmonic, S_percussive = librosa.decompose.hpss(S, margin=1.0)
        harmonic = librosa.istft(S_harmonic, hop_length=hop_length, length=len(audio))
        percussive = librosa.istft(
            S_percussive, hop_length=hop_length, length=len(audio)
        )
        h_energy = float(np.sum(harmonic**2))
        p_energy = float(np.sum(percussive**2))
        total = h_energy + p_energy
        features["harmonic_ratio"] = h_energy / max(total, 1e-10)
    except Exception:
        features["harmonic_ratio"] = 0.5

    # Duration in seconds
    features["duration_s"] = len(audio) / sr

    # --- New features for expanded taxonomy ---

    # Centroid trend: slope of spectral centroid over time (detects sweeps/risers)
    if centroid.shape[1] >= 4:
        x = np.arange(centroid.shape[1])
        centroid_flat = centroid.flatten()
        try:
            slope = np.polyfit(x, centroid_flat, 1)[0]
            features["centroid_slope"] = float(slope)
        except Exception:
            features["centroid_slope"] = 0.0
    else:
        features["centroid_slope"] = 0.0

    # Energy trend: slope of RMS over time (detects risers/drops)
    if rms.shape[1] >= 4:
        x = np.arange(rms.shape[1])
        rms_flat = rms.flatten()
        try:
            slope = np.polyfit(x, rms_flat, 1)[0]
            features["energy_slope"] = float(slope)
        except Exception:
            features["energy_slope"] = 0.0
    else:
        features["energy_slope"] = 0.0

    # Chroma density: number of active chroma bins (detects chords vs single notes)
    # Reuse computed chroma if available from hpss, otherwise compute
    try:
        chroma = librosa.feature.chroma_stft(S=mag, sr=sr)
        chroma_energy = np.mean(chroma, axis=1)
        # Count bins with significant energy (> 20% of max)
        threshold = 0.2 * np.max(chroma_energy) if np.max(chroma_energy) > 0 else 0
        features["chroma_density"] = float(np.sum(chroma_energy > threshold))
    except Exception:
        features["chroma_density"] = 1.0

    # Decay rate: time from peak to -20dB (detects percussive vs sustained)
    if peak > 0:
        threshold_val = peak * 0.1  # -20dB
        post_peak = abs_audio[peak_idx:]
        below = np.where(post_peak < threshold_val)[0]
        if len(below) > 0:
            features["decay_rate"] = float(below[0]) / sr
        else:
            features["decay_rate"] = float(len(post_peak)) / sr
    else:
        features["decay_rate"] = 0.0

    # Pitch stability: std of pyin pitch estimates
    try:
        f0, voiced, _ = librosa.pyin(
            audio,
            fmin=50,
            fmax=2000,
            sr=sr,
            frame_length=1024,
        )
        if f0 is not None and voiced is not None:
            voiced_f0 = f0[voiced]
            if len(voiced_f0) > 1:
                features["pitch_stability"] = float(np.std(voiced_f0))
                features["voicing_ratio"] = float(np.mean(voiced))
            else:
                features["pitch_stability"] = 0.0
                features["voicing_ratio"] = (
                    float(np.mean(voiced)) if len(voiced) > 0 else 0.0
                )
        else:
            features["pitch_stability"] = 0.0
            features["voicing_ratio"] = 0.0
    except Exception:
        features["pitch_stability"] = 0.0
        features["voicing_ratio"] = 0.0

    return features


# =============================================================================
# Drum classifier (8 categories)
# =============================================================================


def classify_drums(features: dict[str, float]) -> str:
    """Classify a drum chop: kick, snare, hihat_closed, hihat_open,
    cymbal_crash, cymbal_ride, tom, percussion."""
    centroid = features.get("centroid_mean", 2000)
    crest = features.get("crest_factor", 1)
    rolloff = features.get("rolloff_mean", 5000)
    flatness = features.get("flatness_mean", 0.1)
    bandwidth = features.get("bandwidth_mean", 2000)
    rms = features.get("rms_mean", 0.01)
    zcr = features.get("zcr_mean", 0.05)
    attack_ratio = features.get("attack_ratio", 0.1)
    duration = features.get("duration_s", 0.1)
    harmonic_ratio = features.get("harmonic_ratio", 0.5)
    mfcc_1 = features.get("mfcc_1", 0)
    decay_rate = features.get("decay_rate", 0.1)
    rms_std = features.get("rms_std", 0)
    rms_mean = features.get("rms_mean", 0.01)

    scores: dict[str, float] = {
        "kick": 0.0,
        "snare": 0.0,
        "hihat_closed": 0.0,
        "hihat_open": 0.0,
        "cymbal_crash": 0.0,
        "cymbal_ride": 0.0,
        "tom": 0.0,
        "percussion": 0.0,
    }

    # --- Kick ---
    if centroid < 2000:
        scores["kick"] += 2.0
    if centroid < 1000:
        scores["kick"] += 2.0
    if rolloff < 4000:
        scores["kick"] += 1.5
    if rms > 0.03:
        scores["kick"] += 1.0
    if attack_ratio < 0.15:
        scores["kick"] += 1.0
    if duration < 0.5:
        scores["kick"] += 0.5
    if harmonic_ratio > 0.4:
        scores["kick"] += 0.5
    if mfcc_1 < -50:
        scores["kick"] += 1.0

    # --- Snare ---
    if 1500 < centroid < 6000:
        scores["snare"] += 2.0
    if crest > 3:
        scores["snare"] += 1.0
    if bandwidth > 3000:
        scores["snare"] += 1.5
    if flatness > 0.05:
        scores["snare"] += 1.0
    if attack_ratio < 0.1:
        scores["snare"] += 1.0
    if 0.05 < duration < 0.5:
        scores["snare"] += 0.5
    if 0.3 < harmonic_ratio < 0.7:
        scores["snare"] += 1.0

    # --- Hihat closed (short, no sustain) ---
    if centroid > 5000:
        scores["hihat_closed"] += 2.0
    if centroid > 8000:
        scores["hihat_closed"] += 1.0
    if zcr > 0.15:
        scores["hihat_closed"] += 2.0
    if duration < 0.15:
        scores["hihat_closed"] += 2.0
    if flatness > 0.1:
        scores["hihat_closed"] += 1.0
    if harmonic_ratio < 0.3:
        scores["hihat_closed"] += 1.0
    if decay_rate < 0.1:
        scores["hihat_closed"] += 1.0

    # --- Hihat open (longer sustain than closed) ---
    if centroid > 5000:
        scores["hihat_open"] += 2.0
    if zcr > 0.12:
        scores["hihat_open"] += 1.5
    if 0.15 < duration < 0.5:
        scores["hihat_open"] += 2.0
    if flatness > 0.1:
        scores["hihat_open"] += 1.0
    if harmonic_ratio < 0.3:
        scores["hihat_open"] += 1.0
    if decay_rate > 0.1:
        scores["hihat_open"] += 1.0

    # --- Cymbal crash (sharp attack, long decay, very wide bandwidth) ---
    if centroid > 4000:
        scores["cymbal_crash"] += 1.5
    if duration > 0.5:
        scores["cymbal_crash"] += 2.0
    if bandwidth > 5000:
        scores["cymbal_crash"] += 1.5
    if attack_ratio < 0.1:
        scores["cymbal_crash"] += 1.0
    if rms_std > rms_mean * 0.5:
        scores["cymbal_crash"] += 1.0

    # --- Cymbal ride (sustained, moderate attack, metallic ring) ---
    if centroid > 3500:
        scores["cymbal_ride"] += 1.0
    if 0.2 < duration < 0.8:
        scores["cymbal_ride"] += 1.5
    if zcr > 0.1:
        scores["cymbal_ride"] += 1.0
    if harmonic_ratio > 0.25:
        scores["cymbal_ride"] += 1.0
    if rms_std < rms_mean * 0.5:
        scores["cymbal_ride"] += 1.0

    # --- Tom (mid centroid, pitched, resonant) ---
    if 1000 < centroid < 4000:
        scores["tom"] += 2.0
    if harmonic_ratio > 0.35:
        scores["tom"] += 1.5
    if 0.1 < duration < 0.6:
        scores["tom"] += 1.0
    if attack_ratio < 0.15:
        scores["tom"] += 1.0
    if decay_rate > 0.05:
        scores["tom"] += 0.5

    # --- Percussion (catchall) ---
    scores["percussion"] += 1.0
    if 2000 < centroid < 5000:
        scores["percussion"] += 0.5
    if 0.1 < duration < 0.5:
        scores["percussion"] += 0.5

    return max(scores, key=scores.get)


# =============================================================================
# Vocal classifier (7 categories)
# =============================================================================


def classify_vocals(features: dict[str, float], audio: np.ndarray, sr: int) -> str:
    """Classify a vocal chop: vocal_phrase, vocal_word, vocal_syllable,
    vocal_chop, vocal_harmony, vocal_fx, breath."""
    duration = features.get("duration_s", 0.1)
    rms = features.get("rms_mean", 0.01)
    flatness = features.get("flatness_mean", 0.1)
    chroma_density = features.get("chroma_density", 1)
    voicing_ratio = features.get("voicing_ratio", 0.0)

    # If voicing_ratio wasn't computed in features, estimate it
    if voicing_ratio == 0.0:
        voicing_ratio = _estimate_voicing_ratio(audio, sr)

    # Breath: low energy, noise-like spectrum
    if rms < 0.008 and flatness > 0.1:
        return "breath"

    # Vocal FX: high spectral flatness + some voicing (vocoder, heavy processing)
    if flatness > 0.2 and voicing_ratio > 0.1 and voicing_ratio < 0.5:
        return "vocal_fx"

    # Vocal harmony: multiple simultaneous pitches (high chroma density)
    if chroma_density >= 4 and voicing_ratio > 0.3 and duration > 0.3:
        return "vocal_harmony"

    # Duration-based classification with voicing gates
    if duration > 2.0 and voicing_ratio > 0.3:
        return "vocal_phrase"
    elif duration > 0.5 and voicing_ratio > 0.2:
        return "vocal_word"
    elif duration > 0.1 and voicing_ratio > 0.15:
        return "vocal_syllable"
    elif rms > 0.01:
        return "vocal_chop"
    else:
        return "breath"


def _estimate_voicing_ratio(audio: np.ndarray, sr: int) -> float:
    """Estimate the ratio of voiced frames in the audio."""
    try:
        import librosa

        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=80,
            fmax=800,
            sr=sr,
            frame_length=1024,
        )
        if voiced_flag is None or len(voiced_flag) == 0:
            return 0.0
        return float(np.mean(voiced_flag))
    except Exception:
        return 0.0


# =============================================================================
# Bass classifier (5 categories)
# =============================================================================


def classify_bass(features: dict[str, float]) -> str:
    """Classify a bass chop: bass_note, bass_riff, bass_slide, bass_pluck, bass_sub."""
    duration = features.get("duration_s", 0.1)
    zcr = features.get("zcr_mean", 0.05)
    centroid = features.get("centroid_mean", 500)
    centroid_std = features.get("centroid_std", 0)
    rms_std = features.get("rms_std", 0)
    rms_mean = features.get("rms_mean", 0.01)
    flatness = features.get("flatness_mean", 0.1)
    attack_ratio = features.get("attack_ratio", 0.1)
    decay_rate = features.get("decay_rate", 0.1)
    pitch_stability = features.get("pitch_stability", 0)

    # Bass sub: very low centroid, clean sine-like, low spectral complexity
    if centroid < 200 and flatness < 0.05:
        return "bass_sub"

    # Bass pluck: sharp attack + fast decay (synth bass, slap)
    if attack_ratio < 0.1 and decay_rate < 0.15 and duration < 0.5:
        return "bass_pluck"

    # Bass slide: pitch variation detected
    if (zcr > 0.08 and centroid_std > 200) or pitch_stability > 50:
        return "bass_slide"

    # Bass riff: longer duration with energy variation
    if duration > 1.0 and rms_std > rms_mean * 0.3:
        return "bass_riff"

    # Default: single bass note
    return "bass_note"


# =============================================================================
# Guitar classifier (5 categories)
# =============================================================================


def classify_guitar(features: dict[str, float]) -> str:
    """Classify a guitar chop: guitar_strum, guitar_pick, guitar_riff,
    guitar_chord, guitar_bend."""
    duration = features.get("duration_s", 0.1)
    bandwidth = features.get("bandwidth_mean", 2000)
    centroid = features.get("centroid_mean", 2000)
    centroid_std = features.get("centroid_std", 0)
    centroid_slope = features.get("centroid_slope", 0)
    harmonic_ratio = features.get("harmonic_ratio", 0.5)
    crest = features.get("crest_factor", 1)
    rms_std = features.get("rms_std", 0)
    rms_mean = features.get("rms_mean", 0.01)
    attack_ratio = features.get("attack_ratio", 0.1)
    chroma_density = features.get("chroma_density", 1)
    pitch_stability = features.get("pitch_stability", 0)

    # Guitar bend: pitch glide detected via centroid trend or pitch instability
    if abs(centroid_slope) > 50 or pitch_stability > 80:
        return "guitar_bend"

    # Guitar strum: wide bandwidth, rhythmic attack, multiple notes
    if bandwidth > 3000 and attack_ratio < 0.15 and chroma_density >= 3:
        return "guitar_strum"

    # Guitar pick: short, sharp attack, single note
    if duration < 0.4 and attack_ratio < 0.1 and crest > 2:
        return "guitar_pick"

    # Guitar riff: >1s, pitch/energy variation
    if duration > 1.0 and (rms_std > rms_mean * 0.3 or centroid_std > 300):
        return "guitar_riff"

    # Guitar chord: sustained, wide bandwidth, harmonic, stable pitch
    if harmonic_ratio > 0.4 and bandwidth > 2000 and duration > 0.3:
        return "guitar_chord"

    # Fallback
    return "guitar_pick" if duration < 0.5 else "guitar_chord"


# =============================================================================
# Piano classifier (4 categories)
# =============================================================================


def classify_piano(features: dict[str, float]) -> str:
    """Classify a piano chop: piano_chord, piano_melody, piano_stab, piano_phrase."""
    duration = features.get("duration_s", 0.1)
    bandwidth = features.get("bandwidth_mean", 2000)
    harmonic_ratio = features.get("harmonic_ratio", 0.5)
    chroma_density = features.get("chroma_density", 1)
    rms_std = features.get("rms_std", 0)
    rms_mean = features.get("rms_mean", 0.01)
    attack_ratio = features.get("attack_ratio", 0.1)
    crest = features.get("crest_factor", 1)

    # Piano stab: short, percussive attack, clean decay
    if duration < 0.3 and attack_ratio < 0.1 and crest > 2:
        return "piano_stab"

    # Piano chord: wide bandwidth, harmonic, multiple simultaneous pitches
    if chroma_density >= 3 and harmonic_ratio > 0.4 and bandwidth > 2000:
        return "piano_chord"

    # Piano phrase: >2s, melodic contour with dynamic variation
    if duration > 2.0 and rms_std > rms_mean * 0.2:
        return "piano_phrase"

    # Piano melody: sequential pitched notes, moderate duration
    if harmonic_ratio > 0.3 and duration > 0.3:
        return "piano_melody"

    # Fallback
    return "piano_stab" if duration < 0.5 else "piano_melody"


# =============================================================================
# Other classifier (8 categories)
# =============================================================================


def classify_other(features: dict[str, float]) -> str:
    """Classify 'Other' stem chops: pad, lead, stab, texture, sweep, riser,
    noise, chord."""
    centroid = features.get("centroid_mean", 2000)
    centroid_slope = features.get("centroid_slope", 0)
    duration = features.get("duration_s", 0.1)
    rms = features.get("rms_mean", 0.01)
    rms_std = features.get("rms_std", 0)
    flatness = features.get("flatness_mean", 0.1)
    harmonic_ratio = features.get("harmonic_ratio", 0.5)
    bandwidth = features.get("bandwidth_mean", 2000)
    crest = features.get("crest_factor", 1)
    energy_slope = features.get("energy_slope", 0)
    centroid_std = features.get("centroid_std", 0)
    chroma_density = features.get("chroma_density", 1)

    # Noise: very flat spectrum, low energy
    if flatness > 0.3 and rms < 0.005:
        return "noise"

    # Riser: rising RMS energy envelope
    if energy_slope > 0.001 and duration > 0.5:
        return "riser"

    # Sweep: rising or falling spectral centroid trend
    if abs(centroid_slope) > 80 and duration > 0.3:
        return "sweep"

    # Pad: long duration, steady energy, rich harmonics, slow spectral change
    if duration > 2.0 and rms_std < rms * 0.3 and harmonic_ratio > 0.3:
        return "pad"

    # Texture: long, evolving spectral content
    if duration > 2.0 and centroid_std > centroid * 0.15:
        return "texture"

    # Stab: short, sharp attack, harmonic content
    if duration < 0.5 and crest > 2 and harmonic_ratio > 0.3:
        return "stab"

    # Lead: clear pitch, moderate duration, prominent spectral peak
    if harmonic_ratio > 0.5 and flatness < 0.1 and 0.3 < duration < 2.0:
        return "lead"

    # Chord: wide bandwidth, harmonic, multiple pitches, stable energy
    if harmonic_ratio > 0.4 and bandwidth > 2000 and chroma_density >= 3:
        return "chord"

    # Fallback
    if duration > 2.0:
        return "texture"
    elif centroid > 3000:
        return "stab" if duration < 0.5 else "lead"
    elif harmonic_ratio > 0.4:
        return "chord"
    else:
        return "stab" if duration < 0.5 else "texture"


# =============================================================================
# Main entry point
# =============================================================================


def classify_chop(audio: np.ndarray, sr: int, stem_type: str, duration_s: float) -> str:
    """Classify a chop using multi-feature analysis.

    Supports all 6 stem types: Drums, Vocals, Bass, Guitar, Piano, Other.
    """
    if len(audio) < 512:
        return f"{stem_type.lower()}_fragment"

    features = _extract_features(audio, sr)
    if not features:
        return f"{stem_type.lower()}_fragment"

    if stem_type == "Drums":
        return classify_drums(features)
    elif stem_type == "Vocals":
        return classify_vocals(features, audio, sr)
    elif stem_type == "Bass":
        return classify_bass(features)
    elif stem_type == "Guitar":
        return classify_guitar(features)
    elif stem_type == "Piano":
        return classify_piano(features)
    else:
        return classify_other(features)
