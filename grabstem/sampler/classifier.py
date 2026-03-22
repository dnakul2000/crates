"""Multi-feature audio classifier — replaces crude spectral centroid thresholds.

Uses a rich feature vector per chop for classification:
- Drums: multi-feature heuristic with spectral shape analysis
- Vocals: voicing ratio + formant-like features
- Bass: pitch stability + harmonic content
- Other: spectral complexity + temporal envelope
"""

import numpy as np


def _extract_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Extract a comprehensive feature vector from an audio segment."""
    import librosa

    n_fft = min(2048, len(audio))
    if n_fft < 64:
        return {}

    features = {}

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft)
    features["centroid_mean"] = float(np.mean(centroid))
    features["centroid_std"] = float(np.std(centroid))

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft)
    features["rolloff_mean"] = float(np.mean(rolloff))

    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft)
    features["bandwidth_mean"] = float(np.mean(bandwidth))

    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft)
    features["flatness_mean"] = float(np.mean(flatness))

    # Spectral contrast (energy distribution across frequency bands)
    try:
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, n_bands=4)
        for i in range(min(5, contrast.shape[0])):
            features[f"contrast_{i}"] = float(np.mean(contrast[i]))
    except Exception:
        pass

    # MFCCs (timbral fingerprint)
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft)
        for i in range(13):
            features[f"mfcc_{i}"] = float(np.mean(mfccs[i]))
    except Exception:
        pass

    # Energy features
    rms = librosa.feature.rms(y=audio, frame_length=n_fft)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))
    features["rms_max"] = float(np.max(rms))

    # Crest factor (peak-to-RMS ratio — high for transient/percussive material)
    peak = float(np.max(np.abs(audio)))
    rms_val = features["rms_mean"]
    features["crest_factor"] = peak / max(rms_val, 1e-10)

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=n_fft)
    features["zcr_mean"] = float(np.mean(zcr))

    # Onset strength (transient sharpness)
    try:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        features["onset_strength_max"] = float(np.max(onset_env)) if len(onset_env) > 0 else 0
        features["onset_strength_mean"] = float(np.mean(onset_env)) if len(onset_env) > 0 else 0
    except Exception:
        features["onset_strength_max"] = 0
        features["onset_strength_mean"] = 0

    # Temporal envelope shape (attack time estimation)
    abs_audio = np.abs(audio)
    peak_idx = np.argmax(abs_audio)
    features["attack_ratio"] = peak_idx / max(len(audio), 1)  # 0 = instant attack, 1 = slow

    # Harmonic-to-noise ratio approximation
    try:
        harmonic, percussive = librosa.effects.hpss(audio)
        h_energy = float(np.sum(harmonic ** 2))
        p_energy = float(np.sum(percussive ** 2))
        total = h_energy + p_energy
        features["harmonic_ratio"] = h_energy / max(total, 1e-10)
    except Exception:
        features["harmonic_ratio"] = 0.5

    # Duration in seconds
    features["duration_s"] = len(audio) / sr

    return features


def classify_drums(features: dict[str, float]) -> str:
    """Classify a drum chop using multi-feature analysis."""
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

    # Score each drum type
    scores = {
        "kick": 0.0,
        "snare": 0.0,
        "hihat": 0.0,
        "cymbal": 0.0,
        "percussion": 0.0,
    }

    # --- Kick detection ---
    # Low centroid, high energy, fast attack, low frequency content dominates
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
    if harmonic_ratio > 0.4:  # kicks have tonal body
        scores["kick"] += 0.5
    if mfcc_1 < -50:  # low-frequency dominated
        scores["kick"] += 1.0

    # --- Snare detection ---
    # Mid centroid, high crest factor, broadband noise component
    if 1500 < centroid < 6000:
        scores["snare"] += 2.0
    if crest > 3:
        scores["snare"] += 1.0
    if bandwidth > 3000:
        scores["snare"] += 1.5
    if flatness > 0.05:  # noise component
        scores["snare"] += 1.0
    if attack_ratio < 0.1:
        scores["snare"] += 1.0
    if 0.05 < duration < 0.5:
        scores["snare"] += 0.5
    if 0.3 < harmonic_ratio < 0.7:  # mix of tonal + noise
        scores["snare"] += 1.0

    # --- Hihat detection ---
    # High centroid, high ZCR, low energy, short duration
    if centroid > 5000:
        scores["hihat"] += 2.0
    if centroid > 8000:
        scores["hihat"] += 1.0
    if zcr > 0.15:
        scores["hihat"] += 2.0
    if rms < 0.05:
        scores["hihat"] += 0.5
    if duration < 0.2:
        scores["hihat"] += 1.5
    if flatness > 0.1:  # noise-like spectrum
        scores["hihat"] += 1.0
    if harmonic_ratio < 0.3:  # mostly noise
        scores["hihat"] += 1.0

    # --- Cymbal detection ---
    # High centroid, sustained, high ZCR, broader bandwidth
    if centroid > 4000:
        scores["cymbal"] += 1.5
    if duration > 0.3:
        scores["cymbal"] += 2.0
    if zcr > 0.1:
        scores["cymbal"] += 1.0
    if bandwidth > 5000:
        scores["cymbal"] += 1.0
    if rms < 0.08:
        scores["cymbal"] += 0.5
    if features.get("rms_std", 0) > features.get("rms_mean", 0.01) * 0.5:
        scores["cymbal"] += 1.0  # decaying energy envelope

    # --- Percussion (default for drum stem) ---
    scores["percussion"] += 1.0  # base score
    if 2000 < centroid < 5000:
        scores["percussion"] += 0.5
    if 0.1 < duration < 0.5:
        scores["percussion"] += 0.5

    return max(scores, key=scores.get)


def classify_vocals(features: dict[str, float], audio: np.ndarray, sr: int) -> str:
    """Classify a vocal chop using voicing analysis."""
    duration = features.get("duration_s", 0.1)
    rms = features.get("rms_mean", 0.01)
    harmonic_ratio = features.get("harmonic_ratio", 0.5)
    centroid = features.get("centroid_mean", 2000)
    flatness = features.get("flatness_mean", 0.1)

    # Detect breaths: low energy, noise-like spectrum
    if rms < 0.008 and flatness > 0.1:
        return "breath"

    # Use voicing ratio for better syllable/word/phrase detection
    voicing_ratio = _estimate_voicing_ratio(audio, sr)

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

        # Use pyin for voicing detection
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=80, fmax=800,
            sr=sr, frame_length=1024,
        )
        if voiced_flag is None or len(voiced_flag) == 0:
            return 0.0
        return float(np.mean(voiced_flag))
    except Exception:
        return 0.0


def classify_bass(features: dict[str, float]) -> str:
    """Classify a bass chop using harmonic + pitch analysis."""
    duration = features.get("duration_s", 0.1)
    zcr = features.get("zcr_mean", 0.05)
    harmonic_ratio = features.get("harmonic_ratio", 0.5)
    centroid_std = features.get("centroid_std", 0)
    rms_std = features.get("rms_std", 0)
    rms_mean = features.get("rms_mean", 0.01)

    # Bass slide: high pitch variation + zero crossing changes
    if zcr > 0.08 and centroid_std > 200:
        return "bass_slide"

    # Bass riff: longer duration with energy variation
    if duration > 1.0 and rms_std > rms_mean * 0.3:
        return "bass_riff"

    # Default: single bass note
    return "bass_note"


def classify_other(features: dict[str, float]) -> str:
    """Classify 'Other' stem chops using spectral complexity."""
    centroid = features.get("centroid_mean", 2000)
    duration = features.get("duration_s", 0.1)
    rms = features.get("rms_mean", 0.01)
    flatness = features.get("flatness_mean", 0.1)
    harmonic_ratio = features.get("harmonic_ratio", 0.5)
    bandwidth = features.get("bandwidth_mean", 2000)
    rms_std = features.get("rms_std", 0)
    crest = features.get("crest_factor", 1)

    # Noise: very flat spectrum, low energy
    if flatness > 0.3 and rms < 0.005:
        return "noise"

    # Texture: long, relatively steady, low spectral variation
    if duration > 2.0 and rms_std < rms * 0.3:
        return "texture"

    # Chord: harmonic content, wide bandwidth, moderate duration
    if harmonic_ratio > 0.5 and bandwidth > 2000 and duration > 0.3:
        return "chord"

    # Melody stab: short, high energy, sharp transient
    if duration < 0.5 and crest > 2 and centroid > 1500:
        return "melody_stab"

    # Melody phrase: longer melodic content
    if harmonic_ratio > 0.4 and duration > 0.5:
        return "melody_phrase"

    # Fallback
    if centroid > 3000:
        return "melody_stab" if duration < 0.5 else "melody_phrase"
    elif duration > 2.0:
        return "texture"
    else:
        return "chord"


def classify_chop(
    audio: np.ndarray, sr: int, stem_type: str, duration_s: float
) -> str:
    """Classify a chop using multi-feature analysis.

    Drop-in replacement for the old spectral-centroid-only classifier.
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
    else:
        return classify_other(features)
