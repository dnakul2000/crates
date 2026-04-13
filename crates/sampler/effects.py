"""Effects pipeline — Pedalboard wrappers + custom DSP effects.

Bridges the Pedalboard library with preset configs and implements
custom DSP effects (tape wobble, vinyl crackle, granular, etc.).
"""

import numpy as np

# Pedalboard effect builders keyed by preset effect type name.
# Each returns a pedalboard plugin instance with given params scaled by intensity.

_DRY_DEFAULTS = {
    "reverb": {"room_size": 0.0, "damping": 0.5, "wet_level": 0.0},
    "delay": {"delay_seconds": 0.0, "feedback": 0.0, "mix": 0.0},
    "saturation": {"drive_db": 0.0},
    "bitcrush": {"bit_depth": 32.0},
    "compression": {"threshold_db": 0.0, "ratio": 1.0, "attack_ms": 10.0, "release_ms": 100.0},
    "filter_lp": {"cutoff_frequency_hz": 20000.0},
    "filter_hp": {"cutoff_frequency_hz": 20.0},
    "pitch_shift": {"semitones": 0.0},
    "gain": {"gain_db": 0.0},
    "limiter": {"threshold_db": 0.0},
    "chorus": {"rate_hz": 0.0, "depth": 0.0, "mix": 0.0},
    "phaser": {"rate_hz": 0.0, "depth": 0.0, "mix": 0.0},
    "resample": {"target_sample_rate": 44100.0},
    "clipping": {"threshold_db": 0.0},
}


# Parameter interpolation curves: maps param name suffixes to curve types.
# dB params need exponential curves, frequency needs log, ratios need power curves.
_PARAM_CURVES = {
    "_db": "exponential",       # threshold_db, drive_db, gain_db
    "ratio": "power",           # compression ratio
    "frequency_hz": "log",      # cutoff_frequency_hz
    "sample_rate": "log",       # target_sample_rate
}


def _get_curve_type(param_name: str) -> str:
    """Determine the interpolation curve for a parameter based on its name."""
    for suffix, curve in _PARAM_CURVES.items():
        if suffix in param_name:
            return curve
    return "linear"


def _interpolate(dry: float, wet: float, intensity: float, param_name: str = "") -> float:
    """Interpolate between dry and wet values using perceptually appropriate curves."""
    if dry == wet or intensity == 0:
        return dry

    curve = _get_curve_type(param_name)

    if curve == "exponential":
        # Exponential for dB params — subtle at low intensity, dramatic at high
        return dry + (wet - dry) * (intensity ** 2)

    elif curve == "power":
        # Power curve for ratios — e.g. compression ratio 1:1 to 8:1
        # For ratio, interpolate as: 1 + (target_ratio - 1) * intensity^1.5
        if "ratio" in param_name and dry >= 1:
            return 1.0 + (wet - 1.0) * (intensity ** 1.5)
        return dry + (wet - dry) * (intensity ** 1.5)

    elif curve == "log":
        # Logarithmic for frequency params — perceptually even spacing
        if dry > 0 and wet > 0:
            return dry * ((wet / dry) ** intensity)
        return dry + (wet - dry) * intensity

    else:
        # Linear for mix, wet_level, depth, etc.
        return dry + (wet - dry) * intensity


def _build_pedalboard_effect(effect_type: str, params: dict, intensity: float):
    """Build a single Pedalboard effect with intensity-scaled parameters."""
    import pedalboard

    defaults = _DRY_DEFAULTS.get(effect_type, {})
    scaled = {}
    for key, dry_val in defaults.items():
        wet_val = params.get(key, dry_val)
        scaled[key] = _interpolate(dry_val, wet_val, intensity, param_name=key)

    if effect_type == "reverb":
        return pedalboard.Reverb(
            room_size=scaled.get("room_size", 0),
            damping=scaled.get("damping", 0.5),
            wet_level=scaled.get("wet_level", scaled.get("room_size", 0) * 0.5),
        )
    elif effect_type == "delay":
        return pedalboard.Delay(
            delay_seconds=scaled.get("delay_seconds", 0.3),
            feedback=scaled.get("feedback", 0.3),
            mix=scaled.get("mix", 0.3),
        )
    elif effect_type == "saturation":
        return pedalboard.Distortion(drive_db=scaled.get("drive_db", 0))
    elif effect_type == "bitcrush":
        return pedalboard.Bitcrush(bit_depth=max(2, scaled.get("bit_depth", 32)))
    elif effect_type == "compression":
        return pedalboard.Compressor(
            threshold_db=scaled.get("threshold_db", 0),
            ratio=max(1, scaled.get("ratio", 1)),
            attack_ms=scaled.get("attack_ms", 10),
            release_ms=scaled.get("release_ms", 100),
        )
    elif effect_type == "filter_lp":
        return pedalboard.LowpassFilter(
            cutoff_frequency_hz=scaled.get("cutoff_frequency_hz", 20000)
        )
    elif effect_type == "filter_hp":
        return pedalboard.HighpassFilter(
            cutoff_frequency_hz=scaled.get("cutoff_frequency_hz", 20)
        )
    elif effect_type == "pitch_shift":
        return pedalboard.PitchShift(semitones=scaled.get("semitones", 0))
    elif effect_type == "gain":
        return pedalboard.Gain(gain_db=scaled.get("gain_db", 0))
    elif effect_type == "limiter":
        return pedalboard.Limiter(threshold_db=scaled.get("threshold_db", -1))
    elif effect_type == "chorus":
        return pedalboard.Chorus(
            rate_hz=scaled.get("rate_hz", 1),
            depth=scaled.get("depth", 0.25),
            mix=scaled.get("mix", 0.3),
        )
    elif effect_type == "phaser":
        return pedalboard.Phaser(
            rate_hz=scaled.get("rate_hz", 1),
            depth=scaled.get("depth", 0.5),
            mix=scaled.get("mix", 0.3),
        )
    elif effect_type == "resample":
        return pedalboard.Resample(
            target_sample_rate=scaled.get("target_sample_rate", 44100)
        )
    elif effect_type == "clipping":
        return pedalboard.Clipping(threshold_db=scaled.get("threshold_db", -1))
    else:
        return None


# --- Custom DSP effects (not in Pedalboard) ---


def tape_wobble(audio: np.ndarray, sr: int, rate: float = 0.3, depth: float = 0.01, intensity: float = 1.0) -> np.ndarray:
    """LFO-modulated pitch wobble simulating tape machine imperfections."""
    depth *= intensity
    if depth <= 0:
        return audio

    n = len(audio)
    t = np.arange(n) / sr
    lfo = 1.0 + depth * np.sin(2 * np.pi * rate * t)

    # Resample using interpolation
    indices = np.cumsum(lfo) - lfo[0]
    indices = indices * (n - 1) / indices[-1]
    indices = np.clip(indices, 0, n - 1)
    return np.interp(np.arange(n), indices, audio).astype(np.float32)


def vinyl_crackle(
    audio: np.ndarray, sr: int,
    vinyl_intensity: float = 0.5,
    dustiness: float = 0.5,
    intensity: float = 1.0,
) -> np.ndarray:
    """Authentic vinyl emulation: damped click transients + 1/f surface noise + wow rumble.

    Args:
        vinyl_intensity: overall wet amount (0-1)
        dustiness: click density independent of surface noise (0-1)
        intensity: global effect scaling
    """
    vinyl_intensity *= intensity
    dustiness = dustiness * intensity
    if vinyl_intensity <= 0 and dustiness <= 0:
        return audio

    n = len(audio)
    # Derive seed from audio content for unique-but-reproducible crackle per chop
    import hashlib
    content_seed = int.from_bytes(
        hashlib.md5(audio[:min(1000, n)].tobytes()).digest()[:4], "little"
    ) % (2**31)
    rng = np.random.default_rng(content_seed)

    # --- Clicks: damped sinusoid bursts (realistic vinyl pops) ---
    clicks = np.zeros(n, dtype=np.float32)
    click_density = int(sr * 0.01 * dustiness)  # clicks per second
    if click_density > 0:
        click_positions = rng.choice(n, size=min(click_density, n), replace=False)
        for pos in click_positions:
            # Each click: short damped sinusoid at random frequency
            click_freq = rng.uniform(1000, 8000)
            click_dur = int(rng.uniform(0.00005, 0.0002) * sr)  # 50-200 microseconds
            click_dur = min(click_dur, n - pos)
            if click_dur < 2:
                continue
            t = np.arange(click_dur) / sr
            # Damped sinusoid with exponential decay
            decay_rate = rng.uniform(5000, 20000)
            click_signal = np.sin(2 * np.pi * click_freq * t) * np.exp(-decay_rate * t)
            # Random amplitude and polarity
            amplitude = rng.uniform(0.005, 0.025) * vinyl_intensity
            polarity = rng.choice([-1, 1])
            clicks[pos:pos + click_dur] += (click_signal * amplitude * polarity).astype(np.float32)

    # --- Surface noise: Voss-McCartney pink (1/f) noise ---
    surface = _voss_mcCartney_pink(n, rng) * 0.004 * vinyl_intensity
    # Bandpass 200Hz-6kHz for vinyl surface noise character
    from scipy.signal import butter, sosfilt
    sos_hp = butter(2, 200 / (sr / 2), btype="high", output="sos")
    sos_lp = butter(2, 6000 / (sr / 2), btype="low", output="sos")
    surface = sosfilt(sos_hp, surface).astype(np.float32)
    surface = sosfilt(sos_lp, surface).astype(np.float32)

    # --- Rumble: brownian noise with LFO wow modulation ---
    white_rumble = rng.standard_normal(n)
    brownian = np.cumsum(white_rumble) / np.sqrt(n)
    # Bandpass 20-80Hz
    sos_rumble_hp = butter(2, 20 / (sr / 2), btype="high", output="sos")
    sos_rumble_lp = butter(2, 80 / (sr / 2), btype="low", output="sos")
    rumble = sosfilt(sos_rumble_hp, brownian)
    rumble = sosfilt(sos_rumble_lp, rumble).astype(np.float32)
    # LFO modulation (turntable wow)
    wow_rate = rng.uniform(0.5, 1.0)
    t = np.arange(n) / sr
    wow_lfo = 1.0 + 0.3 * np.sin(2 * np.pi * wow_rate * t)
    rumble = (rumble * wow_lfo * 0.006 * vinyl_intensity).astype(np.float32)

    # --- Gentle high-frequency rolloff (vinyl has limited HF) ---
    result = audio + clicks + surface + rumble
    sos_hf_cut = butter(1, min(13000, sr * 0.45) / (sr / 2), btype="low", output="sos")
    result = sosfilt(sos_hf_cut, result).astype(np.float32)

    return result


def _voss_mcCartney_pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate pink noise using the Voss-McCartney algorithm.

    Sums white noise generators updated at different octave rates.
    """
    n_octaves = 12
    rows = np.zeros(n_octaves)
    output = np.zeros(n, dtype=np.float64)

    for i in range(n):
        # Determine which octave rows to update
        for octave in range(n_octaves):
            if i % (1 << octave) == 0:
                rows[octave] = rng.standard_normal()
        output[i] = np.sum(rows)

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output /= peak
    return output.astype(np.float32)


def granular_resynthesis(
    audio: np.ndarray, sr: int,
    grain_size_ms: float = 30, scatter: float = 0.5,
    density: float = 2.0,
    pitch_spread_cents: float = 0.0,
    position_jitter: float = 0.0,
    reverse_probability: float = 0.0,
    envelope: str = "hann",
    intensity: float = 1.0,
) -> np.ndarray:
    """Granular resynthesis with variation controls.

    Args:
        grain_size_ms: grain duration in milliseconds
        scatter: how much to shuffle grain order (0-1)
        density: overlap ratio (2.0 = 50% overlap, 4.0 = 75% overlap)
        pitch_spread_cents: random pitch shift per grain in +/- cents
        position_jitter: randomize grain read position (0-1 of grain size)
        reverse_probability: chance of reversing each grain (0-1)
        envelope: grain window shape ("hann", "tukey", "triangle")
        intensity: global effect scaling
    """
    scatter *= intensity
    pitch_spread_cents *= intensity
    position_jitter *= intensity
    reverse_probability *= intensity

    grain_samples = int(sr * grain_size_ms / 1000)
    if grain_samples < 64:
        grain_samples = 64

    n = len(audio)
    n_grains = n // grain_samples
    if n_grains < 2:
        return audio

    # Derive seed from audio content for unique-per-chop but reproducible results
    import hashlib
    content_seed = int.from_bytes(
        hashlib.md5(audio[:min(500, n)].tobytes()).digest()[:4], "little"
    ) % (2**31)
    rng = np.random.default_rng(content_seed)

    # Build grain envelope
    if envelope == "tukey":
        from scipy.signal.windows import tukey
        window = tukey(grain_samples, alpha=0.5).astype(np.float32)
    elif envelope == "triangle":
        window = np.bartlett(grain_samples).astype(np.float32)
    else:
        window = np.hanning(grain_samples).astype(np.float32)

    # Extract grains with optional position jitter
    grains = []
    for i in range(n_grains):
        base_start = i * grain_samples
        # Apply position jitter
        if position_jitter > 0:
            jitter = int(rng.uniform(-position_jitter, position_jitter) * grain_samples)
            start = max(0, min(base_start + jitter, n - grain_samples))
        else:
            start = base_start

        grain = audio[start : start + grain_samples].copy()
        if len(grain) < grain_samples:
            grain = np.pad(grain, (0, grain_samples - len(grain)))

        # Optional: reverse grain
        if reverse_probability > 0 and rng.random() < reverse_probability:
            grain = grain[::-1].copy()

        # Optional: pitch shift individual grain
        if pitch_spread_cents > 0:
            cents = rng.uniform(-pitch_spread_cents, pitch_spread_cents)
            ratio = 2 ** (cents / 1200.0)
            if abs(ratio - 1.0) > 0.001:
                # Simple resampling for pitch shift
                indices = np.arange(grain_samples) * ratio
                indices = np.clip(indices, 0, grain_samples - 1)
                grain = np.interp(np.arange(grain_samples), indices, grain).astype(np.float32)

        grains.append(grain * window)

    # Shuffle grains based on scatter
    n_swap = int(n_grains * scatter * 0.5)
    for _ in range(n_swap):
        i = rng.integers(0, n_grains)
        j = rng.integers(0, n_grains)
        grains[i], grains[j] = grains[j], grains[i]

    # Overlap-add with configurable density
    hop = max(1, int(grain_samples / density))
    output_len = (n_grains - 1) * hop + grain_samples
    output = np.zeros(output_len, dtype=np.float32)
    window_sum = np.zeros(output_len, dtype=np.float32)

    for i, grain in enumerate(grains):
        start = i * hop
        end = start + grain_samples
        if end > output_len:
            break
        output[start:end] += grain
        window_sum[start:end] += window

    # Normalize by window sum to prevent amplitude buildup
    window_sum = np.maximum(window_sum, 1e-8)
    output /= window_sum

    # Match original length
    if len(output) > n:
        output = output[:n]
    elif len(output) < n:
        output = np.pad(output, (0, n - len(output)))

    return output


def transient_shape(
    audio: np.ndarray, sr: int,
    attack_boost_db: float = 6.0, sustain_db: float = 0.0,
    intensity: float = 1.0,
) -> np.ndarray:
    """Shape transients by applying separate attack/sustain gain."""
    attack_boost_db *= intensity
    sustain_db *= intensity

    # Simple envelope follower
    attack_samples = int(sr * 0.01)  # 10ms attack window
    n = len(audio)

    # Detect transient region (first high-energy burst)
    rms_frame = 512
    rms = np.array([
        np.sqrt(np.mean(audio[i : i + rms_frame] ** 2))
        for i in range(0, min(n, sr), rms_frame)  # check first second
    ])

    if len(rms) == 0:
        return audio

    peak_frame = np.argmax(rms)
    transient_end = min((peak_frame + 3) * rms_frame, n)  # ~30ms after peak

    result = audio.copy()

    # Boost attack
    attack_gain = 10 ** (attack_boost_db / 20)
    result[:transient_end] *= attack_gain

    # Adjust sustain
    if sustain_db != 0:
        sustain_gain = 10 ** (sustain_db / 20)
        result[transient_end:] *= sustain_gain

    # Prevent clipping
    peak = np.max(np.abs(result))
    if peak > 1.0:
        result /= peak

    return result


def stereo_width(audio: np.ndarray, width: float = 1.0, intensity: float = 1.0) -> np.ndarray:
    """Adjust stereo width via mid-side processing. For mono, returns unchanged."""
    if audio.ndim == 1:
        return audio

    width = 1.0 + (width - 1.0) * intensity

    mid = (audio[0] + audio[1]) / 2
    side = (audio[0] - audio[1]) / 2
    side *= width

    left = mid + side
    right = mid - side
    return np.array([left, right])


def timestretch(audio: np.ndarray, sr: int, factor: float = 1.0, intensity: float = 1.0) -> np.ndarray:
    """Time-stretch without pitch change."""
    import librosa

    # Interpolate factor toward 1.0 based on intensity
    actual_factor = 1.0 + (factor - 1.0) * intensity
    if abs(actual_factor - 1.0) < 0.01:
        return audio

    return librosa.effects.time_stretch(audio, rate=actual_factor)


# Map custom effect names to their functions
_CUSTOM_EFFECTS = {
    "tape_wobble": tape_wobble,
    "vinyl_crackle": vinyl_crackle,
    "granular": granular_resynthesis,
    "transient_shape": transient_shape,
    "stereo_width": stereo_width,
    "timestretch": timestretch,
}


class EffectsPipeline:
    """Builds and applies effect chains based on preset configuration."""

    def __init__(self, effects_config: dict[str, list[dict]], intensity: float = 0.75):
        """
        effects_config: {stem_type: [{"type": "reverb", "params": {...}}, ...]}
        intensity: 0.0-1.0 scaling factor
        """
        self.effects_config = effects_config
        self.intensity = intensity

    def process(self, audio: np.ndarray, sr: int, stem_type: str) -> np.ndarray:
        """Apply the preset's effect chain for the given stem type."""
        # Get stem-specific effects
        chain = self.effects_config.get(stem_type.lower(), [])
        # Also apply global effects
        global_chain = self.effects_config.get("global", [])

        result = audio.copy()

        for effect_cfg in chain + global_chain:
            if not effect_cfg.get("enabled", True):
                continue
            result = self._apply_single(result, sr, effect_cfg)

        return result

    def _apply_single(self, audio: np.ndarray, sr: int, effect_cfg: dict) -> np.ndarray:
        """Apply a single effect (either Pedalboard or custom DSP)."""
        effect_type = effect_cfg["type"]
        params = effect_cfg.get("params", {})

        # Custom DSP effect?
        if effect_type in _CUSTOM_EFFECTS:
            func = _CUSTOM_EFFECTS[effect_type]
            return func(audio, sr, **params, intensity=self.intensity)

        # Pedalboard effect
        plugin = _build_pedalboard_effect(effect_type, params, self.intensity)
        if plugin is None:
            return audio

        import pedalboard

        board = pedalboard.Pedalboard([plugin])

        # Pedalboard expects (channels, samples) float32
        if audio.ndim == 1:
            processed = board(audio[np.newaxis, :].astype(np.float32), sr)
            return processed[0]
        else:
            return board(audio.astype(np.float32), sr)
