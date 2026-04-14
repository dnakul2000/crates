"""Quality gate — filters out bad chops before they reach pad mapping.

Detects stem bleed, silence, digital artifacts, and spectral anomalies.
Supports per-stem-type adaptive quality thresholds.
"""

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Quality Threshold Constants
# =============================================================================

# Frame analysis
DEFAULT_FRAME_SIZE: Final[int] = 512
MIN_AUDIO_SAMPLES: Final[int] = 128
MIN_FRAMES: Final[int] = 1

# Default quality thresholds
DEFAULT_SILENCE_THRESHOLD_DB: Final[float] = -40.0
DEFAULT_MAX_SILENCE_RATIO: Final[float] = 0.8
DEFAULT_DC_OFFSET_THRESHOLD: Final[float] = 0.01
DEFAULT_CLIPPING_CONSECUTIVE: Final[int] = 3
DEFAULT_MIN_SPECTRAL_COMPLEXITY: Final[float] = 0.01
DEFAULT_MAX_SPECTRAL_COMPLEXITY: Final[float] = 0.95
DEFAULT_BLEED_CORRELATION_THRESHOLD: Final[float] = 0.7

# Feature extraction
DEFAULT_N_FFT: Final[int] = 2048
MIN_N_FFT: Final[int] = 64

# Energy threshold for normalization
MIN_AUDIO_ENERGY: Final[float] = 1e-10

# Clipping detection
FULL_SCALE_THRESHOLD: Final[float] = 0.999

# Score calculation weights
SILENCE_RATIO_PENALTY: Final[float] = 0.3
BLEED_SCORE_PENALTY: Final[float] = 0.3
CLIPPING_PENALTY: Final[float] = 0.2
SPECTRAL_COMPLEXITY_TARGET: Final[float] = 0.3
SPECTRAL_COMPLEXITY_PENALTY: Final[float] = 0.2
SCORE_MIN: Final[float] = 0.0
SCORE_MAX: Final[float] = 1.0

# Quality score baseline for failed chops
FAILED_SCORE_VERY_LOW: Final[float] = 0.1
FAILED_SCORE_LOW: Final[float] = 0.2
FAILED_SCORE_MEDIUM: Final[float] = 0.3


# =============================================================================
# Per-Stem Quality Profiles
# =============================================================================

STEM_QUALITY_PROFILES: dict[str, dict[str, float | bool]] = {
    "Drums": {
        "max_silence_ratio": 0.6,
        "min_spectral_complexity": 0.02,
        "require_transient": True,
        "min_crest_factor": 2.0,
    },
    "Vocals": {
        "max_silence_ratio": 0.7,
        "min_spectral_complexity": 0.005,
        "require_harmonic": True,
        "min_harmonic_ratio": 0.3,
    },
    "Bass": {
        "max_silence_ratio": 0.7,
        "min_spectral_complexity": 0.005,
        "require_harmonic": True,
        "max_centroid": 3000.0,
    },
    "Guitar": {
        "min_spectral_complexity": 0.01,
        "require_harmonic": True,
        "min_harmonic_ratio": 0.25,
    },
    "Piano": {
        "min_spectral_complexity": 0.008,
        "require_harmonic": True,
        "min_harmonic_ratio": 0.3,
    },
    "Other": {
        "max_silence_ratio": 0.8,
        "min_spectral_complexity": 0.005,
    },
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class QualityScore:
    """Per-chop quality assessment."""

    passed: bool
    overall_score: float  # 0-1, higher = better quality
    silence_ratio: float  # fraction of near-silent frames
    has_dc_offset: bool
    has_clipping: bool
    spectral_complexity: float  # 0-1, too low or too high = bad
    bleed_score: float  # 0-1, how much this sounds like another stem
    rejection_reason: str | None = None


# =============================================================================
# Quality Gate Class
# =============================================================================


class QualityGate:
    """Configurable quality filter for audio chops."""

    def __init__(
        self,
        silence_threshold_db: float = DEFAULT_SILENCE_THRESHOLD_DB,
        max_silence_ratio: float = DEFAULT_MAX_SILENCE_RATIO,
        dc_offset_threshold: float = DEFAULT_DC_OFFSET_THRESHOLD,
        clipping_consecutive: int = DEFAULT_CLIPPING_CONSECUTIVE,
        min_spectral_complexity: float = DEFAULT_MIN_SPECTRAL_COMPLEXITY,
        max_spectral_complexity: float = DEFAULT_MAX_SPECTRAL_COMPLEXITY,
        bleed_correlation_threshold: float = DEFAULT_BLEED_CORRELATION_THRESHOLD,
    ):
        self.silence_threshold_db = silence_threshold_db
        self.max_silence_ratio = max_silence_ratio
        self.dc_offset_threshold = dc_offset_threshold
        self.clipping_consecutive = clipping_consecutive
        self.min_spectral_complexity = min_spectral_complexity
        self.max_spectral_complexity = max_spectral_complexity
        self.bleed_correlation_threshold = bleed_correlation_threshold

    def evaluate(
        self,
        audio: np.ndarray,
        sr: int,
        other_stems_audio: list[np.ndarray] | None = None,
        stem_type: str | None = None,
    ) -> QualityScore:
        """Evaluate a chop's quality. Returns QualityScore with pass/fail.

        When stem_type is provided, applies per-stem quality thresholds
        from STEM_QUALITY_PROFILES (e.g. drums require transients, vocals
        require harmonic content).

        Args:
            audio: The audio samples to evaluate.
            sr: Sample rate of the audio.
            other_stems_audio: Optional list of other stem audio for bleed detection.
            stem_type: Optional stem type for profile-specific thresholds.

        Returns:
            QualityScore with pass/fail status and quality metrics.
        """
        # Merge stem-specific profile with instance defaults
        profile: dict[str, float | bool] = {}
        if stem_type:
            profile = STEM_QUALITY_PROFILES.get(stem_type, {})

        max_silence = float(profile.get("max_silence_ratio", self.max_silence_ratio))
        min_complexity = float(
            profile.get("min_spectral_complexity", self.min_spectral_complexity)
        )
        max_complexity = float(
            profile.get("max_spectral_complexity", self.max_spectral_complexity)
        )

        if len(audio) < MIN_AUDIO_SAMPLES:
            return QualityScore(
                passed=False,
                overall_score=FAILED_SCORE_VERY_LOW,
                silence_ratio=1.0,
                has_dc_offset=False,
                has_clipping=False,
                spectral_complexity=0.0,
                bleed_score=0.0,
                rejection_reason="too_short",
            )

        # 1. Silence check: what fraction of frames are below threshold?
        silence_linear = 10 ** (self.silence_threshold_db / 20.0)
        frame_size = DEFAULT_FRAME_SIZE
        n_frames = max(MIN_FRAMES, len(audio) // frame_size)
        silent_frames = 0
        for i in range(n_frames):
            frame = audio[i * frame_size : (i + 1) * frame_size]
            if len(frame) > 0 and np.max(np.abs(frame)) < silence_linear:
                silent_frames += 1
        silence_ratio = silent_frames / n_frames

        if silence_ratio > max_silence:
            logger.debug(
                f"Quality check failed: mostly_silent (ratio={silence_ratio:.2f})"
            )
            return QualityScore(
                passed=False,
                overall_score=FAILED_SCORE_VERY_LOW,
                silence_ratio=silence_ratio,
                has_dc_offset=False,
                has_clipping=False,
                spectral_complexity=0.0,
                bleed_score=0.0,
                rejection_reason="mostly_silent",
            )

        # 2. DC offset check
        dc_offset = abs(float(np.mean(audio)))
        has_dc_offset = dc_offset > self.dc_offset_threshold

        if has_dc_offset:
            logger.debug(f"Quality check failed: dc_offset (offset={dc_offset:.4f})")
            return QualityScore(
                passed=False,
                overall_score=FAILED_SCORE_LOW,
                silence_ratio=silence_ratio,
                has_dc_offset=True,
                has_clipping=False,
                spectral_complexity=0.0,
                bleed_score=0.0,
                rejection_reason="dc_offset",
            )

        # 3. Clipping detection: consecutive full-scale samples
        has_clipping = self._detect_clipping(audio)

        # 4. Spectral complexity (flatness)
        spectral_complexity = self._compute_spectral_complexity(audio)
        complexity_ok = min_complexity <= spectral_complexity <= max_complexity

        if not complexity_ok:
            reason = (
                "tonal_artifact"
                if spectral_complexity < min_complexity
                else "pure_noise"
            )
            logger.debug(
                f"Quality check failed: {reason} (complexity={spectral_complexity:.3f})"
            )
            return QualityScore(
                passed=False,
                overall_score=FAILED_SCORE_MEDIUM,
                silence_ratio=silence_ratio,
                has_dc_offset=has_dc_offset,
                has_clipping=has_clipping,
                spectral_complexity=spectral_complexity,
                bleed_score=0.0,
                rejection_reason=reason,
            )

        # 5. Stem bleed detection via cross-correlation
        bleed_score = 0.0
        if other_stems_audio:
            bleed_score = self._check_bleed(audio, other_stems_audio)
            if bleed_score > self.bleed_correlation_threshold:
                logger.debug(
                    f"Quality check failed: stem_bleed (score={bleed_score:.3f})"
                )
                return QualityScore(
                    passed=False,
                    overall_score=FAILED_SCORE_MEDIUM,
                    silence_ratio=silence_ratio,
                    has_dc_offset=has_dc_offset,
                    has_clipping=has_clipping,
                    spectral_complexity=spectral_complexity,
                    bleed_score=bleed_score,
                    rejection_reason="stem_bleed",
                )

        # 6. Per-stem checks
        result = self._apply_stem_checks(
            audio,
            profile,
            silence_ratio,
            has_dc_offset,
            has_clipping,
            spectral_complexity,
            bleed_score,
        )
        if result is not None:
            return result

        # Calculate overall score
        score = self._calculate_overall_score(
            silence_ratio, bleed_score, has_clipping, spectral_complexity
        )

        return QualityScore(
            passed=True,
            overall_score=score,
            silence_ratio=silence_ratio,
            has_dc_offset=has_dc_offset,
            has_clipping=has_clipping,
            spectral_complexity=spectral_complexity,
            bleed_score=bleed_score,
        )

    def _detect_clipping(self, audio: np.ndarray) -> bool:
        """Detect consecutive full-scale samples indicating clipping."""
        abs_audio = np.abs(audio)
        full_scale = abs_audio >= FULL_SCALE_THRESHOLD
        if not np.any(full_scale):
            return False

        # Count consecutive full-scale samples
        consecutive = 0
        max_consecutive = 0
        for is_clipped in full_scale:
            if is_clipped:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        return max_consecutive >= self.clipping_consecutive

    def _compute_spectral_complexity(self, audio: np.ndarray) -> float:
        """Compute spectral flatness as complexity metric."""
        try:
            import librosa

            n_fft = min(DEFAULT_N_FFT, len(audio))
            flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft)
            return float(np.mean(flatness))
        except Exception as e:
            logger.warning(f"Failed to compute spectral complexity: {e}")
            return 0.5

    def _apply_stem_checks(
        self,
        audio: np.ndarray,
        profile: dict[str, float | bool],
        silence_ratio: float,
        has_dc_offset: bool,
        has_clipping: bool,
        spectral_complexity: float,
        bleed_score: float,
    ) -> QualityScore | None:
        """Apply stem-type specific quality checks.

        Returns:
            QualityScore if check fails, None if all checks pass.
        """
        if profile.get("require_transient"):
            # Drums: check that crest factor indicates a real transient
            peak_val = float(np.max(np.abs(audio)))
            rms_val = float(np.sqrt(np.mean(audio**2)))
            crest = peak_val / max(rms_val, MIN_AUDIO_ENERGY)
            min_crest = float(profile.get("min_crest_factor", 2.0))
            if crest < min_crest:
                logger.debug(
                    f"Quality check failed: weak_transient (crest={crest:.2f})"
                )
                return QualityScore(
                    passed=False,
                    overall_score=FAILED_SCORE_MEDIUM,
                    silence_ratio=silence_ratio,
                    has_dc_offset=has_dc_offset,
                    has_clipping=has_clipping,
                    spectral_complexity=spectral_complexity,
                    bleed_score=bleed_score,
                    rejection_reason="weak_transient",
                )

        if profile.get("require_harmonic"):
            # Melodic stems: check harmonic content
            harmonic_ratio = self._compute_harmonic_ratio(audio)
            min_h = float(profile.get("min_harmonic_ratio", 0.25))
            if harmonic_ratio < min_h:
                logger.debug(
                    f"Quality check failed: low_harmonic_content (ratio={harmonic_ratio:.2f})"
                )
                return QualityScore(
                    passed=False,
                    overall_score=FAILED_SCORE_MEDIUM,
                    silence_ratio=silence_ratio,
                    has_dc_offset=has_dc_offset,
                    has_clipping=has_clipping,
                    spectral_complexity=spectral_complexity,
                    bleed_score=bleed_score,
                    rejection_reason="low_harmonic_content",
                )

        max_centroid = profile.get("max_centroid")
        if max_centroid is not None:
            # Bass: reject if centroid is too high (not actually bass content)
            centroid_mean = self._compute_spectral_centroid(audio)
            if centroid_mean > float(max_centroid):
                logger.debug(
                    f"Quality check failed: centroid_too_high ({centroid_mean:.0f} > {max_centroid})"
                )
                return QualityScore(
                    passed=False,
                    overall_score=FAILED_SCORE_MEDIUM,
                    silence_ratio=silence_ratio,
                    has_dc_offset=has_dc_offset,
                    has_clipping=has_clipping,
                    spectral_complexity=spectral_complexity,
                    bleed_score=bleed_score,
                    rejection_reason="centroid_too_high",
                )

        return None

    def _compute_harmonic_ratio(self, audio: np.ndarray) -> float:
        """Compute ratio of harmonic to percussive energy."""
        try:
            import librosa

            harmonic, percussive = librosa.effects.hpss(audio)
            h_energy = float(np.sum(harmonic**2))
            total_energy = h_energy + float(np.sum(percussive**2))
            return h_energy / max(total_energy, MIN_AUDIO_ENERGY)
        except Exception as e:
            logger.warning(f"Failed to compute harmonic ratio: {e}")
            return 0.5

    def _compute_spectral_centroid(self, audio: np.ndarray) -> float:
        """Compute mean spectral centroid."""
        try:
            import librosa

            n_fft = min(DEFAULT_N_FFT, len(audio))
            # Note: sr is not needed for just the value, but required by librosa
            # We'll use a default since we only need relative comparison
            cent = librosa.feature.spectral_centroid(y=audio, sr=22050, n_fft=n_fft)
            return float(np.mean(cent))
        except Exception as e:
            logger.warning(f"Failed to compute spectral centroid: {e}")
            return 0.0

    def _calculate_overall_score(
        self,
        silence_ratio: float,
        bleed_score: float,
        has_clipping: bool,
        spectral_complexity: float,
    ) -> float:
        """Calculate the overall quality score."""
        score = SCORE_MAX
        score -= silence_ratio * SILENCE_RATIO_PENALTY
        score -= bleed_score * BLEED_SCORE_PENALTY
        if has_clipping:
            score -= CLIPPING_PENALTY
        # Prefer mid-range spectral complexity
        score -= (
            abs(spectral_complexity - SPECTRAL_COMPLEXITY_TARGET)
            * SPECTRAL_COMPLEXITY_PENALTY
        )
        return max(SCORE_MIN, min(SCORE_MAX, score))

    def _check_bleed(self, audio: np.ndarray, other_stems: list[np.ndarray]) -> float:
        """Check cross-correlation with other stems to detect bleed."""
        max_corr = 0.0
        audio_norm = audio - np.mean(audio)
        audio_energy = np.sqrt(np.sum(audio_norm**2))
        if audio_energy < MIN_AUDIO_ENERGY:
            return 0.0

        for other in other_stems:
            if len(other) == 0:
                continue
            # Align lengths
            min_len = min(len(audio_norm), len(other))
            a = audio_norm[:min_len]
            b = other[:min_len] - np.mean(other[:min_len])
            b_energy = np.sqrt(np.sum(b**2))
            if b_energy < MIN_AUDIO_ENERGY:
                continue
            corr = abs(float(np.sum(a * b))) / (audio_energy * b_energy)
            max_corr = max(max_corr, corr)

        return max_corr
