"""Quality gate — filters out bad chops before they reach pad mapping.

Detects stem bleed, silence, digital artifacts, and spectral anomalies.
"""

from dataclasses import dataclass

import numpy as np


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


class QualityGate:
    """Configurable quality filter for audio chops."""

    def __init__(
        self,
        silence_threshold_db: float = -40.0,
        max_silence_ratio: float = 0.8,
        dc_offset_threshold: float = 0.01,
        clipping_consecutive: int = 3,
        min_spectral_complexity: float = 0.01,
        max_spectral_complexity: float = 0.95,
        bleed_correlation_threshold: float = 0.7,
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
    ) -> QualityScore:
        """Evaluate a chop's quality. Returns QualityScore with pass/fail."""
        if len(audio) < 128:
            return QualityScore(
                passed=False, overall_score=0, silence_ratio=1.0,
                has_dc_offset=False, has_clipping=False,
                spectral_complexity=0, bleed_score=0,
                rejection_reason="too_short",
            )

        # 1. Silence check: what fraction of frames are below threshold?
        silence_linear = 10 ** (self.silence_threshold_db / 20.0)
        frame_size = 512
        n_frames = max(1, len(audio) // frame_size)
        silent_frames = 0
        for i in range(n_frames):
            frame = audio[i * frame_size : (i + 1) * frame_size]
            if len(frame) > 0 and np.max(np.abs(frame)) < silence_linear:
                silent_frames += 1
        silence_ratio = silent_frames / n_frames

        if silence_ratio > self.max_silence_ratio:
            return QualityScore(
                passed=False, overall_score=0.1, silence_ratio=silence_ratio,
                has_dc_offset=False, has_clipping=False,
                spectral_complexity=0, bleed_score=0,
                rejection_reason="mostly_silent",
            )

        # 2. DC offset check
        dc_offset = abs(float(np.mean(audio)))
        has_dc_offset = dc_offset > self.dc_offset_threshold

        if has_dc_offset:
            return QualityScore(
                passed=False, overall_score=0.2, silence_ratio=silence_ratio,
                has_dc_offset=True, has_clipping=False,
                spectral_complexity=0, bleed_score=0,
                rejection_reason="dc_offset",
            )

        # 3. Clipping detection: consecutive full-scale samples
        has_clipping = False
        abs_audio = np.abs(audio)
        full_scale = abs_audio >= 0.999
        if np.any(full_scale):
            # Count consecutive full-scale samples
            consecutive = 0
            max_consecutive = 0
            for is_clipped in full_scale:
                if is_clipped:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            has_clipping = max_consecutive >= self.clipping_consecutive

        # 4. Spectral complexity (flatness)
        try:
            import librosa
            n_fft = min(2048, len(audio))
            flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft)
            spectral_complexity = float(np.mean(flatness))
        except Exception:
            spectral_complexity = 0.5

        complexity_ok = (
            self.min_spectral_complexity <= spectral_complexity <= self.max_spectral_complexity
        )

        if not complexity_ok:
            reason = "tonal_artifact" if spectral_complexity < self.min_spectral_complexity else "pure_noise"
            return QualityScore(
                passed=False, overall_score=0.3, silence_ratio=silence_ratio,
                has_dc_offset=has_dc_offset, has_clipping=has_clipping,
                spectral_complexity=spectral_complexity, bleed_score=0,
                rejection_reason=reason,
            )

        # 5. Stem bleed detection via cross-correlation
        bleed_score = 0.0
        if other_stems_audio:
            bleed_score = self._check_bleed(audio, other_stems_audio)
            if bleed_score > self.bleed_correlation_threshold:
                return QualityScore(
                    passed=False, overall_score=0.3, silence_ratio=silence_ratio,
                    has_dc_offset=has_dc_offset, has_clipping=has_clipping,
                    spectral_complexity=spectral_complexity, bleed_score=bleed_score,
                    rejection_reason="stem_bleed",
                )

        # Calculate overall score
        score = 1.0
        score -= silence_ratio * 0.3
        score -= bleed_score * 0.3
        if has_clipping:
            score -= 0.2
        # Prefer mid-range spectral complexity
        score -= abs(spectral_complexity - 0.3) * 0.2
        score = max(0, min(1, score))

        return QualityScore(
            passed=True,
            overall_score=score,
            silence_ratio=silence_ratio,
            has_dc_offset=has_dc_offset,
            has_clipping=has_clipping,
            spectral_complexity=spectral_complexity,
            bleed_score=bleed_score,
        )

    def _check_bleed(self, audio: np.ndarray, other_stems: list[np.ndarray]) -> float:
        """Check cross-correlation with other stems to detect bleed."""
        max_corr = 0.0
        audio_norm = audio - np.mean(audio)
        audio_energy = np.sqrt(np.sum(audio_norm ** 2))
        if audio_energy < 1e-10:
            return 0.0

        for other in other_stems:
            if len(other) == 0:
                continue
            # Align lengths
            min_len = min(len(audio_norm), len(other))
            a = audio_norm[:min_len]
            b = other[:min_len] - np.mean(other[:min_len])
            b_energy = np.sqrt(np.sum(b ** 2))
            if b_energy < 1e-10:
                continue
            corr = abs(float(np.sum(a * b))) / (audio_energy * b_energy)
            max_corr = max(max_corr, corr)

        return max_corr
