"""MPC pad assignment logic — 4 mapping strategies.

MPC Mini MK3: 16 pads x 8 banks (A-H) = 128 slots.
Pad layout (4x4 grid, bottom-left origin):
    13  14  15  16
     9  10  11  12
     5   6   7   8
     1   2   3   4
"""

from dataclasses import dataclass, field

import numpy as np

from .chopper import ChopResult

PADS_PER_BANK = 16
NUM_BANKS = 8
BANK_NAMES = list("ABCDEFGH")
MAX_SLOTS = PADS_PER_BANK * NUM_BANKS  # 128


@dataclass
class PadSlot:
    """A single pad assignment with optional layering."""

    chop: ChopResult  # primary sample
    pad_number: int  # 1-16
    bank: str  # A-H
    velocity_layer: int = 1
    layers: list[ChopResult] | None = None  # additional velocity/round-robin layers


@dataclass
class PadAssignment:
    """Complete pad mapping for a sample pack."""

    banks: dict[str, list[PadSlot | None]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.banks:
            for bank in BANK_NAMES:
                self.banks[bank] = [None] * PADS_PER_BANK

    @property
    def total_assigned(self) -> int:
        return sum(
            1 for bank in self.banks.values()
            for slot in bank if slot is not None
        )


def _score_chop(chop: ChopResult) -> float:
    """Score a chop for selection — multi-factor scoring.

    Considers energy, duration, onset strength, and tonal clarity.
    """
    if len(chop.audio) == 0:
        return 0.0

    rms = float(np.sqrt(np.mean(chop.audio ** 2)))
    duration_s = len(chop.audio) / chop.sr

    # Energy score (0-3): moderate energy preferred over extremes
    energy_score = min(rms * 15, 3.0)

    # Duration score (0-2): prefer 0.1-2s, penalize very short or very long
    if 0.1 <= duration_s <= 2.0:
        duration_score = 2.0
    elif duration_s < 0.1:
        duration_score = duration_s * 20  # 0-2
    else:
        duration_score = max(0, 2.0 - (duration_s - 2.0) * 0.3)

    # Musical salience (0-2): onset strength at start of chop
    try:
        import librosa
        onset_env = librosa.onset.onset_strength(y=chop.audio[:min(len(chop.audio), chop.sr)], sr=chop.sr)
        if len(onset_env) > 0:
            onset_score = min(float(onset_env[0]) / 5.0, 2.0)
        else:
            onset_score = 1.0
    except Exception:
        onset_score = 1.0

    # Tonal clarity (0-2): pitched content scores higher (useful for melodic mapping)
    pitch_score = 1.5 if chop.pitch_midi is not None and chop.pitch_midi > 0 else 0.5

    return energy_score + duration_score + onset_score + pitch_score


def _compute_spectral_signature(audio: np.ndarray, sr: int) -> np.ndarray:
    """Compute a compact spectral signature for diversity comparison."""
    import librosa
    n_fft = min(2048, len(audio))
    if n_fft < 64:
        return np.zeros(4)

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio, n_fft=n_fft)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return np.array([centroid / 10000, bandwidth / 10000, flatness, rms])


def select_diverse_chops(chops: list[ChopResult], max_count: int) -> list[ChopResult]:
    """Select chops using diversity-aware scoring (maximal marginal relevance).

    Two-pass: first guarantee minimum quota per classification,
    then fill remaining slots by spectral diversity.
    """
    if len(chops) <= max_count:
        return chops

    # Group by classification
    groups: dict[str, list[ChopResult]] = {}
    for chop in chops:
        groups.setdefault(chop.classification, []).append(chop)

    # Pass 1: guarantee minimum quota per classification (at least 4 each)
    min_per_class = max(2, min(4, max_count // max(len(groups), 1)))
    selected: list[ChopResult] = []
    selected_set: set[int] = set()

    for cls, cls_chops in groups.items():
        # Sort by score within classification, take top N
        scored = sorted(cls_chops, key=_score_chop, reverse=True)
        for chop in scored[:min_per_class]:
            idx = id(chop)
            if idx not in selected_set:
                selected.append(chop)
                selected_set.add(idx)

    # Pass 2: fill remaining by diversity (maximal marginal relevance)
    remaining = max_count - len(selected)
    if remaining <= 0:
        return selected[:max_count]

    # Compute spectral signatures for diversity
    unselected = [c for c in chops if id(c) not in selected_set]
    if not unselected:
        return selected

    # Pre-compute signatures
    selected_sigs = [_compute_spectral_signature(c.audio, c.sr) for c in selected]
    candidate_sigs = [_compute_spectral_signature(c.audio, c.sr) for c in unselected]
    candidate_scores = [_score_chop(c) for c in unselected]

    for _ in range(remaining):
        if not unselected:
            break

        best_idx = -1
        best_mmr = -float("inf")

        for i, (chop, sig, score) in enumerate(zip(unselected, candidate_sigs, candidate_scores)):
            # Quality relevance
            relevance = score / 10.0  # normalize to ~0-1

            # Diversity: min distance to any already-selected sample
            if selected_sigs:
                min_sim = min(
                    float(np.dot(sig, sel_sig)) / (np.linalg.norm(sig) * np.linalg.norm(sel_sig) + 1e-10)
                    for sel_sig in selected_sigs
                )
            else:
                min_sim = 0

            # Also consider temporal diversity
            time_diversity = 1.0
            for sel_chop in selected[-20:]:  # check recent selections
                if abs(chop.start_time - sel_chop.start_time) < 0.5:
                    time_diversity *= 0.7

            # MMR: balance relevance vs diversity
            mmr = 0.5 * relevance + 0.3 * (1 - min_sim) + 0.2 * time_diversity
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        if best_idx >= 0:
            selected.append(unselected[best_idx])
            selected_sigs.append(candidate_sigs[best_idx])
            unselected.pop(best_idx)
            candidate_sigs.pop(best_idx)
            candidate_scores.pop(best_idx)

    return selected


def _assign_to_bank(
    chops: list[ChopResult],
    assignment: PadAssignment,
    bank: str,
    start_pad: int = 0,
    max_pads: int = 16,
) -> int:
    """Assign chops to pads in a specific bank. Returns number assigned."""
    count = 0
    for i, chop in enumerate(chops):
        pad_idx = start_pad + i
        if pad_idx >= min(start_pad + max_pads, PADS_PER_BANK):
            break
        assignment.banks[bank][pad_idx] = PadSlot(
            chop=chop,
            pad_number=pad_idx + 1,
            bank=bank,
        )
        count += 1
    return count


def _group_by_classification(chops: list[ChopResult]) -> dict[str, list[ChopResult]]:
    """Group chops by their classification."""
    groups: dict[str, list[ChopResult]] = {}
    for chop in chops:
        groups.setdefault(chop.classification, []).append(chop)
    return groups


def _sort_by_key(chops: list[ChopResult], sort_by: str) -> list[ChopResult]:
    """Sort chops by the given key."""
    if sort_by == "pitch":
        return sorted(chops, key=lambda c: c.pitch_midi or 0)
    elif sort_by == "time":
        return sorted(chops, key=lambda c: c.start_time)
    elif sort_by == "energy":
        return sorted(
            chops,
            key=lambda c: float(np.sqrt(np.mean(c.audio ** 2))) if len(c.audio) > 0 else 0,
            reverse=True,
        )
    elif sort_by == "duration":
        return sorted(chops, key=lambda c: len(c.audio), reverse=True)
    elif sort_by == "random":
        rng = np.random.RandomState(42)
        indices = list(range(len(chops)))
        rng.shuffle(indices)
        return [chops[i] for i in indices]
    return chops


def map_grouped(chops: list[ChopResult], sort_by: str = "energy") -> PadAssignment:
    """Grouped strategy: same classification on adjacent pads.

    Bank A: kicks (1-4), snares (5-8), hihats (9-12), cymbals/perc (13-16)
    Bank B: drum overflow
    Bank C: vocal phrases/words (1-8), vocal syllables/chops (9-16)
    Bank D: vocal overflow
    Bank E: bass notes (sorted by pitch)
    Bank F: bass overflow
    Bank G: other/melodic chops (sorted by pitch)
    Bank H: other overflow
    """
    assignment = PadAssignment()
    groups = _group_by_classification(chops)

    # Drum classifications → Banks A-B
    drum_order = ["kick", "snare", "hihat", "cymbal", "percussion"]
    drum_pad = 0
    drum_bank_idx = 0
    for cls in drum_order:
        cls_chops = _sort_by_key(groups.get(cls, []), sort_by)
        for chop in cls_chops:
            if drum_pad >= PADS_PER_BANK:
                drum_bank_idx += 1
                drum_pad = 0
            if drum_bank_idx > 1:
                break
            bank = BANK_NAMES[drum_bank_idx]
            assignment.banks[bank][drum_pad] = PadSlot(chop=chop, pad_number=drum_pad + 1, bank=bank)
            drum_pad += 1

    # Vocal classifications → Banks C-D
    vocal_classes = ["vocal_phrase", "vocal_word", "vocal_syllable", "vocal_chop", "breath"]
    vocal_pad = 0
    vocal_bank_idx = 2
    for cls in vocal_classes:
        cls_chops = _sort_by_key(groups.get(cls, []), sort_by)
        for chop in cls_chops:
            if vocal_pad >= PADS_PER_BANK:
                vocal_bank_idx += 1
                vocal_pad = 0
            if vocal_bank_idx > 3:
                break
            bank = BANK_NAMES[vocal_bank_idx]
            assignment.banks[bank][vocal_pad] = PadSlot(chop=chop, pad_number=vocal_pad + 1, bank=bank)
            vocal_pad += 1

    # Bass → Banks E-F
    bass_classes = ["bass_note", "bass_riff", "bass_slide"]
    bass_chops = []
    for cls in bass_classes:
        bass_chops.extend(groups.get(cls, []))
    bass_chops = _sort_by_key(bass_chops, "pitch")
    bass_pad = 0
    bass_bank_idx = 4
    for chop in bass_chops:
        if bass_pad >= PADS_PER_BANK:
            bass_bank_idx += 1
            bass_pad = 0
        if bass_bank_idx > 5:
            break
        bank = BANK_NAMES[bass_bank_idx]
        assignment.banks[bank][bass_pad] = PadSlot(chop=chop, pad_number=bass_pad + 1, bank=bank)
        bass_pad += 1

    # Other → Banks G-H
    other_classes = ["chord", "melody_stab", "melody_phrase", "texture", "noise"]
    other_chops = []
    for cls in other_classes:
        other_chops.extend(groups.get(cls, []))
    other_chops = _sort_by_key(other_chops, "pitch")
    other_pad = 0
    other_bank_idx = 6
    for chop in other_chops:
        if other_pad >= PADS_PER_BANK:
            other_bank_idx += 1
            other_pad = 0
        if other_bank_idx > 7:
            break
        bank = BANK_NAMES[other_bank_idx]
        assignment.banks[bank][other_pad] = PadSlot(chop=chop, pad_number=other_pad + 1, bank=bank)
        other_pad += 1

    return assignment


def map_chromatic(chops: list[ChopResult]) -> PadAssignment:
    """Chromatic strategy: all pitched chops sorted by MIDI note.

    Pad 1/Bank A = lowest pitch, ascending through banks.
    Turns pads into a chromatic keyboard.
    """
    assignment = PadAssignment()
    sorted_chops = sorted(chops, key=lambda c: c.pitch_midi or 0)

    for i, chop in enumerate(sorted_chops):
        if i >= MAX_SLOTS:
            break
        bank_idx = i // PADS_PER_BANK
        pad_idx = i % PADS_PER_BANK
        bank = BANK_NAMES[bank_idx]
        assignment.banks[bank][pad_idx] = PadSlot(
            chop=chop, pad_number=pad_idx + 1, bank=bank,
        )

    return assignment


def map_sequential(chops: list[ChopResult], sort_by: str = "time") -> PadAssignment:
    """Sequential strategy: chops in timeline order across pads/banks.

    Triggering pads in order replays the song as a chopped reconstruction.
    """
    assignment = PadAssignment()
    sorted_chops = _sort_by_key(chops, sort_by)

    for i, chop in enumerate(sorted_chops):
        if i >= MAX_SLOTS:
            break
        bank_idx = i // PADS_PER_BANK
        pad_idx = i % PADS_PER_BANK
        bank = BANK_NAMES[bank_idx]
        assignment.banks[bank][pad_idx] = PadSlot(
            chop=chop, pad_number=pad_idx + 1, bank=bank,
        )

    return assignment


def map_hybrid(chops: list[ChopResult]) -> PadAssignment:
    """Hybrid strategy: drums grouped (A-B), vocals chromatic (C-D),
    bass chromatic (E), other sequential (F-G), highlights (H).
    """
    assignment = PadAssignment()
    groups = _group_by_classification(chops)

    # Drums → grouped in Banks A-B
    drum_classes = ["kick", "snare", "hihat", "cymbal", "percussion"]
    drum_chops = []
    for cls in drum_classes:
        drum_chops.extend(groups.get(cls, []))
    drum_chops = _sort_by_key(drum_chops, "energy")
    for i, chop in enumerate(drum_chops[:32]):
        bank_idx = i // PADS_PER_BANK
        pad_idx = i % PADS_PER_BANK
        bank = BANK_NAMES[bank_idx]
        assignment.banks[bank][pad_idx] = PadSlot(chop=chop, pad_number=pad_idx + 1, bank=bank)

    # Vocals → chromatic in Banks C-D
    vocal_classes = ["vocal_phrase", "vocal_word", "vocal_syllable", "vocal_chop"]
    vocal_chops = []
    for cls in vocal_classes:
        vocal_chops.extend(groups.get(cls, []))
    vocal_chops = sorted(vocal_chops, key=lambda c: c.pitch_midi or 0)
    for i, chop in enumerate(vocal_chops[:32]):
        bank_idx = 2 + i // PADS_PER_BANK
        pad_idx = i % PADS_PER_BANK
        bank = BANK_NAMES[bank_idx]
        assignment.banks[bank][pad_idx] = PadSlot(chop=chop, pad_number=pad_idx + 1, bank=bank)

    # Bass → chromatic in Bank E
    bass_classes = ["bass_note", "bass_riff", "bass_slide"]
    bass_chops = []
    for cls in bass_classes:
        bass_chops.extend(groups.get(cls, []))
    bass_chops = sorted(bass_chops, key=lambda c: c.pitch_midi or 0)
    for i, chop in enumerate(bass_chops[:16]):
        assignment.banks["E"][i] = PadSlot(chop=chop, pad_number=i + 1, bank="E")

    # Other → sequential in Banks F-G
    other_classes = ["chord", "melody_stab", "melody_phrase", "texture", "noise"]
    other_chops = []
    for cls in other_classes:
        other_chops.extend(groups.get(cls, []))
    other_chops = _sort_by_key(other_chops, "time")
    for i, chop in enumerate(other_chops[:32]):
        bank_idx = 5 + i // PADS_PER_BANK
        pad_idx = i % PADS_PER_BANK
        bank = BANK_NAMES[bank_idx]
        assignment.banks[bank][pad_idx] = PadSlot(chop=chop, pad_number=pad_idx + 1, bank=bank)

    # Bank H → best-of highlights from all stems
    all_chops = sorted(chops, key=_score_chop, reverse=True)
    for i, chop in enumerate(all_chops[:16]):
        assignment.banks["H"][i] = PadSlot(chop=chop, pad_number=i + 1, bank="H")

    return assignment


def assign_pads(
    chops: list[ChopResult],
    strategy: str = "grouped",
    sort_by: str = "time",
) -> PadAssignment:
    """Main entry point: assign chops to pads using the specified strategy."""
    # If too many chops, select using diversity-aware algorithm
    if len(chops) > MAX_SLOTS:
        chops = select_diverse_chops(chops, MAX_SLOTS)

    if strategy == "grouped":
        return map_grouped(chops, sort_by)
    elif strategy == "chromatic":
        return map_chromatic(chops)
    elif strategy == "sequential":
        return map_sequential(chops, sort_by)
    elif strategy == "hybrid":
        return map_hybrid(chops)
    else:
        return map_sequential(chops, sort_by)
