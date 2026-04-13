"""Pad assignment logic — 4 mapping strategies, device-aware.

Supports any pad controller via DeviceProfile. Default: MPK Mini MK3 (8 pads x 2 banks).
Pad layout (bottom-left origin):
    MPK Mini MK3 (2x4):        MPC Mini MK3 (4x4):
     5   6   7   8              13  14  15  16
     1   2   3   4               9  10  11  12
                                 5   6   7   8
                                 1   2   3   4
"""

from dataclasses import dataclass, field

import numpy as np

from ..devices import DEFAULT_DEVICE, DeviceProfile
from .chopper import ChopResult


@dataclass
class PadSlot:
    """A single pad assignment with optional layering."""

    chop: ChopResult  # primary sample
    pad_number: int  # 1-based
    bank: str
    velocity_layer: int = 1
    layers: list[ChopResult] | None = None


@dataclass
class PadAssignment:
    """Complete pad mapping for a sample pack."""

    banks: dict[str, list[PadSlot | None]] = field(default_factory=dict)
    device: DeviceProfile = field(default_factory=lambda: DEFAULT_DEVICE)

    def __post_init__(self):
        if not self.banks:
            for bank in self.device.bank_names:
                self.banks[bank] = [None] * self.device.pads_per_bank

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
        duration_score = duration_s * 20
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

    # Tonal clarity (0-2): pitched content scores higher
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
    return np.array([centroid, bandwidth, flatness, rms])


def select_diverse_chops(
    chops: list[ChopResult],
    max_count: int,
    device: DeviceProfile | None = None,
) -> list[ChopResult]:
    """Select chops using diversity-aware scoring (maximal marginal relevance).

    Two-pass: first guarantee minimum quota per classification,
    then fill remaining slots by spectral diversity.
    """
    if len(chops) <= max_count:
        return chops

    device = device or DEFAULT_DEVICE

    # Group by classification
    groups: dict[str, list[ChopResult]] = {}
    for chop in chops:
        groups.setdefault(chop.classification, []).append(chop)

    # For small pad counts, ensure at least 1 per classification
    if max_count <= 32:
        min_per_class = max(1, min(2, max_count // max(len(groups), 1)))
    else:
        min_per_class = max(2, min(4, max_count // max(len(groups), 1)))

    selected: list[ChopResult] = []
    selected_set: set[int] = set()

    for cls, cls_chops in groups.items():
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

    unselected = [c for c in chops if id(c) not in selected_set]
    if not unselected:
        return selected

    # Compute signatures for all chops, then z-score normalize so all dimensions
    # contribute equally to diversity scoring
    all_sigs_raw = [_compute_spectral_signature(c.audio, c.sr) for c in selected + unselected]
    all_sigs_arr = np.array(all_sigs_raw)
    sig_means = all_sigs_arr.mean(axis=0)
    sig_stds = all_sigs_arr.std(axis=0) + 1e-10
    all_sigs_norm = (all_sigs_arr - sig_means) / sig_stds

    n_selected = len(selected)
    selected_sigs = list(all_sigs_norm[:n_selected])
    candidate_sigs = list(all_sigs_norm[n_selected:])
    candidate_scores = [_score_chop(c) for c in unselected]

    for _ in range(remaining):
        if not unselected:
            break

        best_idx = -1
        best_mmr = -float("inf")

        for i, (chop, sig, score) in enumerate(zip(unselected, candidate_sigs, candidate_scores)):
            relevance = score / 10.0

            if selected_sigs:
                min_sim = min(
                    float(np.dot(sig, sel_sig)) / (np.linalg.norm(sig) * np.linalg.norm(sel_sig) + 1e-10)
                    for sel_sig in selected_sigs
                )
            else:
                min_sim = 0

            time_diversity = 1.0
            for sel_chop in selected[-20:]:
                if abs(chop.start_time - sel_chop.start_time) < 0.5:
                    time_diversity *= 0.7

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
    max_pads: int | None = None,
) -> int:
    """Assign chops to pads in a specific bank. Returns number assigned."""
    pads_per_bank = assignment.device.pads_per_bank
    if max_pads is None:
        max_pads = pads_per_bank
    count = 0
    for i, chop in enumerate(chops):
        pad_idx = start_pad + i
        if pad_idx >= min(start_pad + max_pads, pads_per_bank):
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


def _group_by_stem(chops: list[ChopResult]) -> dict[str, list[ChopResult]]:
    """Group chops by their source stem type."""
    groups: dict[str, list[ChopResult]] = {}
    for chop in chops:
        groups.setdefault(chop.source_stem or "Other", []).append(chop)
    return groups


def _fill_bank_linearly(
    chops: list[ChopResult],
    assignment: PadAssignment,
    bank_names: list[str],
    sort_by: str = "energy",
) -> None:
    """Fill a sequence of banks linearly with chops."""
    pads_per_bank = assignment.device.pads_per_bank
    sorted_chops = _sort_by_key(chops, sort_by)
    total_slots = len(bank_names) * pads_per_bank
    for i, chop in enumerate(sorted_chops[:total_slots]):
        bank_idx = i // pads_per_bank
        pad_idx = i % pads_per_bank
        bank = bank_names[bank_idx]
        assignment.banks[bank][pad_idx] = PadSlot(
            chop=chop, pad_number=pad_idx + 1, bank=bank,
        )


def map_grouped(
    chops: list[ChopResult],
    sort_by: str = "energy",
    device: DeviceProfile | None = None,
) -> PadAssignment:
    """Grouped strategy: same classification on adjacent pads.

    For large devices (MPC, 128 slots): dedicated banks per stem type.
    For small devices (MPK, 16 slots): proportional allocation per stem type.
    """
    device = device or DEFAULT_DEVICE
    assignment = PadAssignment(device=device)
    stem_groups = _group_by_stem(chops)

    if device.max_slots <= 32:
        # Small device: proportional allocation across all banks
        # Allocate slots proportionally to how many chops each stem has
        total_chops = len(chops)
        if total_chops == 0:
            return assignment

        max_slots = device.max_slots
        stem_types = list(stem_groups.keys())

        # Calculate proportional slots, minimum 1 per stem with content
        allocation: dict[str, int] = {}
        for stem in stem_types:
            allocation[stem] = max(1, round(len(stem_groups[stem]) / total_chops * max_slots))

        # Adjust to fit exactly within max_slots
        total_allocated = sum(allocation.values())
        while total_allocated > max_slots:
            # Shrink the largest allocation
            largest = max(allocation, key=allocation.get)
            if allocation[largest] > 1:
                allocation[largest] -= 1
            total_allocated -= 1
        while total_allocated < max_slots and stem_groups:
            # Grow the allocation with most remaining chops
            best = max(stem_types, key=lambda s: len(stem_groups[s]) - allocation.get(s, 0))
            allocation[best] = allocation.get(best, 0) + 1
            total_allocated += 1

        # Fill sequentially across banks
        slot_idx = 0
        pads_per_bank = device.pads_per_bank
        for stem, n_slots in allocation.items():
            best_chops = _sort_by_key(stem_groups[stem], sort_by)[:n_slots]
            for chop in best_chops:
                if slot_idx >= max_slots:
                    break
                bank_idx = slot_idx // pads_per_bank
                pad_idx = slot_idx % pads_per_bank
                bank = device.bank_names[bank_idx]
                assignment.banks[bank][pad_idx] = PadSlot(
                    chop=chop, pad_number=pad_idx + 1, bank=bank,
                )
                slot_idx += 1

        return assignment

    # Large device: dedicated banks per stem type
    groups = _group_by_classification(chops)
    bank_names = device.bank_names
    pads_per_bank = device.pads_per_bank

    # Drum classifications -> first 2 banks
    drum_order = ["kick", "snare", "hihat", "hihat_closed", "hihat_open",
                  "cymbal", "cymbal_crash", "cymbal_ride", "tom", "percussion"]
    drum_pad = 0
    drum_bank_idx = 0
    for cls in drum_order:
        cls_chops = _sort_by_key(groups.get(cls, []), sort_by)
        for chop in cls_chops:
            if drum_pad >= pads_per_bank:
                drum_bank_idx += 1
                drum_pad = 0
            if drum_bank_idx > 1:
                break
            bank = bank_names[drum_bank_idx]
            assignment.banks[bank][drum_pad] = PadSlot(
                chop=chop, pad_number=drum_pad + 1, bank=bank,
            )
            drum_pad += 1

    # Vocal classifications -> banks 2-3
    vocal_classes = ["vocal_phrase", "vocal_word", "vocal_syllable",
                     "vocal_chop", "vocal_harmony", "vocal_fx", "breath"]
    vocal_pad = 0
    vocal_bank_idx = 2
    for cls in vocal_classes:
        cls_chops = _sort_by_key(groups.get(cls, []), sort_by)
        for chop in cls_chops:
            if vocal_pad >= pads_per_bank:
                vocal_bank_idx += 1
                vocal_pad = 0
            if vocal_bank_idx > 3:
                break
            bank = bank_names[vocal_bank_idx]
            assignment.banks[bank][vocal_pad] = PadSlot(
                chop=chop, pad_number=vocal_pad + 1, bank=bank,
            )
            vocal_pad += 1

    # Bass -> bank 4
    bass_classes = ["bass_note", "bass_riff", "bass_slide", "bass_pluck", "bass_sub"]
    bass_chops = []
    for cls in bass_classes:
        bass_chops.extend(groups.get(cls, []))
    bass_chops = _sort_by_key(bass_chops, "pitch")
    if len(bank_names) > 4:
        _fill_bank_linearly(bass_chops, assignment, [bank_names[4]], "pitch")

    # Guitar -> bank 5 (if available)
    guitar_classes = ["guitar_strum", "guitar_pick", "guitar_riff", "guitar_chord", "guitar_bend"]
    guitar_chops = []
    for cls in guitar_classes:
        guitar_chops.extend(groups.get(cls, []))
    if guitar_chops and len(bank_names) > 5:
        _fill_bank_linearly(guitar_chops, assignment, [bank_names[5]], "pitch")

    # Piano -> bank 6 (if available)
    piano_classes = ["piano_chord", "piano_melody", "piano_stab", "piano_phrase"]
    piano_chops = []
    for cls in piano_classes:
        piano_chops.extend(groups.get(cls, []))
    if piano_chops and len(bank_names) > 6:
        _fill_bank_linearly(piano_chops, assignment, [bank_names[6]], "pitch")

    # Other -> remaining banks
    other_classes = ["pad", "lead", "stab", "texture", "sweep", "riser",
                     "noise", "chord", "melody_stab", "melody_phrase"]
    other_chops = []
    for cls in other_classes:
        other_chops.extend(groups.get(cls, []))
    remaining_banks = bank_names[5 if not guitar_chops else 7:]
    if remaining_banks:
        _fill_bank_linearly(other_chops, assignment, remaining_banks, "pitch")

    return assignment


def map_chromatic(
    chops: list[ChopResult],
    device: DeviceProfile | None = None,
) -> PadAssignment:
    """Chromatic strategy: all pitched chops sorted by MIDI note.

    Pad 1/Bank A = lowest pitch, ascending through banks.
    """
    device = device or DEFAULT_DEVICE
    assignment = PadAssignment(device=device)
    sorted_chops = sorted(chops, key=lambda c: c.pitch_midi or 0)

    pads_per_bank = device.pads_per_bank
    bank_names = device.bank_names
    max_slots = device.max_slots

    for i, chop in enumerate(sorted_chops):
        if i >= max_slots:
            break
        bank_idx = i // pads_per_bank
        pad_idx = i % pads_per_bank
        bank = bank_names[bank_idx]
        assignment.banks[bank][pad_idx] = PadSlot(
            chop=chop, pad_number=pad_idx + 1, bank=bank,
        )

    return assignment


def map_sequential(
    chops: list[ChopResult],
    sort_by: str = "time",
    device: DeviceProfile | None = None,
) -> PadAssignment:
    """Sequential strategy: chops in timeline order across pads/banks."""
    device = device or DEFAULT_DEVICE
    assignment = PadAssignment(device=device)
    sorted_chops = _sort_by_key(chops, sort_by)

    pads_per_bank = device.pads_per_bank
    bank_names = device.bank_names
    max_slots = device.max_slots

    for i, chop in enumerate(sorted_chops):
        if i >= max_slots:
            break
        bank_idx = i // pads_per_bank
        pad_idx = i % pads_per_bank
        bank = bank_names[bank_idx]
        assignment.banks[bank][pad_idx] = PadSlot(
            chop=chop, pad_number=pad_idx + 1, bank=bank,
        )

    return assignment


def map_hybrid(
    chops: list[ChopResult],
    device: DeviceProfile | None = None,
) -> PadAssignment:
    """Hybrid strategy: mix of grouped and chromatic.

    For small devices, falls back to grouped (proportional).
    For large devices: drums A-B, vocals C-D, bass E, other F-G, highlights H.
    """
    device = device or DEFAULT_DEVICE

    if device.max_slots <= 32:
        return map_grouped(chops, sort_by="energy", device=device)

    assignment = PadAssignment(device=device)
    groups = _group_by_classification(chops)
    bank_names = device.bank_names
    pads_per_bank = device.pads_per_bank

    # Drums -> grouped in first 2 banks
    drum_classes = ["kick", "snare", "hihat", "hihat_closed", "hihat_open",
                    "cymbal", "cymbal_crash", "cymbal_ride", "tom", "percussion"]
    drum_chops = []
    for cls in drum_classes:
        drum_chops.extend(groups.get(cls, []))
    drum_chops = _sort_by_key(drum_chops, "energy")
    for i, chop in enumerate(drum_chops[:pads_per_bank * 2]):
        bank_idx = i // pads_per_bank
        pad_idx = i % pads_per_bank
        bank = bank_names[bank_idx]
        assignment.banks[bank][pad_idx] = PadSlot(chop=chop, pad_number=pad_idx + 1, bank=bank)

    # Vocals -> chromatic in banks 2-3
    vocal_classes = ["vocal_phrase", "vocal_word", "vocal_syllable",
                     "vocal_chop", "vocal_harmony", "vocal_fx"]
    vocal_chops = []
    for cls in vocal_classes:
        vocal_chops.extend(groups.get(cls, []))
    vocal_chops = sorted(vocal_chops, key=lambda c: c.pitch_midi or 0)
    for i, chop in enumerate(vocal_chops[:pads_per_bank * 2]):
        bank_idx = 2 + i // pads_per_bank
        pad_idx = i % pads_per_bank
        bank = bank_names[bank_idx]
        assignment.banks[bank][pad_idx] = PadSlot(chop=chop, pad_number=pad_idx + 1, bank=bank)

    # Bass -> chromatic in bank 4
    bass_classes = ["bass_note", "bass_riff", "bass_slide", "bass_pluck", "bass_sub"]
    bass_chops = []
    for cls in bass_classes:
        bass_chops.extend(groups.get(cls, []))
    bass_chops = sorted(bass_chops, key=lambda c: c.pitch_midi or 0)
    for i, chop in enumerate(bass_chops[:pads_per_bank]):
        assignment.banks[bank_names[4]][i] = PadSlot(chop=chop, pad_number=i + 1, bank=bank_names[4])

    # Other -> sequential in banks 5-6
    other_classes = ["pad", "lead", "stab", "texture", "sweep", "riser",
                     "noise", "chord", "melody_stab", "melody_phrase",
                     "guitar_strum", "guitar_pick", "guitar_riff", "guitar_chord", "guitar_bend",
                     "piano_chord", "piano_melody", "piano_stab", "piano_phrase"]
    other_chops = []
    for cls in other_classes:
        other_chops.extend(groups.get(cls, []))
    other_chops = _sort_by_key(other_chops, "time")
    for i, chop in enumerate(other_chops[:pads_per_bank * 2]):
        bank_idx = 5 + i // pads_per_bank
        pad_idx = i % pads_per_bank
        if bank_idx < len(bank_names):
            bank = bank_names[bank_idx]
            assignment.banks[bank][pad_idx] = PadSlot(chop=chop, pad_number=pad_idx + 1, bank=bank)

    # Last bank -> best-of highlights from all stems
    if len(bank_names) >= 8:
        all_chops = sorted(chops, key=_score_chop, reverse=True)
        last_bank = bank_names[-1]
        for i, chop in enumerate(all_chops[:pads_per_bank]):
            assignment.banks[last_bank][i] = PadSlot(chop=chop, pad_number=i + 1, bank=last_bank)

    return assignment


def assign_pads(
    chops: list[ChopResult],
    strategy: str = "grouped",
    sort_by: str = "time",
    device: DeviceProfile | None = None,
) -> PadAssignment:
    """Main entry point: assign chops to pads using the specified strategy."""
    device = device or DEFAULT_DEVICE
    max_slots = device.max_slots

    # If too many chops, select using diversity-aware algorithm
    if len(chops) > max_slots:
        chops = select_diverse_chops(chops, max_slots, device)

    if strategy == "grouped":
        return map_grouped(chops, sort_by, device)
    elif strategy == "chromatic":
        return map_chromatic(chops, device)
    elif strategy == "sequential":
        return map_sequential(chops, sort_by, device)
    elif strategy == "hybrid":
        return map_hybrid(chops, device)
    else:
        return map_sequential(chops, sort_by, device)
