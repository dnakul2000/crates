"""WAV export + metadata tagging for MPC-compatible sample packs."""

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..common.audio_utils import (
    NormalizationStrategy,
    fade,
    normalize,
    normalize_group,
    save_audio,
)
from ..common.file_utils import ensure_dir, sanitize_filename
from ..config import DEFAULT_BIT_DEPTH, DEFAULT_SAMPLE_RATE, PACKS_DIR
from .pad_mapper import PadAssignment


def export_pack(
    assignment: PadAssignment,
    pack_name: str,
    preset_name: str,
    intensity: int,
    source_songs: list[str],
    output_dir: Path | None = None,
    bit_depth: int = DEFAULT_BIT_DEPTH,
    normalization: str = "peak_group",
) -> tuple[Path, list[str]]:
    """Export a complete sample pack as WAV files + manifest.

    Returns (pack_directory, warnings) tuple.
    """
    output_dir = output_dir or PACKS_DIR
    pack_dir = ensure_dir(output_dir / sanitize_filename(pack_name))

    # Collect all samples for group-aware normalization
    strategy = NormalizationStrategy(normalization)
    sample_entries = []  # (key, filename, chop, bank_name, pad_number)

    for bank_name in assignment.device.bank_names:
        bank = assignment.banks.get(bank_name, [])
        for i, slot in enumerate(bank):
            if slot is None:
                continue
            chop = slot.chop
            source_short = sanitize_filename(chop.source_stem.lower())
            classification = sanitize_filename(chop.classification)
            filename = (
                f"{bank_name}{slot.pad_number:02d}_{classification}_{source_short}.wav"
            )
            # Key format: "classification::bank_pad" for group-aware normalization
            key = f"{chop.classification}::{bank_name}{slot.pad_number:02d}"
            sample_entries.append((key, filename, chop, bank_name, slot.pad_number))

    # Normalize all samples as a group
    audio_items = [
        (key, chop.audio.copy(), chop.sr) for key, _, chop, _, _ in sample_entries
    ]
    normalized_audio = normalize_group(audio_items, strategy)

    # Export each sample with chunked processing to manage resources
    samples_meta = []
    total_exported = 0
    CHUNK_SIZE = 50  # Process in chunks to limit concurrent resource usage

    def process_chunk(chunk_items):
        """Process a chunk of samples and return their metadata."""
        chunk_meta = []
        for key, filename, chop, bank_name, pad_number in chunk_items:
            filepath = pack_dir / filename

            # Get group-normalized audio, then apply fades
            audio = normalized_audio[key]
            audio = fade(audio, chop.sr)

            # Save WAV with explicit resource management
            save_audio(audio, chop.sr, filepath, bit_depth=bit_depth)

            # Compute RMS safely — guard against empty or all-zero audio
            if len(chop.audio) > 0:
                rms_val = float(np.sqrt(np.mean(chop.audio**2)))
                rms_db = round(20 * np.log10(max(rms_val, 1e-10)), 1)
            else:
                rms_db = -100.0
            if not np.isfinite(rms_db):
                rms_db = -100.0

            chunk_meta.append(
                {
                    "filename": filename,
                    "bank": bank_name,
                    "pad": pad_number,
                    "classification": chop.classification,
                    "bpm": chop.source_bpm,
                    "key": chop.source_key,
                    "beat_length": chop.beat_length,
                    "duration_ms": int(len(chop.audio) / chop.sr * 1000),
                    "pitch_midi": chop.pitch_midi,
                    "pitch_name": chop.pitch_name,
                    "source_stem": chop.source_stem,
                    "start_time": round(chop.start_time, 3),
                    "end_time": round(chop.end_time, 3),
                    "temporal_group": chop.temporal_group,
                    "rms_db": rms_db,
                }
            )
        return chunk_meta

    # Process samples in chunks
    for i in range(0, len(sample_entries), CHUNK_SIZE):
        chunk = sample_entries[i : i + CHUNK_SIZE]
        samples_meta.extend(process_chunk(chunk))
        total_exported += len(chunk)
        # Allow garbage collection between chunks
        gc.collect()

    # Write manifest
    manifest = {
        "name": pack_name,
        "preset": preset_name,
        "intensity": intensity,
        "normalization": normalization,
        "device": assignment.device.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bit_depth": bit_depth,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "total_samples": total_exported,
        "source_songs": source_songs,
        "samples": samples_meta,
    }

    manifest_path = pack_dir / "pack_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Generate MIDI and MPC program files
    warnings: list[str] = []
    try:
        from .midi_exporter import export_midi, export_mpc_program

        export_midi(assignment, pack_dir / pack_name)
        export_mpc_program(
            assignment, pack_dir / pack_name, pack_dir, program_name=pack_name
        )
    except ImportError:
        warnings.append(
            "MIDI/MPC program files not generated — 'mido' package not installed. "
            "Run: pip install mido python-rtmidi"
        )
    except Exception as e:
        warnings.append(f"MIDI export error: {e}")

    return pack_dir, warnings
