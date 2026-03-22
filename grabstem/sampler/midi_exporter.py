"""MIDI and MPC program file export.

Generates:
- .mid file with pad-to-note mapping
- .xpm MPC program file (XML format) mapping pads to WAV samples
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from .pad_mapper import BANK_NAMES, PADS_PER_BANK, PadAssignment


# MPC pad-to-MIDI note mapping (standard MPC layout)
# Bank A: pads 1-16 -> MIDI notes 36-51
# Bank B: pads 1-16 -> MIDI notes 52-67
# etc.
def _pad_to_midi_note(bank: str, pad: int) -> int:
    """Convert bank + pad number to MIDI note."""
    bank_offset = BANK_NAMES.index(bank) * PADS_PER_BANK
    return 36 + bank_offset + (pad - 1)


def export_midi(
    assignment: PadAssignment,
    output_path: Path,
    bpm: float = 120.0,
) -> Path:
    """Generate a MIDI file mapping each assigned pad to its MIDI note.

    Creates one note per pad at the correct MIDI pitch, velocity from RMS.
    """
    import mido

    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo
    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(mido.MetaMessage("track_name", name="GrabStem Pad Map", time=0))

    # Add one note per assigned pad (quarter note each, sequential)
    import numpy as np
    ticks_per_note = 480  # one quarter note

    for bank_name in BANK_NAMES:
        bank = assignment.banks.get(bank_name, [])
        for i, slot in enumerate(bank):
            if slot is None:
                continue

            note = _pad_to_midi_note(bank_name, slot.pad_number)

            # Velocity from RMS energy (scale 40-127)
            rms = float(np.sqrt(np.mean(slot.chop.audio ** 2))) if len(slot.chop.audio) > 0 else 0.5
            velocity = max(40, min(127, int(rms * 500)))

            track.append(mido.Message("note_on", note=note, velocity=velocity, time=0))
            track.append(mido.Message("note_off", note=note, velocity=0, time=ticks_per_note))

    midi_path = output_path.with_suffix(".mid")
    mid.save(str(midi_path))
    return midi_path


def export_mpc_program(
    assignment: PadAssignment,
    output_path: Path,
    pack_dir: Path,
    program_name: str = "GrabStem Pack",
) -> Path:
    """Generate an MPC program file (.xpm) mapping pads to WAV samples.

    The .xpm format is XML-based and used by MPC Software/hardware.
    """
    from ..common.file_utils import sanitize_filename

    root = ET.Element("MPCVObject")
    root.set("Type", "Program")
    root.set("Version", "3.0")

    program = ET.SubElement(root, "Program")
    ET.SubElement(program, "ProgramName").text = program_name
    ET.SubElement(program, "ProgramType").text = "Drum"

    instruments = ET.SubElement(program, "Instruments")

    for bank_name in BANK_NAMES:
        bank = assignment.banks.get(bank_name, [])
        for i, slot in enumerate(bank):
            if slot is None:
                continue

            chop = slot.chop
            note = _pad_to_midi_note(bank_name, slot.pad_number)

            # Build the WAV filename (must match exporter output)
            source_short = sanitize_filename(chop.source_stem.lower())
            classification = sanitize_filename(chop.classification)
            wav_filename = f"{bank_name}{slot.pad_number:02d}_{classification}_{source_short}.wav"

            instrument = ET.SubElement(instruments, "Instrument")
            ET.SubElement(instrument, "InstrumentName").text = f"{classification}_{source_short}"
            ET.SubElement(instrument, "InstrumentNumber").text = str(note - 36)

            layers = ET.SubElement(instrument, "Layers")
            layer = ET.SubElement(layers, "Layer")
            ET.SubElement(layer, "SampleName").text = wav_filename
            ET.SubElement(layer, "SamplePath").text = str(pack_dir / wav_filename)
            ET.SubElement(layer, "VelocityStart").text = "0"
            ET.SubElement(layer, "VelocityEnd").text = "127"
            ET.SubElement(layer, "TuneCoarse").text = "0"
            ET.SubElement(layer, "TuneFine").text = "0"

            # Set root note from pitch detection
            if chop.pitch_midi and chop.pitch_midi > 0:
                ET.SubElement(layer, "RootNote").text = str(chop.pitch_midi)
            else:
                ET.SubElement(layer, "RootNote").text = str(note)

            ET.SubElement(layer, "KeyTrack").text = "False"
            ET.SubElement(layer, "OneShot").text = "True"

    xpm_path = output_path.with_suffix(".xpm")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(xpm_path), encoding="utf-8", xml_declaration=True)
    return xpm_path
