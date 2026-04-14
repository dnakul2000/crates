"""Audio playback engine using sounddevice for low-latency pad triggering."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


@dataclass
class PadSample:
    """A loaded sample ready for playback."""

    filename: str
    bank: str
    pad: int  # 1-16
    classification: str
    duration_ms: float
    pitch_name: str
    audio: np.ndarray = field(repr=False)  # float32 mono/stereo
    sr: int = 44100


class PlaybackEngine:
    """Manages sample loading and polyphonic playback."""

    MAX_CONCURRENT_STREAMS = 8  # Limit to prevent resource exhaustion

    def __init__(self):
        self.samples: dict[str, list[PadSample | None]] = {}  # bank -> 16 slots
        self.pack_dir: Path | None = None
        self.manifest: dict | None = None
        self._active_streams: list[sd.OutputStream] = []
        from collections import deque

        self._stream_queue: deque[sd.OutputStream] = deque(
            maxlen=self.MAX_CONCURRENT_STREAMS
        )

    def load_pack(self, pack_dir: Path) -> bool:
        """Load all samples from a pack directory into memory."""
        manifest_path = pack_dir / "pack_manifest.json"
        if not manifest_path.exists():
            return False

        raw = manifest_path.read_text()
        # Handle NaN/Infinity values that aren't valid JSON
        raw = (
            raw.replace(": NaN", ": null")
            .replace(": Infinity", ": null")
            .replace(": -Infinity", ": null")
        )
        try:
            self.manifest = json.loads(raw)
        except json.JSONDecodeError:
            # Manifest may be truncated or corrupted — build from filenames
            self.manifest = self._build_manifest_from_files(pack_dir)

        self.pack_dir = pack_dir
        self.samples.clear()

        # Init empty banks
        for bank in "ABCDEFGH":
            self.samples[bank] = [None] * 16

        for entry in self.manifest.get("samples", []):
            wav_path = pack_dir / entry["filename"]
            if not wav_path.exists():
                continue

            audio, sr = sf.read(str(wav_path), dtype="float32")
            # Ensure mono for simple playback
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            sample = PadSample(
                filename=entry["filename"],
                bank=entry["bank"],
                pad=entry["pad"],
                classification=entry.get("classification", ""),
                duration_ms=entry.get("duration_ms", 0) or len(audio) / sr * 1000,
                pitch_name=entry.get("pitch_name", ""),
                audio=audio,
                sr=sr,
            )

            pad_index = sample.pad - 1
            if 0 <= pad_index < 16 and sample.bank in self.samples:
                self.samples[sample.bank][pad_index] = sample

        return True

    @staticmethod
    def _build_manifest_from_files(pack_dir: Path) -> dict:
        """Build a minimal manifest by parsing WAV filenames (e.g. A01_kick_drums.wav)."""
        import re

        samples = []
        for wav in sorted(pack_dir.glob("*.wav")):
            match = re.match(r"^([A-H])(\d{2})_(.+?)_\w+\.wav$", wav.name)
            if match:
                bank, pad_str, classification = match.groups()
                samples.append(
                    {
                        "filename": wav.name,
                        "bank": bank,
                        "pad": int(pad_str),
                        "classification": classification,
                        "duration_ms": 0,
                        "pitch_name": "",
                    }
                )

        return {
            "name": pack_dir.name,
            "samples": samples,
            "total_samples": len(samples),
        }

    def play(self, bank: str, pad: int):
        """Trigger playback for a pad (1-indexed). Non-blocking, polyphonic."""
        pad_index = pad - 1
        slots = self.samples.get(bank)
        if not slots or pad_index < 0 or pad_index >= 16:
            return

        sample = slots[pad_index]
        if sample is None:
            return

        # Stop any existing playback that's finished
        self._cleanup_streams()

        # Enforce max concurrent streams limit - close oldest if needed
        while len(self._active_streams) >= self.MAX_CONCURRENT_STREAMS:
            oldest = self._active_streams.pop(0)
            try:
                oldest.stop()
                oldest.close()
            except Exception:
                pass

        # Play in a non-blocking stream
        audio = sample.audio.copy()
        pos = 0

        def callback(outdata, frames, time_info, status):
            nonlocal pos
            remaining = len(audio) - pos
            if remaining <= 0:
                outdata[:] = 0
                raise sd.CallbackStop
            chunk = min(frames, remaining)
            outdata[:chunk, 0] = audio[pos : pos + chunk]
            outdata[chunk:] = 0
            pos += chunk

        stream = sd.OutputStream(
            samplerate=sample.sr,
            channels=1,
            callback=callback,
            blocksize=512,
        )
        stream.start()
        self._active_streams.append(stream)

    def stop_all(self):
        """Stop all currently playing samples."""
        for stream in self._active_streams:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self._active_streams.clear()

    def get_sample(self, bank: str, pad: int) -> PadSample | None:
        """Get the sample info for a pad (1-indexed)."""
        slots = self.samples.get(bank)
        if not slots:
            return None
        pad_index = pad - 1
        if 0 <= pad_index < 16:
            return slots[pad_index]
        return None

    def _cleanup_streams(self):
        """Remove finished streams."""
        active = []
        for stream in self._active_streams:
            if stream.active:
                active.append(stream)
            else:
                try:
                    stream.close()
                except Exception:
                    pass
        self._active_streams = active

    def cleanup(self):
        """Clean up all resources."""
        self.stop_all()
