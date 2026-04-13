"""App-wide constants, paths, and session state."""

from dataclasses import dataclass, field
from pathlib import Path

from .paths import data_dir

# Root directory for runtime data (Downloads, Stems, Packs)
APP_DIR = data_dir()

# Runtime directories
DOWNLOADS_DIR = APP_DIR / "Downloads"
STEMS_DIR = APP_DIR / "Stems"
PACKS_DIR = APP_DIR / "Packs"

# Stem sub-categories
STEM_TYPES = ("Vocals", "Drums", "Bass", "Other")
STEM_TYPES_6 = ("Vocals", "Drums", "Bass", "Guitar", "Piano", "Other")

# Map model filenames to their stem types
MODEL_STEM_MAP: dict[str, tuple[str, ...]] = {
    "htdemucs_ft.yaml": STEM_TYPES,
    "htdemucs.yaml": STEM_TYPES,
    "htdemucs_6s.yaml": STEM_TYPES_6,
    "mdx_extra.yaml": STEM_TYPES,
    "mdx_extra_q.yaml": STEM_TYPES,
}


def get_stem_types(model_name: str) -> tuple[str, ...]:
    """Return the stem types produced by the given model."""
    key = model_name if model_name.endswith((".yaml", ".ckpt", ".pth", ".onnx")) else f"{model_name}.yaml"
    return MODEL_STEM_MAP.get(key, STEM_TYPES)


# Audio defaults
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BIT_DEPTH = 24

# Ensure runtime directories exist
for d in [DOWNLOADS_DIR, STEMS_DIR, PACKS_DIR]:
    d.mkdir(exist_ok=True)


@dataclass
class SessionState:
    """Tracks state across tabs within a single app session."""

    downloaded_files: list[Path] = field(default_factory=list)
    separated_stems: dict[str, list[Path]] = field(default_factory=dict)
    current_pack_name: str = ""
    selected_preset: str = ""
    intensity: int = 75

    def refresh_downloads(self) -> list[Path]:
        """Rescan the Downloads directory for audio files."""
        self.downloaded_files = sorted(
            p for p in DOWNLOADS_DIR.iterdir()
            if p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a")
        )
        return self.downloaded_files

    def refresh_stems(self) -> dict[str, list[Path]]:
        """Rescan the Stems directory (per-song folder structure).

        Layout: Stems/{SongName}/{vocals,drums,bass,...}.wav
        Returns dict mapping stem_type -> list of file paths.
        """
        self.separated_stems = {}
        if not STEMS_DIR.exists():
            return self.separated_stems

        for song_dir in sorted(STEMS_DIR.iterdir()):
            if not song_dir.is_dir() or song_dir.name.startswith((".", "_")):
                continue
            for f in sorted(song_dir.iterdir()):
                if f.suffix.lower() not in (".wav", ".flac"):
                    continue
                # Stem type from filename: vocals.wav -> Vocals
                stem_type = f.stem.capitalize()
                self.separated_stems.setdefault(stem_type, []).append(f)

        return self.separated_stems


# Global session state
session = SessionState()
