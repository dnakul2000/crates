"""App-wide constants, paths, and session state."""

from dataclasses import dataclass, field
from pathlib import Path

# Root directory of the GrabStem project
APP_DIR = Path(__file__).resolve().parent.parent

# Runtime directories
DOWNLOADS_DIR = APP_DIR / "Downloads"
STEMS_DIR = APP_DIR / "Stems"
PACKS_DIR = APP_DIR / "Packs"

# Stem sub-categories
STEM_TYPES = ("Vocals", "Drums", "Bass", "Other")

# Audio defaults
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BIT_DEPTH = 24

# Ensure runtime directories exist
for d in [DOWNLOADS_DIR, STEMS_DIR, PACKS_DIR]:
    d.mkdir(exist_ok=True)
for stem in STEM_TYPES:
    (STEMS_DIR / stem).mkdir(exist_ok=True)


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
        """Rescan the Stems directory."""
        self.separated_stems = {}
        for stem_type in STEM_TYPES:
            stem_dir = STEMS_DIR / stem_type
            self.separated_stems[stem_type] = sorted(
                p for p in stem_dir.iterdir()
                if p.suffix.lower() in (".wav", ".flac")
            ) if stem_dir.exists() else []
        return self.separated_stems


# Global session state
session = SessionState()
