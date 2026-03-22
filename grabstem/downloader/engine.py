"""yt-dlp download logic with QThread worker."""

import subprocess
from pathlib import Path

from PyQt6.QtCore import pyqtSignal

from ..common.file_utils import safe_name
from ..common.workers import BaseWorker
from ..config import DOWNLOADS_DIR
from ..spotify.models import TrackInfo


def download_song(track: TrackInfo, output_dir: Path) -> Path | None:
    """Download a song from YouTube using yt-dlp. Returns output path or None."""
    query = f"{track.artist} - {track.title} official audio"
    filename = safe_name(track.artist, track.title)
    output_template = str(output_dir / f"{filename}.%(ext)s")

    cmd = [
        "yt-dlp",
        f"ytsearch1:{query}",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", output_template,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
    except subprocess.CalledProcessError:
        return None
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None

    # Find the output file
    expected = output_dir / f"{filename}.mp3"
    if expected.exists():
        return expected

    # yt-dlp might use a slightly different name; search for it
    for p in output_dir.glob(f"{filename}.*"):
        if p.suffix.lower() in (".mp3", ".m4a", ".wav", ".opus"):
            return p

    return None


class DownloadWorker(BaseWorker):
    """Downloads a list of tracks in a background thread."""

    track_done = pyqtSignal(int, str)  # index, output_path_or_empty

    def __init__(self, tracks: list[TrackInfo], output_dir: Path | None = None, parent=None):
        super().__init__(parent)
        self.tracks = tracks
        self.output_dir = output_dir or DOWNLOADS_DIR

    def run(self):
        total = len(self.tracks)
        for i, track in enumerate(self.tracks):
            if self.is_cancelled:
                break
            if not track.selected:
                self.track_done.emit(i, "")
                continue

            self.progress.emit(i, total, f"Downloading {track.artist} - {track.title}")
            result = download_song(track, self.output_dir)
            self.track_done.emit(i, str(result) if result else "")

        self.progress.emit(total, total, "Done")
        self.finished_work.emit()
