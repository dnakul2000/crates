"""yt-dlp download logic with QThread worker.

Multi-candidate search with scoring to avoid downloading wrong songs.
"""

import difflib
import json
import re
import subprocess
from pathlib import Path

from PyQt6.QtCore import pyqtSignal

from ..common.file_utils import safe_name
from ..common.workers import BaseWorker
from ..config import DOWNLOADS_DIR
from ..paths import yt_dlp_path
from ..spotify.models import TrackInfo


# Words that indicate a non-original version (unless the source track also has them)
_REJECT_KEYWORDS = {"cover", "karaoke", "tutorial", "reaction", "lesson", "instrumental"}
_VARIANT_KEYWORDS = {"remix", "live", "acoustic", "demo", "slowed", "sped up", "reverb"}


def _search_candidates(query: str, max_results: int = 5) -> list[dict]:
    """Search YouTube for candidates and return their metadata without downloading."""
    cmd = [
        yt_dlp_path(),
        f"ytsearch{max_results}:{query}",
        "--dump-json",
        "--flat-playlist",
        "--no-warnings",
        "--quiet",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
        candidates = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return candidates
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def _fuzzy_match(a: str, b: str) -> float:
    """Fuzzy string similarity (0-1)."""
    a = re.sub(r"[^\w\s]", "", a.lower()).strip()
    b = re.sub(r"[^\w\s]", "", b.lower()).strip()
    return difflib.SequenceMatcher(None, a, b).ratio()


def _score_candidate(track: TrackInfo, candidate: dict) -> tuple[float, str]:
    """Score a YouTube candidate against the Spotify track.

    Returns (score 0-100, reason string).
    """
    score = 50.0  # baseline
    reasons = []

    title = candidate.get("title", "") or candidate.get("fulltitle", "")
    uploader = candidate.get("uploader", "") or candidate.get("channel", "")
    duration = candidate.get("duration")  # seconds
    view_count = candidate.get("view_count", 0) or 0
    title_lower = title.lower()

    # --- Duration match (heaviest weight: up to +30 / -50) ---
    if track.duration_ms and duration:
        track_duration_s = track.duration_ms / 1000
        diff = abs(duration - track_duration_s)
        if diff < 5:
            score += 30
            reasons.append(f"duration match ({diff:.0f}s off)")
        elif diff < 15:
            score += 15
            reasons.append(f"duration close ({diff:.0f}s off)")
        elif diff < 30:
            score -= 10
            reasons.append(f"duration mismatch ({diff:.0f}s off)")
        else:
            score -= 50
            reasons.append(f"duration way off ({diff:.0f}s)")

    # --- Title match (up to +20) ---
    title_sim = _fuzzy_match(track.title, title)
    score += title_sim * 20
    if title_sim > 0.6:
        reasons.append(f"title match ({title_sim:.0%})")

    # --- Artist match (up to +15) ---
    # Check both the uploader/channel and whether artist appears in title
    artist_in_uploader = _fuzzy_match(track.artist, uploader)
    artist_in_title = _fuzzy_match(track.artist, title)
    artist_score = max(artist_in_uploader, artist_in_title)
    score += artist_score * 15
    if artist_score > 0.5:
        reasons.append(f"artist match ({artist_score:.0%})")

    # --- Reject filters ---
    track_title_lower = track.title.lower()
    for keyword in _REJECT_KEYWORDS:
        if keyword in title_lower and keyword not in track_title_lower:
            score -= 40
            reasons.append(f"reject: '{keyword}' in title")

    # --- Variant penalty (remix/live etc unless source also has it) ---
    for keyword in _VARIANT_KEYWORDS:
        if keyword in title_lower and keyword not in track_title_lower:
            score -= 15
            reasons.append(f"variant: '{keyword}'")

    # --- View count tiebreaker (small: up to +5) ---
    if view_count > 1_000_000:
        score += 5
    elif view_count > 100_000:
        score += 3
    elif view_count > 10_000:
        score += 1

    # --- "Official" bonus ---
    if "official" in title_lower:
        score += 5
        reasons.append("official in title")

    return max(0, min(100, score)), "; ".join(reasons) if reasons else "baseline"


def _download_url(url: str, output_template: str) -> bool:
    """Download a specific YouTube URL."""
    cmd = [
        yt_dlp_path(),
        url,
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
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def download_song(track: TrackInfo, output_dir: Path) -> tuple[Path | None, float, str]:
    """Download a song from YouTube using multi-candidate search + scoring.

    Returns (output_path_or_None, match_score_0_100, reason).
    """
    query = f"{track.artist} - {track.title} official audio"
    filename = safe_name(track.artist, track.title)
    output_template = str(output_dir / f"{filename}.%(ext)s")

    # Search for candidates
    candidates = _search_candidates(query, max_results=5)

    if not candidates:
        # Fallback: try direct ytsearch1 download (old behavior)
        cmd = [
            yt_dlp_path(),
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
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return None, 0, "search failed"

        expected = output_dir / f"{filename}.mp3"
        if expected.exists():
            return expected, 50, "fallback (no candidates scored)"
        return None, 0, "download failed"

    # Score all candidates
    scored = []
    for c in candidates:
        score, reason = _score_candidate(track, c)
        url = c.get("webpage_url") or c.get("url") or c.get("original_url")
        if url:
            scored.append((score, reason, url, c))

    if not scored:
        return None, 0, "no valid candidates"

    # Sort by score descending and download the best
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_reason, best_url, best_candidate = scored[0]

    if not _download_url(best_url, output_template):
        return None, best_score, f"download failed ({best_reason})"

    # Find the output file
    expected = output_dir / f"{filename}.mp3"
    if expected.exists():
        return expected, best_score, best_reason

    for p in output_dir.glob(f"{filename}.*"):
        if p.suffix.lower() in (".mp3", ".m4a", ".wav", ".opus"):
            return p, best_score, best_reason

    return None, best_score, f"file not found after download ({best_reason})"


class DownloadWorker(BaseWorker):
    """Downloads a list of tracks in a background thread."""

    track_done = pyqtSignal(int, str, float, str)  # index, path, score, reason

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
                self.track_done.emit(i, "", 0, "skipped")
                continue

            self.progress.emit(i, total, f"Downloading {track.artist} - {track.title}")
            result_path, score, reason = download_song(track, self.output_dir)
            self.track_done.emit(
                i, str(result_path) if result_path else "", score, reason,
            )

        self.progress.emit(total, total, "Done")
        self.finished_work.emit()
