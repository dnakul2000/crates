"""yt-dlp download logic with QThread worker.

Multi-candidate search with scoring to avoid downloading wrong songs.
"""

import difflib
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Final

from PyQt6.QtCore import pyqtSignal

from ..common.file_utils import safe_name
from ..common.workers import BaseWorker
from ..config import DOWNLOADS_DIR
from ..paths import yt_dlp_path
from ..spotify.models import TrackInfo

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Search and Download Constants
# =============================================================================

# Search configuration
DEFAULT_MAX_RESULTS: Final[int] = 5
SEARCH_TIMEOUT_S: Final[int] = 30
DOWNLOAD_TIMEOUT_S: Final[int] = 120

# Security limits
MAX_RESPONSE_SIZE_BYTES: Final[int] = 10 * 1024 * 1024  # 10 MB
MAX_JSON_LINE_SIZE: Final[int] = 100000  # 100 KB per JSON object

# Words that indicate a non-original version (unless the source track also has them)
_REJECT_KEYWORDS: set[str] = {
    "cover",
    "karaoke",
    "tutorial",
    "reaction",
    "lesson",
    "instrumental",
}
_VARIANT_KEYWORDS: set[str] = {
    "remix",
    "live",
    "acoustic",
    "demo",
    "slowed",
    "sped up",
    "reverb",
}


# =============================================================================
# Scoring Weights and Thresholds
# =============================================================================

# Score baseline
SCORE_BASELINE: Final[float] = 50.0
SCORE_MIN: Final[float] = 0.0
SCORE_MAX: Final[float] = 100.0

# Duration match weights
DURATION_MATCH_BONUS_LARGE: Final[float] = 30.0  # < 5s diff
DURATION_MATCH_BONUS_MEDIUM: Final[float] = 15.0  # < 15s diff
DURATION_MISMATCH_PENALTY_SMALL: Final[float] = 10.0  # < 30s diff
DURATION_MISMATCH_PENALTY_LARGE: Final[float] = 50.0  # >= 30s diff

DURATION_THRESHOLD_EXCELLENT: Final[float] = 5.0  # seconds
DURATION_THRESHOLD_GOOD: Final[float] = 15.0  # seconds
DURATION_THRESHOLD_ACCEPTABLE: Final[float] = 30.0  # seconds

# Title match weights
TITLE_MATCH_MAX_BONUS: Final[float] = 20.0
TITLE_MATCH_THRESHOLD: Final[float] = 0.6

# Artist match weights
ARTIST_MATCH_MAX_BONUS: Final[float] = 15.0
ARTIST_MATCH_THRESHOLD: Final[float] = 0.5

# Rejection penalties
REJECT_KEYWORD_PENALTY: Final[float] = 40.0
VARIANT_KEYWORD_PENALTY: Final[float] = 15.0

# View count bonuses
VIEW_COUNT_BONUS_LARGE: Final[int] = 5  # > 1M views
VIEW_COUNT_BONUS_MEDIUM: Final[int] = 3  # > 100K views
VIEW_COUNT_BONUS_SMALL: Final[int] = 1  # > 10K views
VIEW_COUNT_THRESHOLD_LARGE: Final[int] = 1_000_000
VIEW_COUNT_THRESHOLD_MEDIUM: Final[int] = 100_000
VIEW_COUNT_THRESHOLD_SMALL: Final[int] = 10_000

# Official bonus
OFFICIAL_BONUS: Final[int] = 5

# Minimum acceptable score
MIN_ACCEPTABLE_SCORE: Final[float] = 50.0


# =============================================================================
# Error Types
# =============================================================================


class DownloadError(Exception):
    """Raised when a download operation fails."""

    def __init__(self, message: str, reason: str = "unknown"):
        super().__init__(message)
        self.reason = reason


class SearchError(DownloadError):
    """Raised when a search operation fails."""

    pass


# =============================================================================
# Functions
# =============================================================================


def _sanitize_query(query: str) -> str:
    """Sanitize user input to prevent command injection.

    SECURITY FIX: Whitelist approach - only allow alphanumeric, spaces,
    hyphens, and safe punctuation. Removes shell metacharacters.
    """
    # Allow alphanumeric, spaces, and safe punctuation common in music titles
    # This whitelist approach prevents command injection
    return re.sub(r'[^\w\s\-\'"().&]', "", query)


def _search_candidates(
    query: str, max_results: int = DEFAULT_MAX_RESULTS
) -> list[dict]:
    """Search YouTube for candidates and return their metadata without downloading.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of candidate metadata dictionaries.

    Raises:
        SearchError: If the search command fails or times out.
    """
    # SECURITY FIX: Sanitize user input to prevent command injection
    safe_query = _sanitize_query(query)
    cmd = [
        yt_dlp_path(),
        f"ytsearch{max_results}:{safe_query}",
        "--dump-json",
        "--flat-playlist",
        "--no-warnings",
        "--quiet",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SEARCH_TIMEOUT_S,
        )
        result.check_returncode()
        candidates = []
        # SECURITY FIX: Limit total response size to prevent DoS (10MB max)
        total_size = len(result.stdout.encode("utf-8"))
        if total_size > MAX_RESPONSE_SIZE_BYTES:
            logger.warning(f"Search response too large: {total_size} bytes")
            return []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                # SECURITY FIX: Validate JSON structure and limit size per line
                if len(line) > MAX_JSON_LINE_SIZE:
                    continue
                parsed = json.loads(line)
                # SECURITY FIX: Validate expected fields exist and have correct types
                if not isinstance(parsed, dict):
                    continue
                # Validate required fields have string types
                if "title" in parsed and not isinstance(parsed.get("title"), str):
                    continue
                if "url" in parsed and not isinstance(parsed.get("url"), str):
                    continue
                candidates.append(parsed)
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse candidate JSON: {e}")
                continue
        return candidates
    except subprocess.TimeoutExpired as e:
        logger.error(f"Search timed out after {SEARCH_TIMEOUT_S}s")
        raise SearchError(
            f"Search timed out after {SEARCH_TIMEOUT_S}s", "timeout"
        ) from e
    except FileNotFoundError as e:
        logger.error(f"yt-dlp not found at {yt_dlp_path()}")
        raise SearchError("yt-dlp not found", "not_found") from e
    except subprocess.CalledProcessError as e:
        logger.error(f"Search command failed: {e.stderr}")
        raise SearchError(f"Search failed: {e.stderr}", "command_failed") from e


def _fuzzy_match(a: str, b: str) -> float:
    """Fuzzy string similarity (0-1)."""
    a = re.sub(r"[^\w\s]", "", a.lower()).strip()
    b = re.sub(r"[^\w\s]", "", b.lower()).strip()
    return difflib.SequenceMatcher(None, a, b).ratio()


def _score_candidate(track: TrackInfo, candidate: dict) -> tuple[float, str]:
    """Score a YouTube candidate against the Spotify track.

    Returns:
        Tuple of (score 0-100, reason string).
    """
    score = SCORE_BASELINE
    reasons = []

    title = candidate.get("title", "") or candidate.get("fulltitle", "")
    uploader = candidate.get("uploader", "") or candidate.get("channel", "")
    duration = candidate.get("duration")  # seconds
    view_count = candidate.get("view_count", 0) or 0
    title_lower = title.lower()

    # --- Duration match (heaviest weight) ---
    if track.duration_ms and duration:
        track_duration_s = track.duration_ms / 1000
        diff = abs(duration - track_duration_s)
        if diff < DURATION_THRESHOLD_EXCELLENT:
            score += DURATION_MATCH_BONUS_LARGE
            reasons.append(f"duration match ({diff:.0f}s off)")
        elif diff < DURATION_THRESHOLD_GOOD:
            score += DURATION_MATCH_BONUS_MEDIUM
            reasons.append(f"duration close ({diff:.0f}s off)")
        elif diff < DURATION_THRESHOLD_ACCEPTABLE:
            score -= DURATION_MISMATCH_PENALTY_SMALL
            reasons.append(f"duration mismatch ({diff:.0f}s off)")
        else:
            score -= DURATION_MISMATCH_PENALTY_LARGE
            reasons.append(f"duration way off ({diff:.0f}s)")

    # --- Title match ---
    title_sim = _fuzzy_match(track.title, title)
    score += title_sim * TITLE_MATCH_MAX_BONUS
    if title_sim > TITLE_MATCH_THRESHOLD:
        reasons.append(f"title match ({title_sim:.0%})")

    # --- Artist match ---
    # Check both the uploader/channel and whether artist appears in title
    artist_in_uploader = _fuzzy_match(track.artist, uploader)
    artist_in_title = _fuzzy_match(track.artist, title)
    artist_score = max(artist_in_uploader, artist_in_title)
    score += artist_score * ARTIST_MATCH_MAX_BONUS
    if artist_score > ARTIST_MATCH_THRESHOLD:
        reasons.append(f"artist match ({artist_score:.0%})")

    # --- Reject filters ---
    track_title_lower = track.title.lower()
    for keyword in _REJECT_KEYWORDS:
        if keyword in title_lower and keyword not in track_title_lower:
            score -= REJECT_KEYWORD_PENALTY
            reasons.append(f"reject: '{keyword}' in title")

    # --- Variant penalty (remix/live etc unless source also has it) ---
    for keyword in _VARIANT_KEYWORDS:
        if keyword in title_lower and keyword not in track_title_lower:
            score -= VARIANT_KEYWORD_PENALTY
            reasons.append(f"variant: '{keyword}'")

    # --- View count tiebreaker ---
    if view_count > VIEW_COUNT_THRESHOLD_LARGE:
        score += VIEW_COUNT_BONUS_LARGE
    elif view_count > VIEW_COUNT_THRESHOLD_MEDIUM:
        score += VIEW_COUNT_BONUS_MEDIUM
    elif view_count > VIEW_COUNT_THRESHOLD_SMALL:
        score += VIEW_COUNT_BONUS_SMALL

    # --- "Official" bonus ---
    if "official" in title_lower:
        score += OFFICIAL_BONUS
        reasons.append("official in title")

    return max(SCORE_MIN, min(SCORE_MAX, score)), "; ".join(
        reasons
    ) if reasons else "baseline"


def _download_url(url: str, output_template: str) -> bool:
    """Download a specific YouTube URL.

    Args:
        url: The YouTube URL to download.
        output_template: The output file template.

    Returns:
        True if download succeeded, False otherwise.
    """
    # SECURITY FIX: Validate URL format to prevent command injection
    # Only allow http/https URLs with valid characters
    if not re.match(r"^https?://[\w\-.]+/", url):
        logger.warning(f"Invalid URL format: {url}")
        return False
    cmd = [
        yt_dlp_path(),
        url,
        "-x",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "-o",
        output_template,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]
    try:
        subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT_S
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Download failed for {url}: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"yt-dlp not found at {yt_dlp_path()}")
        return False
    except subprocess.TimeoutExpired:
        logger.warning(f"Download timed out for {url} after {DOWNLOAD_TIMEOUT_S}s")
        return False


def download_song(track: TrackInfo, output_dir: Path) -> tuple[Path | None, float, str]:
    """Download a song from YouTube using multi-candidate search + scoring.

    Args:
        track: The track information to search for.
        output_dir: Directory to save the downloaded file.

    Returns:
        Tuple of (output_path_or_None, match_score_0_100, reason).

    Raises:
        DownloadError: If the search or download fails critically.
    """
    query = f"{track.artist} - {track.title} official audio"
    filename = safe_name(track.artist, track.title)
    output_template = str(output_dir / f"{filename}.%(ext)s")

    # Search for candidates
    try:
        candidates = _search_candidates(query, max_results=DEFAULT_MAX_RESULTS)
    except SearchError as e:
        logger.error(f"Search failed for '{track.title}': {e}")
        candidates = []

    if not candidates:
        # Fallback: try direct ytsearch1 download (old behavior)
        logger.info(
            f"No candidates found for '{track.title}', attempting fallback download"
        )
        # SECURITY FIX: Sanitize query to prevent command injection
        safe_query = _sanitize_query(query)
        cmd = [
            yt_dlp_path(),
            f"ytsearch1:{safe_query}",
            "-x",
            "--audio-format",
            "mp3",
            "--audio-quality",
            "0",
            "-o",
            output_template,
            "--no-playlist",
            "--quiet",
            "--no-warnings",
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=DOWNLOAD_TIMEOUT_S,
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ) as e:
            logger.error(f"Fallback download failed for '{track.title}': {e}")
            return None, 0, "search failed"

        expected = output_dir / f"{filename}.mp3"
        if expected.exists():
            return expected, MIN_ACCEPTABLE_SCORE, "fallback (no candidates scored)"
        return None, 0, "download failed"

    # Score all candidates
    scored = []
    for c in candidates:
        score, reason = _score_candidate(track, c)
        url = c.get("webpage_url") or c.get("url") or c.get("original_url")
        if url:
            scored.append((score, reason, url, c))
            logger.debug(f"Candidate scored {score:.1f}: {url} ({reason})")

    if not scored:
        logger.warning(f"No valid candidates with URLs for '{track.title}'")
        return None, 0, "no valid candidates"

    # Sort by score descending and download the best
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_reason, best_url, best_candidate = scored[0]
    logger.info(
        f"Best candidate for '{track.title}': score={best_score:.1f}, url={best_url}"
    )

    if not _download_url(best_url, output_template):
        return None, best_score, f"download failed ({best_reason})"

    # Find the output file
    expected = output_dir / f"{filename}.mp3"
    if expected.exists():
        return expected, best_score, best_reason

    for p in output_dir.glob(f"{filename}.*"):
        if p.suffix.lower() in (".mp3", ".m4a", ".wav", ".opus"):
            return p, best_score, best_reason

    logger.error(f"File not found after download for '{track.title}'")
    return None, best_score, f"file not found after download ({best_reason})"


class DownloadWorker(BaseWorker):
    """Downloads a list of tracks in a background thread."""

    track_done = pyqtSignal(int, str, float, str)  # index, path, score, reason

    def __init__(
        self, tracks: list[TrackInfo], output_dir: Path | None = None, parent=None
    ):
        super().__init__(parent)
        self.tracks = tracks
        self.output_dir = output_dir or DOWNLOADS_DIR

    def run(self):
        total = len(self.tracks)
        for i, track in enumerate(self.tracks):
            if self.is_cancelled:
                logger.info("Download worker cancelled")
                break
            if not track.selected:
                self.track_done.emit(i, "", 0, "skipped")
                continue

            self.progress.emit(i, total, f"Downloading {track.artist} - {track.title}")
            try:
                result_path, score, reason = download_song(track, self.output_dir)
                self.track_done.emit(
                    i,
                    str(result_path) if result_path else "",
                    score,
                    reason,
                )
            except DownloadError as e:
                logger.error(f"Download error for '{track.title}': {e}")
                self.error.emit(f"Failed to download {track.title}: {e.reason}")
                self.track_done.emit(i, "", 0, f"error: {e.reason}")
            except Exception as e:
                logger.exception(f"Unexpected error downloading '{track.title}'")
                self.error.emit(f"Unexpected error downloading {track.title}: {e}")
                self.track_done.emit(i, "", 0, f"unexpected error: {e}")

        self.progress.emit(total, total, "Done")
        self.finished_work.emit()
