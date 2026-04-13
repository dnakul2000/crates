"""Data models for Spotify track info."""

from dataclasses import dataclass, field


@dataclass
class TrackInfo:
    """Represents a single track extracted from Spotify."""

    title: str
    artist: str
    album: str = ""
    duration_ms: int | None = None
    selected: bool = True
    download_status: str = "pending"  # pending, downloading, done, failed, skipped
