"""Spotify URL parsing and track extraction.

Supports playlists, albums, and single tracks via embed page scraping (no API key).
Adapted from Karaoke/usdx-loader.py.
"""

import json
import re
import urllib.request
from typing import List

from .models import TrackInfo

_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"


def parse_spotify_url(url: str) -> tuple[str, str]:
    """Parse a Spotify URL and return (type, id).

    Supports:
        https://open.spotify.com/playlist/37i9d...
        https://open.spotify.com/album/6dVIq...
        https://open.spotify.com/track/4uLU6...
        spotify:playlist:37i9d...
    """
    url = url.strip()

    for kind in ("playlist", "album", "track"):
        match = re.search(rf"{kind}[/:]([a-zA-Z0-9]+)", url)
        if match:
            return kind, match.group(1)

    raise ValueError(f"Could not parse Spotify URL: {url}")


def _fetch_embed_data(entity_type: str, entity_id: str) -> dict:
    """Fetch and parse the __NEXT_DATA__ blob from a Spotify embed page."""
    url = f"https://open.spotify.com/embed/{entity_type}/{entity_id}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    match = re.search(
        r'<script\s+id="__NEXT_DATA__"\s+type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    if not match:
        return {}

    return json.loads(match.group(1))


def _extract_tracks_from_embed(data: dict) -> list[TrackInfo]:
    """Navigate the embed JSON to extract track list."""
    try:
        entity = data["props"]["pageProps"]["state"]["data"]["entity"]
    except (KeyError, TypeError):
        return []

    tracks = []

    # Playlist / album style: has trackList
    track_list = entity.get("trackList", [])
    if track_list:
        for item in track_list:
            title = item.get("title", "")
            artist = item.get("subtitle", "")
            duration = item.get("duration", None)
            if title and artist:
                tracks.append(TrackInfo(
                    title=title,
                    artist=artist,
                    album=entity.get("name", ""),
                    duration_ms=duration,
                ))
        return tracks

    # Single track style: entity itself is the track
    title = entity.get("title") or entity.get("name", "")
    artist = entity.get("subtitle", "")
    if title and artist:
        tracks.append(TrackInfo(
            title=title,
            artist=artist,
            album=entity.get("albumName", ""),
        ))

    return tracks


def extract_tracks(url: str) -> list[TrackInfo]:
    """Extract tracks from any Spotify URL (playlist, album, or track)."""
    entity_type, entity_id = parse_spotify_url(url)
    data = _fetch_embed_data(entity_type, entity_id)
    if not data:
        raise RuntimeError(f"Failed to fetch data for {entity_type}/{entity_id}")
    tracks = _extract_tracks_from_embed(data)
    if not tracks:
        raise RuntimeError(f"No tracks found in {entity_type}/{entity_id}")
    return tracks
