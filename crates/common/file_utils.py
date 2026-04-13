"""File and path utilities."""

import re
from pathlib import Path


def sanitize_filename(name: str, max_bytes: int = 200) -> str:
    """Remove problematic characters and enforce filesystem length limit."""
    cleaned = re.sub(r'[<>:"/\\|?*]', "", name).strip()
    # Truncate if UTF-8 encoding exceeds the byte limit (leaves room for extensions)
    while len(cleaned.encode("utf-8")) > max_bytes and cleaned:
        cleaned = cleaned[:-1]
    return cleaned.rstrip()


def safe_name(artist: str, title: str) -> str:
    """Build a consistent sanitized filename from artist + title."""
    return sanitize_filename(f"{artist} - {title}")


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
