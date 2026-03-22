"""File and path utilities."""

import re
from pathlib import Path


def sanitize_filename(name: str) -> str:
    """Remove characters that are problematic in filenames."""
    return re.sub(r'[<>:"/\\|?*]', "", name).strip()


def safe_name(artist: str, title: str) -> str:
    """Build a consistent sanitized filename from artist + title."""
    return sanitize_filename(f"{artist} - {title}")


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
