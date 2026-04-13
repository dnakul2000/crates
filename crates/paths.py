"""Centralized path resolution for both development and frozen (PyInstaller) builds."""

import sys
from pathlib import Path


def is_frozen() -> bool:
    """True when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def bundle_dir() -> Path:
    """Root for bundled resources (stylesheet, presets).

    In a frozen build this is the PyInstaller temp directory.
    In development this is the project root.
    """
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Root for user-writable runtime data (Downloads, Stems, Packs).

    In a frozen build this is ~/Documents/Crates so data persists
    across app updates.  In development this is the project root.
    """
    if is_frozen():
        d = Path.home() / "Documents" / "Crates"
        d.mkdir(exist_ok=True)
        return d
    return Path(__file__).resolve().parent.parent


def yt_dlp_path() -> str:
    """Return the path to the yt-dlp executable.

    Prefers a binary bundled inside the app, falls back to PATH.
    """
    if is_frozen():
        bundled = bundle_dir() / "yt-dlp"
        if bundled.exists():
            return str(bundled)
    return "yt-dlp"
