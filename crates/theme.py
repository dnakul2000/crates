"""Crates design system — single source of truth for palette and colors."""

from PyQt6.QtGui import QColor


# ── Palette ─────────────────────────────────────────────────────────

BG_DEEP = "#121216"
BG_BASE = "#1a1a20"
BG_CARD = "#222228"
BG_HOVER = "#2c2c34"
BG_INPUT = "#18181e"

BORDER = "#303038"
BORDER_HI = "#444450"
BORDER_SUBTLE = "#282830"

ACCENT = "#D4943C"
ACCENT_DIM = "#B07828"
ACCENT_HOVER = "#E4A84C"

TEXT = "#E4E2DE"
TEXT_SEC = "#9C9A94"
TEXT_MUTED = "#6C6A64"

SUCCESS = "#5CB870"
WARNING = "#D4A43C"
ERROR = "#D45C5C"


# ── Classification colors ───────────────────────────────────────────

CLASSIFICATION_COLORS: dict[str, str] = {
    # Drums — warm earth tones
    "kick": "#A04828",
    "snare": "#B86020",
    "hihat": "#A87828",
    "hihat_closed": "#A87828",
    "hihat_open": "#986C24",
    "cymbal": "#907020",
    "cymbal_crash": "#907020",
    "cymbal_ride": "#806820",
    "tom": "#8C5428",
    "percussion": "#844020",
    # Vocals — muted purple
    "vocal_phrase": "#6840A8",
    "vocal_word": "#5C34A0",
    "vocal_syllable": "#503098",
    "vocal_chop": "#7C58C0",
    "vocal_harmony": "#9480C0",
    "vocal_fx": "#442888",
    "breath": "#303038",
    # Bass — forest green
    "bass_note": "#287858",
    "bass_riff": "#246850",
    "bass_slide": "#205844",
    "bass_pluck": "#30906C",
    "bass_sub": "#1C4838",
    # Guitar — amber/copper
    "guitar_strum": "#C08020",
    "guitar_pick": "#A06818",
    "guitar_riff": "#885418",
    "guitar_chord": "#704814",
    "guitar_bend": "#D49030",
    # Piano — cool purple
    "piano_chord": "#6840A8",
    "piano_melody": "#5C34A0",
    "piano_stab": "#503098",
    "piano_phrase": "#442888",
    # Other / synth — cool blue
    "pad": "#2C4468",
    "lead": "#4070C0",
    "stab": "#3060B0",
    "texture": "#2C2C38",
    "sweep": "#2848A0",
    "riser": "#3058B8",
    "noise": "#242430",
    "chord": "#3060B0",
    "melody_stab": "#4070C0",
    "melody_phrase": "#2848A0",
    # Fallback
    "": "#222228",
}


# ── Color utilities ─────────────────────────────────────────────────

def _parse_hex(hex_color: str) -> tuple[int, int, int]:
    """Parse a '#RRGGBB' string into (r, g, b) ints."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return 21, 21, 23  # fallback to BG_CARD equivalent
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def lighten(hex_color: str, amount: int = 50) -> str:
    """Lighten a hex color by adding *amount* to each channel."""
    r, g, b = _parse_hex(hex_color)
    r = min(255, r + amount)
    g = min(255, g + amount)
    b = min(255, b + amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def darken(hex_color: str, amount: int = 30) -> str:
    """Darken a hex color by subtracting *amount* from each channel."""
    r, g, b = _parse_hex(hex_color)
    r = max(0, r - amount)
    g = max(0, g - amount)
    b = max(0, b - amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def with_alpha(hex_color: str, alpha: int) -> QColor:
    """Return a QColor from a hex string with the given alpha (0-255)."""
    r, g, b = _parse_hex(hex_color)
    return QColor(r, g, b, alpha)


def lerp_color(color_a: str, color_b: str, t: float) -> QColor:
    """Linearly interpolate between two hex colors. t=0 -> a, t=1 -> b."""
    r1, g1, b1 = _parse_hex(color_a)
    r2, g2, b2 = _parse_hex(color_b)
    t = max(0.0, min(1.0, t))
    return QColor(
        int(r1 + (r2 - r1) * t),
        int(g1 + (g2 - g1) * t),
        int(b1 + (b2 - b1) * t),
    )
