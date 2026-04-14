"""Device profiles for hardware pad controllers."""

from dataclasses import dataclass, field
from typing import Final


# MIDI note constants
DEFAULT_BASE_MIDI_NOTE: Final[int] = 36  # C2 - standard for drum pads

# Grid layout constants
DEFAULT_GRID_COLS_SMALL: Final[int] = 4
DEFAULT_GRID_COLS_LARGE: Final[int] = 4

# Pad count thresholds
SMALL_PAD_THRESHOLD: Final[int] = 8


@dataclass
class DeviceProfile:
    """Describes a hardware pad controller's layout."""

    name: str
    display_name: str
    pads_per_bank: int
    num_banks: int
    bank_names: list[str] = field(default_factory=list)
    base_midi_note: int = DEFAULT_BASE_MIDI_NOTE

    @property
    def max_slots(self) -> int:
        return self.pads_per_bank * self.num_banks

    @property
    def grid_cols(self) -> int:
        """Number of columns for the pad grid UI."""
        if self.pads_per_bank <= SMALL_PAD_THRESHOLD:
            return DEFAULT_GRID_COLS_SMALL
        return DEFAULT_GRID_COLS_LARGE

    @property
    def grid_rows(self) -> int:
        """Number of rows for the pad grid UI."""
        return (self.pads_per_bank + self.grid_cols - 1) // self.grid_cols


# --- Built-in device profiles ---

MPK_MINI_MK3 = DeviceProfile(
    name="mpk_mini_mk3",
    display_name="Akai MPK Mini MK3",
    pads_per_bank=8,
    num_banks=2,
    bank_names=["A", "B"],
    base_midi_note=DEFAULT_BASE_MIDI_NOTE,
)

MPC_MINI_MK3 = DeviceProfile(
    name="mpc_mini_mk3",
    display_name="Akai MPC Mini MK3",
    pads_per_bank=16,
    num_banks=8,
    bank_names=list("ABCDEFGH"),
    base_midi_note=DEFAULT_BASE_MIDI_NOTE,
)

# Default device — user's MPK Mini MK3
DEFAULT_DEVICE = MPK_MINI_MK3

DEVICE_REGISTRY: dict[str, DeviceProfile] = {
    d.name: d for d in [MPK_MINI_MK3, MPC_MINI_MK3]
}


def get_device(name: str) -> DeviceProfile:
    """Look up a device profile by name.

    Args:
        name: The device profile identifier.

    Returns:
        The matching DeviceProfile.

    Raises:
        KeyError: If the device name is not found in the registry.
    """
    if name not in DEVICE_REGISTRY:
        available = ", ".join(DEVICE_REGISTRY.keys())
        raise KeyError(f"Device '{name}' not found. Available devices: {available}")
    return DEVICE_REGISTRY[name]


def get_device_or_default(name: str | None) -> DeviceProfile:
    """Look up a device profile by name, falling back to default if not found.

    Args:
        name: The device profile identifier, or None for default.

    Returns:
        The matching DeviceProfile or DEFAULT_DEVICE if not found.
    """
    if name is None:
        return DEFAULT_DEVICE
    return DEVICE_REGISTRY.get(name, DEFAULT_DEVICE)
