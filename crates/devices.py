"""Device profiles for hardware pad controllers."""

from dataclasses import dataclass, field


@dataclass
class DeviceProfile:
    """Describes a hardware pad controller's layout."""

    name: str
    display_name: str
    pads_per_bank: int
    num_banks: int
    bank_names: list[str] = field(default_factory=list)
    base_midi_note: int = 36

    @property
    def max_slots(self) -> int:
        return self.pads_per_bank * self.num_banks

    @property
    def grid_cols(self) -> int:
        """Number of columns for the pad grid UI."""
        if self.pads_per_bank <= 8:
            return 4
        return 4

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
    base_midi_note=36,
)

MPC_MINI_MK3 = DeviceProfile(
    name="mpc_mini_mk3",
    display_name="Akai MPC Mini MK3",
    pads_per_bank=16,
    num_banks=8,
    bank_names=list("ABCDEFGH"),
    base_midi_note=36,
)

# Default device — user's MPK Mini MK3
DEFAULT_DEVICE = MPK_MINI_MK3

DEVICE_REGISTRY: dict[str, DeviceProfile] = {
    d.name: d for d in [MPK_MINI_MK3, MPC_MINI_MK3]
}


def get_device(name: str) -> DeviceProfile:
    """Look up a device profile by name."""
    return DEVICE_REGISTRY[name]
