"""Tab 4: Pad Player — MPC-style sample playback from generated packs."""

import subprocess
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import PACKS_DIR
from .engine import PlaybackEngine


# Reuse the same color map as the sampler's PadButton
PAD_COLORS = {
    "kick": "#b34116",
    "snare": "#ca5a0c",
    "hihat": "#b8720a",
    "cymbal": "#9a5e10",
    "percussion": "#8c3f0e",
    "vocal_phrase": "#7c3aed",
    "vocal_word": "#6d28d9",
    "vocal_syllable": "#5b21b6",
    "vocal_chop": "#8b5cf6",
    "breath": "#27272a",
    "bass_note": "#059669",
    "bass_riff": "#047857",
    "bass_slide": "#065f46",
    "chord": "#2563eb",
    "melody_stab": "#3b82f6",
    "melody_phrase": "#1e40af",
    "texture": "#1f1f23",
    "noise": "#18181b",
    "": "#151517",
}


class PlayPadButton(QPushButton):
    """A playable pad button with press animation."""

    def __init__(self, pad_number: int, parent=None):
        super().__init__(str(pad_number), parent)
        self.pad_number = pad_number
        self.classification = ""
        self._base_color = "#151517"
        self._is_playing = False
        self.setFixedSize(100, 100)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_style()

    def set_sample(self, classification: str, label: str, info: str):
        self.classification = classification
        self._base_color = PAD_COLORS.get(classification, "#151517")
        # Show classification as main text, short
        display = classification.replace("_", "\n") if classification else str(self.pad_number)
        self.setText(display)
        self.setToolTip(info)
        self._update_style()

    def clear_sample(self):
        self.classification = ""
        self._base_color = "#151517"
        self.setText(str(self.pad_number))
        self.setToolTip("")
        self._update_style()

    def flash_play(self):
        """Visual feedback when pad is triggered."""
        self._is_playing = True
        self._update_style()
        QTimer.singleShot(150, self._end_flash)

    def _end_flash(self):
        self._is_playing = False
        self._update_style()

    def _update_style(self):
        color = self._base_color
        border = "#F59E0B" if self._is_playing else (
            "#3f3f46" if self.classification else "#27272a"
        )
        border_width = "2px" if self._is_playing else "1px"
        brightness = "brightness(1.4)" if self._is_playing else "brightness(1.0)"

        # For the flash effect, lighten the background color
        if self._is_playing and self.classification:
            # Approximate brightening by blending with white
            bg = self._lighten_color(color)
        else:
            bg = color

        self.setStyleSheet(
            f"QPushButton {{ background-color: {bg}; color: #fafafa; "
            f"border: {border_width} solid {border}; border-radius: 12px; "
            f"font-family: 'SF Mono', 'Menlo', monospace; font-weight: 600; font-size: 11px; "
            f"text-align: center; }}"
            f"QPushButton:hover {{ border-color: #F59E0B; }}"
            f"QPushButton:pressed {{ border-color: #FBBF24; border-width: 2px; }}"
        )

    @staticmethod
    def _lighten_color(hex_color: str) -> str:
        """Lighten a hex color by ~30% for flash effect."""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) != 6:
            return f"#{hex_color}"
        r = min(255, int(hex_color[0:2], 16) + 50)
        g = min(255, int(hex_color[2:4], 16) + 50)
        b = min(255, int(hex_color[4:6], 16) + 50)
        return f"#{r:02x}{g:02x}{b:02x}"


class PlayerTab(QWidget):
    """MPC-style pad player for generated sample packs."""

    # Keyboard mapping: bottom-left origin matching MPC layout
    # Row 1 (bottom): z x c v -> pads 1-4
    # Row 2: a s d f -> pads 5-8
    # Row 3: q w e r -> pads 9-12
    # Row 4 (top): 1 2 3 4 -> pads 13-16
    KEY_TO_PAD = {
        Qt.Key.Key_Z: 1, Qt.Key.Key_X: 2, Qt.Key.Key_C: 3, Qt.Key.Key_V: 4,
        Qt.Key.Key_A: 5, Qt.Key.Key_S: 6, Qt.Key.Key_D: 7, Qt.Key.Key_F: 8,
        Qt.Key.Key_Q: 9, Qt.Key.Key_W: 10, Qt.Key.Key_E: 11, Qt.Key.Key_R: 12,
        Qt.Key.Key_1: 13, Qt.Key.Key_2: 14, Qt.Key.Key_3: 15, Qt.Key.Key_4: 16,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = PlaybackEngine()
        self._midi_port = None
        self._setup_ui()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._start_midi_listener()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(24)

        # === LEFT SIDE: Pack selector + info ===
        left = QVBoxLayout()
        left.setSpacing(12)

        pack_label = QLabel("Sample Pack:")
        pack_label.setProperty("heading", True)
        left.addWidget(pack_label)

        self.pack_combo = QComboBox()
        self.pack_combo.currentIndexChanged.connect(self._on_pack_changed)
        left.addWidget(self.pack_combo)

        self.pack_info = QLabel("")
        self.pack_info.setWordWrap(True)
        self.pack_info.setStyleSheet("color: #71717a; font-size: 12px; padding: 4px;")
        left.addWidget(self.pack_info)

        # Bank selector
        bank_label = QLabel("Bank:")
        bank_label.setProperty("heading", True)
        left.addWidget(bank_label)

        self.bank_combo = QComboBox()
        self.bank_combo.addItems(list("ABCDEFGH"))
        self.bank_combo.currentTextChanged.connect(self._on_bank_changed)
        left.addWidget(self.bank_combo)

        # Sample info display
        info_label = QLabel("Current Pad:")
        info_label.setProperty("heading", True)
        left.addWidget(info_label)

        self.sample_info = QLabel("Click a pad to play")
        self.sample_info.setWordWrap(True)
        self.sample_info.setStyleSheet(
            "color: #a1a1aa; font-family: 'SF Mono', 'Menlo', monospace; "
            "font-size: 12px; padding: 8px; background-color: #0f0f11; "
            "border: 1px solid #1a1a1e; border-radius: 8px;"
        )
        self.sample_info.setMinimumHeight(80)
        left.addWidget(self.sample_info)

        left.addStretch()

        # Keyboard hint
        hint = QLabel(
            "Keyboard:\n"
            "1 2 3 4  →  pads 13-16\n"
            "Q W E R  →  pads 9-12\n"
            "A S D F  →  pads 5-8\n"
            "Z X C V  →  pads 1-4"
        )
        hint.setStyleSheet(
            "color: #3f3f46; font-family: 'SF Mono', 'Menlo', monospace; "
            "font-size: 11px; padding: 8px; background-color: #0f0f11; "
            "border: 1px solid #1a1a1e; border-radius: 8px;"
        )
        left.addWidget(hint)

        # Stop all button
        stop_btn = QPushButton("Stop All")
        stop_btn.setProperty("danger", True)
        stop_btn.clicked.connect(self.engine.stop_all)
        left.addWidget(stop_btn)

        # Open pack folder
        open_btn = QPushButton("Open Pack Folder")
        open_btn.setProperty("secondary", True)
        open_btn.clicked.connect(self._open_pack_folder)
        left.addWidget(open_btn)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(240)
        main_layout.addWidget(left_widget)

        # === RIGHT SIDE: Pad Grid ===
        right = QVBoxLayout()
        right.setSpacing(12)

        grid_label = QLabel("Pads:")
        grid_label.setProperty("heading", True)
        right.addWidget(grid_label)

        # 4x4 grid (bottom-left origin like MPC)
        grid = QGridLayout()
        grid.setSpacing(8)
        self.pads: list[PlayPadButton] = []
        for pad_num in range(1, 17):
            btn = PlayPadButton(pad_num)
            btn.clicked.connect(lambda checked, p=pad_num: self._trigger_pad(p))
            self.pads.append(btn)
            row = 3 - (pad_num - 1) // 4
            col = (pad_num - 1) % 4
            grid.addWidget(btn, row, col)

        right.addLayout(grid)

        # Pad count
        self.pad_count = QLabel("")
        self.pad_count.setStyleSheet(
            "font-family: 'SF Mono', 'Menlo', monospace; font-size: 12px; color: #52525b;"
        )
        right.addWidget(self.pad_count)

        right.addStretch()

        main_layout.addLayout(right, 1)

    def refresh_packs(self):
        """Rescan Packs directory and populate the combo box."""
        self.pack_combo.blockSignals(True)
        current = self.pack_combo.currentText()
        self.pack_combo.clear()

        packs = sorted(
            d for d in PACKS_DIR.iterdir()
            if d.is_dir() and (d / "pack_manifest.json").exists()
        )

        for pack_dir in packs:
            self.pack_combo.addItem(pack_dir.name, str(pack_dir))

        # Restore selection if possible
        if current:
            idx = self.pack_combo.findText(current)
            if idx >= 0:
                self.pack_combo.setCurrentIndex(idx)

        self.pack_combo.blockSignals(False)

        # Load current selection
        if self.pack_combo.count() > 0:
            self._on_pack_changed(self.pack_combo.currentIndex())
        else:
            self.pack_info.setText("No packs found. Generate one first (Tab 3).")

    def _on_pack_changed(self, index: int):
        pack_path = self.pack_combo.itemData(index)
        if not pack_path:
            return

        pack_dir = Path(pack_path)
        if self.engine.load_pack(pack_dir):
            manifest = self.engine.manifest
            total = manifest.get("total_samples", 0)
            preset = manifest.get("preset", "Unknown")
            intensity = manifest.get("intensity", 0)
            self.pack_info.setText(
                f"Preset: {preset}\n"
                f"Intensity: {intensity}%\n"
                f"Samples: {total}"
            )
            self._refresh_pads()
        else:
            self.pack_info.setText("Failed to load pack.")

    def _on_bank_changed(self, bank: str):
        self._refresh_pads()

    def _refresh_pads(self):
        bank = self.bank_combo.currentText()
        assigned = 0

        for i, pad_btn in enumerate(self.pads):
            sample = self.engine.get_sample(bank, i + 1)
            if sample:
                duration = f"{sample.duration_ms:.0f}ms" if sample.duration_ms else ""
                pitch = sample.pitch_name or ""
                info_parts = [f"Pad {bank}{i + 1:02d}", sample.classification]
                if pitch:
                    info_parts.append(pitch)
                if duration:
                    info_parts.append(duration)
                pad_btn.set_sample(
                    sample.classification,
                    sample.classification,
                    " | ".join(info_parts),
                )
                assigned += 1
            else:
                pad_btn.clear_sample()

        self.pad_count.setText(f"Bank {bank}: {assigned} / 16 pads loaded")

    def _trigger_pad(self, pad_number: int):
        bank = self.bank_combo.currentText()
        sample = self.engine.get_sample(bank, pad_number)

        # Visual flash
        self.pads[pad_number - 1].flash_play()

        if sample:
            self.engine.play(bank, pad_number)
            self.sample_info.setText(
                f"Pad {bank}{pad_number:02d}\n"
                f"{sample.classification}\n"
                f"{sample.pitch_name or ''}"
                f"{'  ' + str(int(sample.duration_ms)) + 'ms' if sample.duration_ms else ''}"
            )
        else:
            self.sample_info.setText(f"Pad {bank}{pad_number:02d}\n(empty)")

    def keyPressEvent(self, event):
        """Handle keyboard input for pad triggering."""
        key = event.key()
        if key in self.KEY_TO_PAD and not event.isAutoRepeat():
            self._trigger_pad(self.KEY_TO_PAD[key])
        else:
            super().keyPressEvent(event)

    def _open_pack_folder(self):
        import platform
        pack_path = self.pack_combo.currentData()
        path = pack_path if pack_path else str(PACKS_DIR)
        if platform.system() == "Darwin":
            subprocess.Popen(["open", path])

    def _start_midi_listener(self):
        """Start listening for MIDI input from MPC Mini MK3 or other controller."""
        try:
            import mido
            available = mido.get_input_names()
            if not available:
                return

            # Prefer MPC Mini, fall back to first available
            port_name = None
            for name in available:
                if "mpc" in name.lower() or "mini" in name.lower():
                    port_name = name
                    break
            if port_name is None:
                port_name = available[0]

            self._midi_port = mido.open_input(port_name, callback=self._on_midi_message)
            self.pack_info.setText(
                self.pack_info.text() + f"\nMIDI: {port_name}"
            )
        except (ImportError, Exception):
            pass  # mido/rtmidi not installed or no MIDI devices

    def _on_midi_message(self, msg):
        """Handle incoming MIDI note-on messages."""
        if msg.type == "note_on" and msg.velocity > 0:
            # MPC Mini MK3 sends notes 36-51 for pads 1-16
            pad = msg.note - 35  # convert to 1-indexed pad number
            if 1 <= pad <= 16:
                # Must call UI from main thread
                from PyQt6.QtCore import QMetaObject, Q_ARG
                QMetaObject.invokeMethod(
                    self, "_trigger_pad_from_midi",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(int, pad),
                )

    @pyqtSlot(int)
    def _trigger_pad_from_midi(self, pad_number: int):
        """Trigger a pad from MIDI input (called on main thread)."""
        self._trigger_pad(pad_number)

    def cleanup(self):
        self.engine.cleanup()
        if hasattr(self, "_midi_port") and self._midi_port:
            try:
                self._midi_port.close()
            except Exception:
                pass
