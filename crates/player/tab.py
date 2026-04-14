"""Tab 4: Pad Player — MPC-style sample playback from generated packs."""

import subprocess
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSlot
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
from ..devices import DEFAULT_DEVICE, DEVICE_REGISTRY
from ..widgets.pad_button import StudioPad
from .engine import PlaybackEngine


class PlayerTab(QWidget):
    """Pad player for generated sample packs. Adapts to device profile."""

    # Keyboard mapping: bottom-left origin (first 8 pads always mapped)
    KEY_TO_PAD = {
        Qt.Key.Key_Z: 1,
        Qt.Key.Key_X: 2,
        Qt.Key.Key_C: 3,
        Qt.Key.Key_V: 4,
        Qt.Key.Key_A: 5,
        Qt.Key.Key_S: 6,
        Qt.Key.Key_D: 7,
        Qt.Key.Key_F: 8,
        Qt.Key.Key_Q: 9,
        Qt.Key.Key_W: 10,
        Qt.Key.Key_E: 11,
        Qt.Key.Key_R: 12,
        Qt.Key.Key_1: 13,
        Qt.Key.Key_2: 14,
        Qt.Key.Key_3: 15,
        Qt.Key.Key_4: 16,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = PlaybackEngine()
        self._device = DEFAULT_DEVICE
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
        self.pack_info.setProperty("role", "info")
        left.addWidget(self.pack_info)

        # Bank selector
        bank_label = QLabel("Bank:")
        bank_label.setProperty("heading", True)
        left.addWidget(bank_label)

        self.bank_combo = QComboBox()
        self.bank_combo.addItems(self._device.bank_names)
        self.bank_combo.currentTextChanged.connect(self._on_bank_changed)
        left.addWidget(self.bank_combo)

        # Sample info display
        info_label = QLabel("Current Pad:")
        info_label.setProperty("heading", True)
        left.addWidget(info_label)

        self.sample_info = QLabel("Click a pad to play")
        self.sample_info.setWordWrap(True)
        self.sample_info.setProperty("role", "detail-box")
        self.sample_info.setMinimumHeight(80)
        left.addWidget(self.sample_info)

        left.addStretch()

        # Keyboard hint (adapts to device pad count)
        if self._device.pads_per_bank <= 8:
            hint_text = "Keyboard:\nA S D F  →  pads 5-8\nZ X C V  →  pads 1-4"
        else:
            hint_text = (
                "Keyboard:\n"
                "1 2 3 4  →  pads 13-16\n"
                "Q W E R  →  pads 9-12\n"
                "A S D F  →  pads 5-8\n"
                "Z X C V  →  pads 1-4"
            )
        hint = QLabel(hint_text)
        hint.setProperty("role", "hint-box")
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

        # Build pad grid based on device profile (bottom-left origin)
        grid = QGridLayout()
        grid.setSpacing(10)
        self.pads: list[StudioPad] = []
        cols = self._device.grid_cols
        rows = self._device.grid_rows
        for pad_num in range(1, self._device.pads_per_bank + 1):
            btn = StudioPad(pad_num, size=110)
            btn.pad_clicked.connect(self._trigger_pad)
            self.pads.append(btn)
            row = (rows - 1) - (pad_num - 1) // cols
            col = (pad_num - 1) % cols
            grid.addWidget(btn, row, col)

        right.addLayout(grid)

        # Pad count
        self.pad_count = QLabel("")
        self.pad_count.setProperty("role", "pad-count")
        right.addWidget(self.pad_count)

        right.addStretch()

        main_layout.addLayout(right, 1)

    def refresh_packs(self):
        """Rescan Packs directory and populate the combo box."""
        self.pack_combo.blockSignals(True)
        current = self.pack_combo.currentText()
        self.pack_combo.clear()

        packs = sorted(
            d
            for d in PACKS_DIR.iterdir()
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
                f"Preset: {preset}\nIntensity: {intensity}%\nSamples: {total}"
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

        self.pad_count.setText(
            f"Bank {bank}: {assigned} / {self._device.pads_per_bank} pads loaded"
        )

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
        import logging

        logger = logging.getLogger(__name__)

        try:
            import mido
        except ImportError:
            logger.info("MIDI support not available: mido package not installed")
            self.pack_info.setText(
                self.pack_info.text()
                + "\nMIDI: Not installed (pip install mido python-rtmidi)"
            )
            return

        try:
            available = mido.get_input_names()
            if not available:
                logger.info("No MIDI input devices found")
                self.pack_info.setText(
                    self.pack_info.text() + "\nMIDI: No devices found"
                )
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
            midi_text = f"\nMIDI: {port_name}"
            self.pack_info.setText(self.pack_info.text() + midi_text)
            logger.info(f"MIDI listener started on: {port_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize MIDI: {e}")
            self.pack_info.setText(
                self.pack_info.text() + f"\nMIDI: Error - {str(e)[:50]}"
            )

    def _on_midi_message(self, msg):
        """Handle incoming MIDI note-on messages."""
        if msg.type == "note_on" and msg.velocity > 0:
            # Convert MIDI note to pad number using device's base note
            pad = msg.note - (self._device.base_midi_note - 1)  # 1-indexed
            if 1 <= pad <= self._device.pads_per_bank:
                # Must call UI from main thread
                from PyQt6.QtCore import QMetaObject, Q_ARG

                QMetaObject.invokeMethod(
                    self,
                    "_trigger_pad_from_midi",
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
