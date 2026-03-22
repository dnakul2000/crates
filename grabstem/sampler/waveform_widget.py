"""Waveform display widget — shows stem audio with chop boundaries overlaid."""

import numpy as np
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QLinearGradient
from PyQt6.QtWidgets import QWidget


class WaveformWidget(QWidget):
    """Renders a waveform with chop regions color-coded by classification."""

    chop_clicked = pyqtSignal(int)  # emits chop index when clicked

    # Classification colors matching PadButton.COLORS
    COLORS = {
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
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)

        self._audio: np.ndarray | None = None
        self._sr: int = 44100
        self._duration: float = 0
        self._chops: list[dict] = []  # [{start, end, classification}]
        self._peaks: np.ndarray | None = None  # downsampled waveform for display
        self._zoom: float = 1.0
        self._scroll_offset: float = 0.0

    def set_audio(self, audio: np.ndarray, sr: int):
        """Set the waveform data to display."""
        if audio.ndim > 1:
            audio = audio[0]  # mono for display
        self._audio = audio
        self._sr = sr
        self._duration = len(audio) / sr
        self._downsample()
        self.update()

    def set_chops(self, chops: list[dict]):
        """Set chop regions. Each dict: {start_time, end_time, classification}."""
        self._chops = chops
        self.update()

    def clear(self):
        self._audio = None
        self._chops = []
        self._peaks = None
        self.update()

    def _downsample(self):
        """Downsample audio to display resolution (one peak per pixel)."""
        if self._audio is None:
            self._peaks = None
            return

        width = max(self.width(), 200)
        n = len(self._audio)
        samples_per_pixel = max(1, n // width)

        # Compute min/max per pixel column
        n_cols = n // samples_per_pixel
        if n_cols == 0:
            self._peaks = None
            return

        trimmed = self._audio[:n_cols * samples_per_pixel]
        reshaped = trimmed.reshape(n_cols, samples_per_pixel)
        self._peaks = np.column_stack([
            reshaped.min(axis=1),
            reshaped.max(axis=1),
        ])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._downsample()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor("#0a0a0b"))

        if self._peaks is None:
            painter.setPen(QColor("#3f3f46"))
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "No waveform loaded")
            painter.end()
            return

        mid_y = h / 2
        n_cols = len(self._peaks)

        # Draw chop regions as colored backgrounds
        if self._chops and self._duration > 0:
            for chop in self._chops:
                x_start = int(chop["start_time"] / self._duration * n_cols)
                x_end = int(chop["end_time"] / self._duration * n_cols)
                x_start = max(0, min(x_start, n_cols))
                x_end = max(x_start, min(x_end, n_cols))

                color = QColor(self.COLORS.get(chop.get("classification", ""), "#1f1f23"))
                color.setAlpha(50)
                painter.fillRect(x_start, 0, x_end - x_start, h, color)

                # Chop boundary line
                boundary_color = QColor(self.COLORS.get(chop.get("classification", ""), "#3f3f46"))
                boundary_color.setAlpha(120)
                painter.setPen(QPen(boundary_color, 1))
                painter.drawLine(x_start, 0, x_start, h)

        # Draw waveform
        waveform_color = QColor("#F59E0B")
        painter.setPen(QPen(waveform_color, 1))

        scale = mid_y * 0.9
        for x in range(min(n_cols, w)):
            min_val = self._peaks[x, 0]
            max_val = self._peaks[x, 1]
            y_top = int(mid_y - max_val * scale)
            y_bot = int(mid_y - min_val * scale)
            painter.drawLine(x, y_top, x, y_bot)

        # Center line
        painter.setPen(QPen(QColor("#27272a"), 1))
        painter.drawLine(0, int(mid_y), w, int(mid_y))

        painter.end()

    def mousePressEvent(self, event):
        """Click on waveform to play the chop at that position."""
        if not self._chops or self._duration == 0:
            return

        x = event.position().x()
        click_time = (x / self.width()) * self._duration

        for i, chop in enumerate(self._chops):
            if chop["start_time"] <= click_time <= chop["end_time"]:
                self.chop_clicked.emit(i)
                return
