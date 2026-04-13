"""Waveform display widget — shows stem audio with chop boundaries overlaid."""

import numpy as np
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QLinearGradient, QFont
from PyQt6.QtWidgets import QWidget, QToolTip

from ..theme import ACCENT, ACCENT_DIM, BG_DEEP, BORDER, CLASSIFICATION_COLORS, TEXT_MUTED


class WaveformWidget(QWidget):
    """Renders a waveform with chop regions color-coded by classification."""

    chop_clicked = pyqtSignal(int)  # emits chop index when clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(120)
        self.setMouseTracking(True)

        self._audio: np.ndarray | None = None
        self._sr: int = 44100
        self._duration: float = 0
        self._chops: list[dict] = []  # [{start, end, classification}]
        self._peaks: np.ndarray | None = None  # downsampled waveform for display
        self._hover_chop: int = -1  # index of chop under cursor, -1 if none
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
        self._hover_chop = -1
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

    def _chop_at_x(self, x: float) -> int:
        """Return the index of the chop at pixel x, or -1."""
        if not self._chops or self._duration == 0 or self._peaks is None:
            return -1
        click_time = (x / self.width()) * self._duration
        for i, chop in enumerate(self._chops):
            if chop["start_time"] <= click_time <= chop["end_time"]:
                return i
        return -1

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background
        p.fillRect(0, 0, w, h, QColor(BG_DEEP))

        if self._peaks is None:
            font = QFont("Avenir Next", 12)
            p.setFont(font)
            p.setPen(QColor(TEXT_MUTED))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "No waveform loaded")
            p.end()
            return

        mid_y = h / 2
        n_cols = len(self._peaks)

        # ── Oscilloscope grid lines ──────────────────────────────────
        grid_color = QColor("#282830")
        p.setPen(QPen(grid_color, 1))
        for frac in (0.25, 0.5, 0.75):
            y = int(h * frac)
            p.drawLine(0, y, w, y)

        # ── Chop regions as colored backgrounds ──────────────────────
        if self._chops and self._duration > 0:
            for i, chop in enumerate(self._chops):
                x_start = int(chop["start_time"] / self._duration * n_cols)
                x_end = int(chop["end_time"] / self._duration * n_cols)
                x_start = max(0, min(x_start, n_cols))
                x_end = max(x_start, min(x_end, n_cols))

                cls = chop.get("classification", "")
                color = QColor(CLASSIFICATION_COLORS.get(cls, "#1f1f23"))

                # Brighter alpha for hovered chop
                if i == self._hover_chop:
                    color.setAlpha(90)
                else:
                    color.setAlpha(50)
                p.fillRect(x_start, 0, x_end - x_start, h, color)

                # Chop boundary line
                boundary_color = QColor(CLASSIFICATION_COLORS.get(cls, BORDER))
                boundary_color.setAlpha(140 if i == self._hover_chop else 100)
                p.setPen(QPen(boundary_color, 1))
                p.drawLine(x_start, 0, x_start, h)

        # ── Gradient waveform ────────────────────────────────────────
        scale = mid_y * 0.9
        accent = QColor(ACCENT)
        accent_dim = QColor(ACCENT_DIM)

        for x in range(min(n_cols, w)):
            min_val = self._peaks[x, 0]
            max_val = self._peaks[x, 1]
            y_top = int(mid_y - max_val * scale)
            y_bot = int(mid_y - min_val * scale)

            # Vertical gradient per column: bright at peaks, dim at center
            if y_bot - y_top > 1:
                grad = QLinearGradient(x, y_top, x, y_bot)
                grad.setColorAt(0.0, accent)
                grad.setColorAt(0.45, accent_dim)
                grad.setColorAt(0.55, accent_dim)
                grad.setColorAt(1.0, accent)
                p.setPen(QPen(grad, 1))
            else:
                p.setPen(QPen(accent_dim, 1))
            p.drawLine(x, y_top, x, y_bot)

        p.end()

    def mousePressEvent(self, event):
        """Click on waveform to play the chop at that position."""
        idx = self._chop_at_x(event.position().x())
        if idx >= 0:
            self.chop_clicked.emit(idx)

    def mouseMoveEvent(self, event):
        """Track which chop is under the cursor for hover highlight."""
        new_hover = self._chop_at_x(event.position().x())
        if new_hover != self._hover_chop:
            self._hover_chop = new_hover
            self.update()

            # Show classification tooltip
            if new_hover >= 0 and new_hover < len(self._chops):
                chop = self._chops[new_hover]
                cls = chop.get("classification", "unknown")
                QToolTip.showText(event.globalPosition().toPoint(), cls, self)
            else:
                QToolTip.hideText()

    def leaveEvent(self, event):
        if self._hover_chop != -1:
            self._hover_chop = -1
            self.update()
