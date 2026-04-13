"""StudioPad — custom-painted pad widget with tactile visuals and glow animations.

Replaces both PadButton (sampler) and PlayPadButton (player) with a single
QPainter-based widget that looks and feels like real MPC hardware pads.
"""

from PyQt6.QtCore import (
    QPropertyAnimation,
    QSequentialAnimationGroup,
    QEasingCurve,
    QMimeData,
    QUrl,
    Qt,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QDrag,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QWidget

from ..theme import (
    ACCENT,
    ACCENT_HOVER,
    BG_CARD,
    BG_DEEP,
    BORDER,
    BORDER_HI,
    CLASSIFICATION_COLORS,
    TEXT,
    TEXT_MUTED,
    darken,
    lighten,
    with_alpha,
)


class StudioPad(QWidget):
    """A tactile, hardware-feel pad with gradient rendering and glow animation."""

    pad_clicked = pyqtSignal(int)

    def __init__(self, pad_number: int, size: int = 72, parent=None):
        super().__init__(parent)
        self.pad_number = pad_number
        self.classification = ""
        self._label = str(pad_number)
        self._wav_path: str | None = None

        # Visual state
        self._base_color = BG_CARD
        self._glow = 0.0
        self._pressed = False
        self._hovered = False

        # Animation
        self._anim_group: QSequentialAnimationGroup | None = None

        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

    # ── Qt properties for animation ──────────────────────────────────

    def _get_glow(self) -> float:
        return self._glow

    def _set_glow(self, value: float):
        self._glow = value
        self.update()

    glow_intensity = pyqtProperty(float, _get_glow, _set_glow)

    # ── Public API ───────────────────────────────────────────────────

    def set_assignment(self, classification: str, label: str):
        """Assign a sample to this pad."""
        self.classification = classification
        self._base_color = CLASSIFICATION_COLORS.get(classification, BG_CARD)
        self._label = label[:8] if label else str(self.pad_number)
        self.setToolTip(f"Pad {self.pad_number}: {classification}\n{label}\nClick to audition")
        self.update()

    def clear_assignment(self):
        """Reset pad to empty state."""
        self.classification = ""
        self._base_color = BG_CARD
        self._label = str(self.pad_number)
        self.setToolTip("")
        self.update()

    # Alias used by player tab
    def set_sample(self, classification: str, label: str, info: str):
        """Assign a sample (player tab compatibility)."""
        self.set_assignment(classification, label)
        self.setToolTip(info)

    def clear_sample(self):
        """Clear sample (player tab compatibility)."""
        self.clear_assignment()

    def set_wav_path(self, path: str):
        """Set the WAV file path for drag-and-drop to DAW."""
        self._wav_path = path

    def flash(self):
        """Trigger a smooth percussive flash: fast attack, medium decay."""
        # Stop any running animation
        if self._anim_group is not None:
            self._anim_group.stop()

        attack = QPropertyAnimation(self, b"glow_intensity")
        attack.setDuration(50)
        attack.setStartValue(0.0)
        attack.setEndValue(1.0)
        attack.setEasingCurve(QEasingCurve.Type.OutQuad)

        decay = QPropertyAnimation(self, b"glow_intensity")
        decay.setDuration(250)
        decay.setStartValue(1.0)
        decay.setEndValue(0.0)
        decay.setEasingCurve(QEasingCurve.Type.InQuad)

        self._anim_group = QSequentialAnimationGroup(self)
        self._anim_group.addAnimation(attack)
        self._anim_group.addAnimation(decay)
        self._anim_group.start()

    # Player tab compatibility
    def flash_play(self):
        """Alias for flash(), used by the player tab."""
        self.flash()

    # ── Events ───────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = False
            self.update()
            if self.rect().contains(event.position().toPoint()):
                self.pad_clicked.emit(self.pad_number)

    def mouseMoveEvent(self, event):
        # Drag-and-drop WAV to DAW
        if self._wav_path and event.buttons() == Qt.MouseButton.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(self._wav_path)])
            drag.setMimeData(mime)
            drag.exec(Qt.DropAction.CopyAction)
        else:
            super().mouseMoveEvent(event)

    def enterEvent(self, event):
        self._hovered = True
        self.update()

    def leaveEvent(self, event):
        self._hovered = False
        self.update()

    # ── Painting ─────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        radius = 10.0
        has_sample = bool(self.classification)
        glow = self._glow

        # ── 1. Outer glow (when flashing or hovered with sample) ────
        if glow > 0.05 and has_sample:
            glow_color = with_alpha(self._base_color, int(60 * glow))
            grad = QRadialGradient(w / 2, h / 2, w * 0.7)
            grad.setColorAt(0.0, glow_color)
            grad.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setBrush(grad)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRectF(-4, -4, w + 8, h + 8), radius + 4, radius + 4)

        # ── 2. Pad body path ────────────────────────────────────────
        pad_path = QPainterPath()
        y_offset = 1.0 if self._pressed else 0.0
        body_rect = QRectF(0.5, 0.5 + y_offset, w - 1.0, h - 1.0)
        pad_path.addRoundedRect(body_rect, radius, radius)

        # ── 3. Background gradient ──────────────────────────────────
        if has_sample:
            base = self._base_color
            # Glow brightens the base color
            if glow > 0.05:
                bright = lighten(base, int(70 * glow))
            else:
                bright = base
            top_color = QColor(lighten(bright, 18))
            bot_color = QColor(darken(bright, 12))
        else:
            top_color = QColor(BG_CARD)
            bot_color = QColor(darken(BG_CARD, 8))

        if self._pressed and has_sample:
            top_color, bot_color = bot_color, top_color  # Invert = pushed-in

        grad = QLinearGradient(0, y_offset, 0, h + y_offset)
        grad.setColorAt(0.0, top_color)
        grad.setColorAt(1.0, bot_color)

        p.setBrush(grad)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(pad_path)

        # ── 4. Top inner highlight (simulates overhead light) ───────
        if has_sample:
            highlight = QColor(255, 255, 255, 20)
        else:
            highlight = QColor(255, 255, 255, 8)
        p.setPen(QPen(highlight, 1.0))
        p.drawLine(int(radius), int(1 + y_offset), int(w - radius), int(1 + y_offset))

        # ── 5. Border ───────────────────────────────────────────────
        if glow > 0.3:
            border_color = with_alpha(ACCENT, int(180 + 75 * glow))
            border_width = 1.5
        elif self._hovered and has_sample:
            border_color = with_alpha(ACCENT, 140)
            border_width = 1.5
        elif has_sample:
            border_color = with_alpha(self._base_color, 160)
            border_width = 1.0
        else:
            border_color = QColor(BORDER)
            border_width = 1.0

        p.setPen(QPen(border_color, border_width))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(body_rect, radius, radius)

        # ── 6. Text label ───────────────────────────────────────────
        font = QFont("Menlo", 10 if w < 90 else 11)
        font.setWeight(QFont.Weight.DemiBold)
        p.setFont(font)

        if has_sample:
            text_color = QColor(TEXT)
            text_color.setAlpha(220 + int(35 * glow))
        else:
            text_color = QColor(TEXT_MUTED)

        p.setPen(text_color)

        # Multi-line for classifications with underscores
        display = self._label.replace("_", "\n") if has_sample and "_" in self._label else self._label
        text_rect = self.rect().adjusted(4, int(4 + y_offset), -4, -4)
        p.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, display)

        p.end()
