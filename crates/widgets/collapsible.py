"""CollapsibleSection — animated accordion widget for grouping controls."""

from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, Qt, pyqtProperty
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ..theme import ACCENT, BG_CARD, BORDER_SUBTLE, TEXT, TEXT_MUTED, TEXT_SEC


class _SectionHeader(QWidget):
    """Clickable header bar with chevron and optional "modified" dot."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._expanded = False
        self._modified = False
        self.setFixedHeight(32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_expanded(self, expanded: bool):
        self._expanded = expanded
        self.update()

    def set_modified(self, modified: bool):
        self._modified = modified
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background
        p.fillRect(0, 0, w, h, QColor(BG_CARD))

        # Bottom border
        p.setPen(QPen(QColor(BORDER_SUBTLE), 1))
        p.drawLine(0, h - 1, w, h - 1)

        # Chevron
        p.setPen(QPen(QColor(TEXT_SEC), 1.5))
        cx, cy = 14, h // 2
        if self._expanded:
            p.drawLine(cx - 4, cy - 2, cx, cy + 2)
            p.drawLine(cx, cy + 2, cx + 4, cy - 2)
        else:
            p.drawLine(cx - 2, cy - 4, cx + 2, cy)
            p.drawLine(cx + 2, cy, cx - 2, cy + 4)

        # Title
        font = QFont("Avenir Next", 12)
        font.setWeight(QFont.Weight.DemiBold)
        p.setFont(font)
        p.setPen(QColor(TEXT if self._expanded else TEXT_SEC))
        p.drawText(28, 0, w - 40, h, Qt.AlignmentFlag.AlignVCenter, self._title)

        # Modified dot (amber)
        if self._modified:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(ACCENT))
            p.drawEllipse(w - 18, h // 2 - 3, 6, 6)

        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.parent()._toggle()


class CollapsibleSection(QWidget):
    """Animated collapsible section with a header and content area.

    Usage:
        section = CollapsibleSection("Vocals")
        section.content_layout.addWidget(some_widget)
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = _SectionHeader(title, self)
        layout.addWidget(self._header)

        self._content = QWidget()
        self.content_layout = QVBoxLayout(self._content)
        self.content_layout.setContentsMargins(8, 6, 8, 6)
        self.content_layout.setSpacing(4)
        layout.addWidget(self._content)

        self._expanded = False
        self._content.setMaximumHeight(0)
        self._content.setVisible(False)

        self._anim: QPropertyAnimation | None = None

    def set_modified(self, modified: bool):
        """Show/hide the amber dot indicating non-default values."""
        self._header.set_modified(modified)

    def _toggle(self):
        self._expanded = not self._expanded
        self._header.set_expanded(self._expanded)

        # Get the natural height of the content
        content_height = self._content.sizeHint().height()

        if self._expanded:
            self._content.setVisible(True)
            start, end = 0, content_height
        else:
            start, end = self._content.height(), 0

        if self._anim is not None:
            self._anim.stop()

        self._anim = QPropertyAnimation(self._content, b"maximumHeight")
        self._anim.setDuration(180)
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        if not self._expanded:
            self._anim.finished.connect(lambda: self._content.setVisible(False))

        self._anim.start()
