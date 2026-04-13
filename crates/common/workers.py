"""QThread base classes and worker patterns."""

from PyQt6.QtCore import QThread, pyqtSignal


class BaseWorker(QThread):
    """Abstract base worker with common signals."""

    progress = pyqtSignal(int, int, str)  # current, total, status message
    error = pyqtSignal(str)
    finished_work = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def cancel(self):
        """Request cancellation."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled
