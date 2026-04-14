"""QThread base classes and worker patterns."""

import logging
from typing import Final

from PyQt6.QtCore import QThread, pyqtSignal

# Configure module logger
logger = logging.getLogger(__name__)

# Default progress update interval (ms)
DEFAULT_PROGRESS_INTERVAL_MS: Final[int] = 100


class BaseWorker(QThread):
    """Abstract base worker with common signals and error handling.

    Provides standardized patterns for:
    - Progress reporting via pyqtSignal
    - Cancellation support
    - Error handling with logging
    - Lifecycle management
    """

    # Signals emitted during worker execution
    progress = pyqtSignal(int, int, str)  # current, total, status message
    error = pyqtSignal(str)  # error message
    finished_work = pyqtSignal()  # completion signal

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled: bool = False
        self._current: int = 0
        self._total: int = 0
        self._status: str = ""

    def cancel(self) -> None:
        """Request cancellation of the worker.

        The worker should check is_cancelled periodically and exit gracefully.
        """
        logger.debug(f"Cancellation requested for {self.__class__.__name__}")
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled

    def update_progress(self, current: int, total: int, status: str) -> None:
        """Update progress state and emit signal.

        Args:
            current: Current progress index.
            total: Total items to process.
            status: Status message to display.
        """
        self._current = current
        self._total = total
        self._status = status
        self.progress.emit(current, total, status)
        logger.debug(f"Progress: {current}/{total} - {status}")

    def report_error(self, message: str, exception: Exception | None = None) -> None:
        """Report an error with optional exception details.

        Args:
            message: User-friendly error message.
            exception: Optional exception for logging.
        """
        if exception:
            logger.exception(f"Worker error: {message}")
        else:
            logger.error(f"Worker error: {message}")
        self.error.emit(message)

    def finish(self) -> None:
        """Emit finished signal and log completion."""
        logger.debug(f"Worker {self.__class__.__name__} finished")
        self.finished_work.emit()
