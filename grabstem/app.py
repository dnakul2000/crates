"""Main application window."""

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QToolBar

from .downloader.tab import DownloaderTab
from .separator.tab import SeparatorTab
from .sampler.tab import SamplerTab
from .player.tab import PlayerTab


class MainWindow(QMainWindow):
    """GrabStem main window with three tabs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GrabStem")
        self.setMinimumSize(1200, 800)

        # Unified title bar on macOS — merge title bar into content
        self.setUnifiedTitleAndToolBarOnMac(True)

        # Add an empty toolbar so macOS merges the title bar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setFixedHeight(0)
        self.addToolBar(toolbar)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)

        # Create tabs
        self.downloader_tab = DownloaderTab()
        self.separator_tab = SeparatorTab()
        self.sampler_tab = SamplerTab()
        self.player_tab = PlayerTab()

        self.tabs.addTab(self.downloader_tab, "  Song Downloader  ")
        self.tabs.addTab(self.separator_tab, "  Stem Separator  ")
        self.tabs.addTab(self.sampler_tab, "  Sample Pack Generator  ")
        self.tabs.addTab(self.player_tab, "  Pad Player  ")

        # Refresh stems when switching to separator or sampler tab
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index: int):
        if index == 1:
            self.separator_tab.refresh_file_list()
        elif index == 2:
            self.sampler_tab.refresh_stems()
        elif index == 3:
            self.player_tab.refresh_packs()

    def closeEvent(self, event):
        """Clean up workers on close."""
        self.downloader_tab.cleanup()
        self.separator_tab.cleanup()
        self.sampler_tab.cleanup()
        self.player_tab.cleanup()
        event.accept()


def run() -> int:
    """Launch the application."""
    app = QApplication(sys.argv)

    # Load stylesheet
    style_path = Path(__file__).parent / "resources" / "style.qss"
    if style_path.exists():
        app.setStyleSheet(style_path.read_text())

    window = MainWindow()
    window.show()
    return app.exec()
