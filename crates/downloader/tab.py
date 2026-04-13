"""Tab 1: Song Downloader GUI."""

import shutil
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import DOWNLOADS_DIR, session
from ..spotify.extractor import extract_tracks
from ..spotify.models import TrackInfo
from .engine import DownloadWorker

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".aiff", ".wma"}


class DownloaderTab(QWidget):
    """Spotify URL → track list → download via yt-dlp."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracks: list[TrackInfo] = []
        self.worker: DownloadWorker | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # URL input row
        url_row = QHBoxLayout()
        url_label = QLabel("Spotify URL:")
        url_label.setProperty("heading", True)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText(
            "Paste a Spotify playlist, album, or track URL..."
        )
        self.url_input.returnPressed.connect(self._extract_tracks)
        self.extract_btn = QPushButton("Extract Tracks")
        self.extract_btn.clicked.connect(self._extract_tracks)
        self.import_btn = QPushButton("Import Local Files")
        self.import_btn.setProperty("secondary", True)
        self.import_btn.clicked.connect(self._import_local_files)
        url_row.addWidget(url_label)
        url_row.addWidget(self.url_input, 1)
        url_row.addWidget(self.extract_btn)
        url_row.addWidget(self.import_btn)
        layout.addLayout(url_row)

        # Enable drag-and-drop
        self.setAcceptDrops(True)

        # Track table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["", "#", "Title", "Artist", "Duration", "Status"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setMouseTracking(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(0, 40)
        self.table.setColumnWidth(1, 40)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        # Select / deselect + count
        sel_row = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.setProperty("secondary", True)
        self.select_all_btn.clicked.connect(lambda: self._set_all_selected(True))
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.setProperty("secondary", True)
        self.deselect_all_btn.clicked.connect(lambda: self._set_all_selected(False))
        self.count_label = QLabel("")
        sel_row.addWidget(self.select_all_btn)
        sel_row.addWidget(self.deselect_all_btn)
        sel_row.addStretch()
        sel_row.addWidget(self.count_label)
        layout.addLayout(sel_row)

        # Output directory
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Output:"))
        self.dir_input = QLineEdit(str(DOWNLOADS_DIR))
        self.dir_input.setReadOnly(True)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setProperty("secondary", True)
        self.browse_btn.clicked.connect(self._browse_output)
        dir_row.addWidget(self.dir_input, 1)
        dir_row.addWidget(self.browse_btn)
        layout.addLayout(dir_row)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Action buttons
        btn_row = QHBoxLayout()
        self.download_btn = QPushButton("Download Selected")
        self.download_btn.clicked.connect(self._start_download)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setProperty("secondary", True)
        self.cancel_btn.clicked.connect(self._cancel_download)
        self.cancel_btn.setVisible(False)
        self.open_folder_btn = QPushButton("Open Downloads Folder")
        self.open_folder_btn.setProperty("secondary", True)
        self.open_folder_btn.clicked.connect(self._open_downloads)
        btn_row.addWidget(self.download_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.open_folder_btn)
        layout.addLayout(btn_row)

        # Status
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

    def _extract_tracks(self):
        url = self.url_input.text().strip()
        if not url:
            self.status_label.setText("Please enter a Spotify URL.")
            return

        self.extract_btn.setEnabled(False)
        self.status_label.setText("Extracting tracks...")

        try:
            self.tracks = extract_tracks(url)
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.extract_btn.setEnabled(True)
            return

        self._populate_table()
        self.status_label.setText(f"Found {len(self.tracks)} tracks.")
        self.extract_btn.setEnabled(True)

    def _populate_table(self):
        self.table.setRowCount(len(self.tracks))
        for i, track in enumerate(self.tracks):
            # Checkbox
            cb = QCheckBox()
            cb.setChecked(track.selected)
            cb.stateChanged.connect(lambda state, idx=i: self._toggle_track(idx, state))
            cb_widget = QWidget()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            self.table.setCellWidget(i, 0, cb_widget)

            # Number
            num_item = QTableWidgetItem(str(i + 1))
            num_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            num_item.setFlags(num_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 1, num_item)

            # Title
            title_item = QTableWidgetItem(track.title)
            title_item.setFlags(title_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 2, title_item)

            # Artist
            artist_item = QTableWidgetItem(track.artist)
            artist_item.setFlags(artist_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 3, artist_item)

            # Duration
            if track.duration_ms:
                mins = track.duration_ms // 60000
                secs = (track.duration_ms % 60000) // 1000
                dur_text = f"{mins}:{secs:02d}"
            else:
                dur_text = ""
            dur_item = QTableWidgetItem(dur_text)
            dur_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            dur_item.setFlags(dur_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 4, dur_item)

            # Status
            status_item = QTableWidgetItem("Ready")
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 5, status_item)

        self._update_count()

    def _toggle_track(self, index: int, state: int):
        self.tracks[index].selected = state == Qt.CheckState.Checked.value
        self._update_count()

    def _set_all_selected(self, selected: bool):
        for i, track in enumerate(self.tracks):
            track.selected = selected
            widget = self.table.cellWidget(i, 0)
            if widget:
                cb = widget.findChild(QCheckBox)
                if cb:
                    cb.setChecked(selected)
        self._update_count()

    def _update_count(self):
        selected = sum(1 for t in self.tracks if t.selected)
        self.count_label.setText(f"{selected} / {len(self.tracks)} selected")

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.dir_input.setText(path)

    def _start_download(self):
        selected = [t for t in self.tracks if t.selected]
        if not selected:
            self.status_label.setText("No tracks selected.")
            return

        from pathlib import Path

        output_dir = Path(self.dir_input.text())
        output_dir.mkdir(parents=True, exist_ok=True)

        self.download_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.progress_bar.setMaximum(len(self.tracks))
        self.progress_bar.setValue(0)

        self.worker = DownloadWorker(self.tracks, output_dir)
        self.worker.progress.connect(self._on_progress)
        self.worker.track_done.connect(self._on_track_done)
        self.worker.finished_work.connect(self._on_download_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    @pyqtSlot(int, int, str)
    def _on_progress(self, current: int, total: int, status: str):
        self.progress_bar.setValue(current)
        self.status_label.setText(status)

    @pyqtSlot(int, str, float, str)
    def _on_track_done(self, index: int, path: str, score: float, reason: str):
        from PyQt6.QtGui import QColor
        status_item = self.table.item(index, 5)
        if status_item:
            if path:
                # Show match confidence alongside Done status
                if score >= 80:
                    status_item.setText(f"Done ({score:.0f}%)")
                    status_item.setForeground(QColor("#34d399"))
                elif score >= 50:
                    status_item.setText(f"Done ({score:.0f}%)")
                    status_item.setForeground(QColor("#F59E0B"))
                else:
                    status_item.setText(f"Done ({score:.0f}% low)")
                    status_item.setForeground(QColor("#f87171"))
                    status_item.setToolTip(f"Low confidence match: {reason}")
                self.tracks[index].download_status = "done"
            elif not self.tracks[index].selected:
                status_item.setText("Skipped")
                status_item.setForeground(QColor("#71717a"))
                self.tracks[index].download_status = "skipped"
            else:
                status_item.setText("Failed")
                status_item.setForeground(QColor("#f87171"))
                status_item.setToolTip(reason)
                self.tracks[index].download_status = "failed"
        self.progress_bar.setValue(self.progress_bar.value() + 1)

    @pyqtSlot()
    def _on_download_finished(self):
        self.download_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        done = sum(1 for t in self.tracks if t.download_status == "done")
        failed = sum(1 for t in self.tracks if t.download_status == "failed")
        self.status_label.setText(f"Complete: {done} downloaded, {failed} failed.")
        session.refresh_downloads()

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self.status_label.setText(f"Error: {msg}")

    def _cancel_download(self):
        if self.worker:
            self.worker.cancel()
            self.status_label.setText("Cancelling...")

    def _open_downloads(self):
        import platform
        path = self.dir_input.text()
        if platform.system() == "Darwin":
            subprocess.Popen(["open", path])

    def _import_local_files(self):
        """Open file picker for local audio files and copy them to Downloads."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Audio Files",
            "",
            "Audio Files (*.mp3 *.wav *.flac *.m4a *.ogg *.aac *.aiff);;All Files (*)",
        )
        if files:
            self._copy_local_files([Path(f) for f in files])

    def _copy_local_files(self, file_paths: list[Path]):
        """Copy local audio files to the Downloads directory and add to track list."""
        output_dir = Path(self.dir_input.text())
        output_dir.mkdir(parents=True, exist_ok=True)

        imported = 0
        for src in file_paths:
            if src.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            dest = output_dir / src.name
            if not dest.exists():
                shutil.copy2(str(src), str(dest))
            # Add as a local track entry
            track = TrackInfo(
                artist="Local",
                title=src.stem,
                duration_ms=None,
                selected=True,
            )
            track.download_status = "done"
            self.tracks.append(track)
            imported += 1

        if imported > 0:
            self._populate_table()
            # Mark imported tracks as Done in the table
            for i in range(len(self.tracks) - imported, len(self.tracks)):
                status_item = self.table.item(i, 5)
                if status_item:
                    from PyQt6.QtGui import QColor
                    status_item.setText("Imported")
                    status_item.setForeground(QColor("#34d399"))
            self.status_label.setText(f"Imported {imported} local file(s).")
            session.refresh_downloads()
        else:
            self.status_label.setText("No supported audio files found.")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if Path(url.toLocalFile()).suffix.lower() in AUDIO_EXTENSIONS:
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event):
        file_paths = []
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local:
                file_paths.append(Path(local))
        if file_paths:
            self._copy_local_files(file_paths)

    def cleanup(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
