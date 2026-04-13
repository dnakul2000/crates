"""Tab 2: Stem Separator GUI."""

import subprocess
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..common.file_utils import sanitize_filename
from ..config import DOWNLOADS_DIR, STEMS_DIR, session
from .engine import AVAILABLE_MODELS, SeparatorWorker


def _has_existing_stems(file_path: Path) -> bool:
    """Check if stems already exist for this file (per-song folder structure)."""
    name = sanitize_filename(file_path.stem)
    song_dir = STEMS_DIR / name
    if song_dir.is_dir():
        return any(f.suffix.lower() in (".wav", ".flac") for f in song_dir.iterdir())
    return False


class SeparatorTab(QWidget):
    """Audio files -> stem separation via python-audio-separator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.files: list[Path] = []
        self._selected: dict[str, bool] = {}  # filename -> checked state
        self._status: dict[str, str] = {}  # filename -> status text
        self.worker: SeparatorWorker | None = None
        self._worker_file_list: list[Path] = []  # the list sent to the worker
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Top row: model selector + controls
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Source Files:"))
        top_row.addStretch()

        top_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(AVAILABLE_MODELS)
        self.model_combo.setCurrentText("htdemucs_ft.yaml")
        top_row.addWidget(self.model_combo)

        self.add_files_btn = QPushButton("Add Files")
        self.add_files_btn.setProperty("secondary", True)
        self.add_files_btn.clicked.connect(self._add_files)
        top_row.addWidget(self.add_files_btn)

        self.add_downloads_btn = QPushButton("Add from Downloads")
        self.add_downloads_btn.setProperty("secondary", True)
        self.add_downloads_btn.clicked.connect(self._add_from_downloads)
        top_row.addWidget(self.add_downloads_btn)

        layout.addLayout(top_row)

        # File table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["", "File", "Size", "Status"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setMouseTracking(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(0, 40)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        # Select/deselect + remove + clear
        sel_row = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.setProperty("secondary", True)
        self.select_all_btn.clicked.connect(lambda: self._set_all_selected(True))
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.setProperty("secondary", True)
        self.deselect_all_btn.clicked.connect(lambda: self._set_all_selected(False))
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.setProperty("danger", True)
        self.remove_btn.clicked.connect(self._remove_selected)
        self.clear_btn = QPushButton("Clear List")
        self.clear_btn.setProperty("danger", True)
        self.clear_btn.clicked.connect(self._clear_list)
        self.count_label = QLabel("")
        sel_row.addWidget(self.select_all_btn)
        sel_row.addWidget(self.deselect_all_btn)
        sel_row.addWidget(self.remove_btn)
        sel_row.addWidget(self.clear_btn)
        sel_row.addStretch()
        sel_row.addWidget(self.count_label)
        layout.addLayout(sel_row)

        # Progress
        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Action buttons
        btn_row = QHBoxLayout()
        self.separate_btn = QPushButton("Separate Selected")
        self.separate_btn.clicked.connect(self._start_separation)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setProperty("secondary", True)
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setVisible(False)
        self.open_folder_btn = QPushButton("Open Stems Folder")
        self.open_folder_btn.setProperty("secondary", True)
        self.open_folder_btn.clicked.connect(self._open_stems)
        btn_row.addWidget(self.separate_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.open_folder_btn)
        layout.addLayout(btn_row)

        # Stem output preview
        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)

    def refresh_file_list(self):
        """Auto-populate from Downloads folder (additive)."""
        session.refresh_downloads()
        existing = {f.name for f in self.files}
        for f in session.downloaded_files:
            if f.name not in existing:
                self.files.append(f)
        self._rebuild_table()

    def _rebuild_table(self):
        """Rebuild the table while preserving checkbox and status state."""
        self.table.setRowCount(len(self.files))
        for i, f in enumerate(self.files):
            key = f.name

            # Checkbox — preserve previous state, default to checked
            cb = QCheckBox()
            cb.setChecked(self._selected.get(key, True))
            cb.stateChanged.connect(lambda state, k=key: self._on_check_changed(k, state))
            cb_widget = QWidget()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            self.table.setCellWidget(i, 0, cb_widget)

            # Filename
            name_item = QTableWidgetItem(f.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 1, name_item)

            # Size
            size_mb = f.stat().st_size / (1024 * 1024) if f.exists() else 0
            size_item = QTableWidgetItem(f"{size_mb:.1f} MB")
            size_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            size_item.setFlags(size_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 2, size_item)

            # Status — preserve previous state, or detect existing stems
            status_text = self._status.get(key, "")
            if not status_text:
                if _has_existing_stems(f):
                    status_text = "Has Stems"
                    self._selected.setdefault(key, False)  # default unchecked if already done
                    cb.setChecked(self._selected.get(key, False))
                else:
                    status_text = "Ready"
            status_item = QTableWidgetItem(status_text)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            # Color the status
            from PyQt6.QtGui import QColor
            color = "#22c55e" if status_text.startswith("Done") else self._STATUS_COLORS.get(status_text, "#a1a1aa")
            status_item.setForeground(QColor(color))
            self.table.setItem(i, 3, status_item)

        self._update_count()

    def _on_check_changed(self, key: str, state: int):
        self._selected[key] = (state == Qt.CheckState.Checked.value)
        self._update_count()

    def _set_all_selected(self, selected: bool):
        for f in self.files:
            self._selected[f.name] = selected
        self._rebuild_table()

    def _remove_selected(self):
        """Remove checked files from the list."""
        self.files = [f for f in self.files if not self._selected.get(f.name, True)]
        # Clean up state for removed files
        remaining = {f.name for f in self.files}
        self._selected = {k: v for k, v in self._selected.items() if k in remaining}
        self._status = {k: v for k, v in self._status.items() if k in remaining}
        self._rebuild_table()

    def _clear_list(self):
        self.files.clear()
        self._selected.clear()
        self._status.clear()
        self._rebuild_table()

    def _update_count(self):
        selected = sum(1 for f in self.files if self._selected.get(f.name, True))
        self.count_label.setText(f"{selected} / {len(self.files)} selected")

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "",
            "Audio Files (*.mp3 *.wav *.flac *.m4a *.ogg);;All Files (*)",
        )
        existing = {f.name for f in self.files}
        for p in paths:
            path = Path(p)
            if path.name not in existing:
                self.files.append(path)
        self._rebuild_table()

    def _add_from_downloads(self):
        session.refresh_downloads()
        existing = {f.name for f in self.files}
        for f in session.downloaded_files:
            if f.name not in existing:
                self.files.append(f)
        self._rebuild_table()

    def _get_selected_files(self) -> list[Path]:
        return [f for f in self.files if self._selected.get(f.name, True)]

    def _start_separation(self):
        selected = self._get_selected_files()
        if not selected:
            self.progress_label.setText("No files selected.")
            return

        # Store the worker's file list so we can map indices back
        self._worker_file_list = selected

        self.separate_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.progress_bar.setMaximum(len(selected))
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Separating 0 / {len(selected)}...")

        # Mark selected as "Queued"
        for f in selected:
            self._status[f.name] = "Queued"
        self._rebuild_table()

        model = self.model_combo.currentText()
        self.worker = SeparatorWorker(selected, model)
        self.worker.progress.connect(self._on_progress)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.finished_work.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    @pyqtSlot(int, int, str)
    def _on_progress(self, current: int, total: int, status: str):
        self.progress_bar.setValue(current)
        self.progress_label.setText(status)

        # Mark the current file as "Separating..."
        if current < len(self._worker_file_list):
            f = self._worker_file_list[current]
            self._status[f.name] = "Separating..."
            self._update_status_cell(f.name, "Separating...")

    @pyqtSlot(int, dict)
    def _on_file_done(self, index: int, stems: dict):
        if index < len(self._worker_file_list):
            f = self._worker_file_list[index]
            if stems:
                self._status[f.name] = f"Done ({len(stems)} stems)"
                self._update_status_cell(f.name, f"Done ({len(stems)} stems)")
            else:
                self._status[f.name] = "Failed"
                self._update_status_cell(f.name, "Failed")

        if stems:
            preview_parts = []
            for stem_type, path in stems.items():
                p = Path(path)
                size = p.stat().st_size / (1024 * 1024) if p.exists() else 0
                preview_parts.append(f"  {p.parent.name}/{p.name}  ({size:.1f} MB)")
            self.preview_label.setText("Latest output:\n" + "\n".join(preview_parts))

        self.progress_bar.setValue(self.progress_bar.value() + 1)

    _STATUS_COLORS = {
        "Ready": "#71717a",
        "Queued": "#F59E0B",
        "Separating...": "#34d399",
        "Has Stems": "#52525b",
        "Failed": "#f87171",
    }

    def _update_status_cell(self, filename: str, status: str):
        """Update the status column for a specific file by name with color."""
        from PyQt6.QtGui import QColor
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 1)
            if name_item and name_item.text() == filename:
                status_item = self.table.item(row, 3)
                if status_item:
                    status_item.setText(status)
                    # Apply color based on status
                    color = "#22c55e" if status.startswith("Done") else self._STATUS_COLORS.get(status, "#a1a1aa")
                    status_item.setForeground(QColor(color))
                break

    @pyqtSlot()
    def _on_finished(self):
        self.separate_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        done = sum(1 for s in self._status.values() if s.startswith("Done"))
        failed = sum(1 for s in self._status.values() if s == "Failed")
        self.progress_label.setText(f"Complete: {done} separated, {failed} failed.")
        session.refresh_stems()

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self.progress_label.setText(f"Error: {msg}")

    def _cancel(self):
        if self.worker:
            self.worker.cancel()
            self.progress_label.setText("Cancelling...")

    def _open_stems(self):
        import platform
        if platform.system() == "Darwin":
            subprocess.Popen(["open", str(STEMS_DIR)])

    def cleanup(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
