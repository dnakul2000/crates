"""Tab 3: Sample Pack Generator GUI — three-panel layout with pad grid."""

import subprocess
import traceback
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QMimeData, QThread, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..common.workers import BaseWorker
from ..config import PACKS_DIR, STEMS_DIR, STEM_TYPES, session


def _assign_temporal_groups(chops: list) -> None:
    """Assign temporal_group IDs to chops with >50% time overlap.

    Chops from different stems that overlap in time get the same group ID,
    allowing pad mappers to place related sounds on adjacent pads.
    """
    if not chops:
        return

    # Sort by start time for efficient overlap detection
    sorted_chops = sorted(chops, key=lambda c: c.start_time)
    group_id = 0
    assigned = set()

    for i, chop_a in enumerate(sorted_chops):
        if id(chop_a) in assigned:
            continue

        chop_a.temporal_group = group_id
        assigned.add(id(chop_a))
        dur_a = chop_a.end_time - chop_a.start_time

        for j in range(i + 1, len(sorted_chops)):
            chop_b = sorted_chops[j]
            if chop_b.start_time > chop_a.end_time:
                break  # no more overlaps possible
            if id(chop_b) in assigned:
                continue

            # Check overlap ratio
            overlap_start = max(chop_a.start_time, chop_b.start_time)
            overlap_end = min(chop_a.end_time, chop_b.end_time)
            overlap = max(0, overlap_end - overlap_start)
            dur_b = chop_b.end_time - chop_b.start_time
            min_dur = min(dur_a, dur_b)

            if min_dur > 0 and overlap / min_dur > 0.5:
                chop_b.temporal_group = group_id
                assigned.add(id(chop_b))

        group_id += 1


class PadButton(QPushButton):
    """A single pad in the 4x4 grid — clickable for audition."""

    pad_clicked = pyqtSignal(int)  # emits pad_number when clicked

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
        "": "#151517",
    }

    def __init__(self, pad_number: int, parent=None):
        super().__init__(str(pad_number), parent)
        self.pad_number = pad_number
        self.classification = ""
        self._wav_path: str | None = None
        self.setFixedSize(72, 72)
        self._update_style()
        self.clicked.connect(lambda: self.pad_clicked.emit(self.pad_number))

    def set_wav_path(self, path: str):
        """Set the path to the WAV file for drag-and-drop."""
        self._wav_path = path

    def mouseMoveEvent(self, event):
        """Enable drag-and-drop of WAV files to DAW."""
        if self._wav_path and event.buttons() == Qt.MouseButton.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(self._wav_path)])
            drag.setMimeData(mime)
            drag.exec(Qt.DropAction.CopyAction)
        else:
            super().mouseMoveEvent(event)

    def set_assignment(self, classification: str, label: str):
        self.classification = classification
        self.setText(label[:6] if label else str(self.pad_number))
        self.setToolTip(f"Pad {self.pad_number}: {classification}\n{label}\nClick to audition")
        self._update_style()

    def clear_assignment(self):
        self.classification = ""
        self.setText(str(self.pad_number))
        self.setToolTip("")
        self._update_style()

    def flash(self):
        """Brief visual flash on trigger."""
        original = self.styleSheet()
        self.setStyleSheet(original.replace("border-color:", "border-color: #F59E0B; /*"))
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(120, lambda: self.setStyleSheet(original))

    def _update_style(self):
        color = self.COLORS.get(self.classification, "#151517")
        border = "#3f3f46" if self.classification else "#27272a"
        self.setStyleSheet(
            f"QPushButton {{ background-color: {color}; color: #fafafa; "
            f"border: 1px solid {border}; border-radius: 10px; "
            f"font-family: 'SF Mono', 'Menlo', monospace; font-weight: 600; font-size: 10px; }}"
            f"QPushButton:hover {{ border-color: #F59E0B; background-color: {color}; }}"
        )


class PadGrid(QWidget):
    """4x4 MPC pad grid with bank selector and audition playback."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Bank selector
        bank_row = QHBoxLayout()
        bank_row.addWidget(QLabel("Bank:"))
        self.bank_combo = QComboBox()
        self.bank_combo.addItems(list("ABCDEFGH"))
        self.bank_combo.currentTextChanged.connect(self._on_bank_changed)
        bank_row.addWidget(self.bank_combo)
        bank_row.addStretch()
        layout.addLayout(bank_row)

        # 4x4 grid (bottom-left origin like MPC)
        grid = QGridLayout()
        grid.setSpacing(6)
        self.pads: list[PadButton] = []
        for pad_num in range(1, 17):
            btn = PadButton(pad_num)
            btn.pad_clicked.connect(self._on_pad_clicked)
            self.pads.append(btn)
            row = 3 - (pad_num - 1) // 4
            col = (pad_num - 1) % 4
            grid.addWidget(btn, row, col)
        layout.addLayout(grid)

        self._assignment = None
        self._current_bank = "A"
        self._playback_engine = None

    def set_assignment(self, assignment):
        """Set the pad assignment and refresh display."""
        self._assignment = assignment
        self._refresh()

    def _on_bank_changed(self, bank: str):
        self._current_bank = bank
        self._refresh()

    def _on_pad_clicked(self, pad_number: int):
        """Audition a pad by playing its audio directly."""
        if self._assignment is None:
            return

        pad_index = pad_number - 1
        bank_data = self._assignment.banks.get(self._current_bank, [])
        if pad_index >= len(bank_data) or bank_data[pad_index] is None:
            return

        slot = bank_data[pad_index]
        self.pads[pad_index].flash()

        # Play the chop audio directly using sounddevice
        try:
            import sounddevice as sd
            audio = slot.chop.audio.copy()
            if audio.ndim > 1:
                audio = audio[0]  # mono
            # Normalize for playback
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio * (0.8 / peak)
            sd.play(audio.astype(np.float32), slot.chop.sr)
        except Exception:
            pass

    def _refresh(self):
        for pad in self.pads:
            pad.clear_assignment()

        if self._assignment is None:
            return

        bank_data = self._assignment.banks.get(self._current_bank, [])
        for i, slot in enumerate(bank_data):
            if slot is not None and i < len(self.pads):
                self.pads[i].set_assignment(
                    slot.chop.classification,
                    slot.chop.classification,
                )


class GenerateWorker(BaseWorker):
    """Background worker for generating sample packs."""

    pack_done = pyqtSignal(str)  # pack directory path
    assignment_ready = pyqtSignal(object)  # PadAssignment for the pad grid
    awaiting_export = pyqtSignal(object, str, str, int, list, int, str)  # assignment + export params
    generation_stats = pyqtSignal(str)  # detailed stats string

    def __init__(
        self,
        stem_files: dict[str, list[Path]],
        preset_name: str,
        intensity: int,
        pack_name: str,
        bit_depth: int,
        effect_overrides: dict[str, dict[str, float]] | None = None,
        max_samples: int = 128,
        parent=None,
    ):
        super().__init__(parent)
        self.stem_files = stem_files
        self.preset_name = preset_name
        self.intensity = intensity
        self.pack_name = pack_name
        self.bit_depth = bit_depth
        self.effect_overrides = effect_overrides or {}
        self.max_samples = max_samples

    def run(self):
        try:
            self._do_generate()
        except Exception as e:
            self.error.emit(f"Generation failed: {e}\n{traceback.format_exc()}")
        finally:
            # ALWAYS emit finished so the button re-enables
            self.finished_work.emit()

    def _do_generate(self):
        from ..common.audio_utils import load_audio
        from ..sampler.analyzer import analyze
        from ..sampler.chopper import chop
        from ..sampler.effects import EffectsPipeline
        from ..sampler.exporter import export_pack
        from ..sampler.pad_mapper import assign_pads
        from ..sampler.preset_engine import PresetEngine

        engine = PresetEngine()
        preset = engine.get(self.preset_name)
        if not preset:
            self.error.emit(f"Preset '{self.preset_name}' not found")
            return

        chop_config = engine.get_chop_config(preset)
        effects_config = engine.get_effects_config(preset)
        intensity_f = self.intensity / 100.0

        # Merge per-stem effect overrides from the UI sliders
        self._apply_overrides(effects_config)

        effects = EffectsPipeline(effects_config, intensity_f)
        all_chops = []
        source_songs = []

        total_files = sum(len(files) for files in self.stem_files.values())
        processed = 0

        for stem_type, files in self.stem_files.items():
            for file_path in files:
                if self.is_cancelled:
                    return

                self.progress.emit(
                    processed, total_files,
                    f"Analyzing {stem_type}: {file_path.name}",
                )

                try:
                    analysis = analyze(file_path)
                    audio, sr = load_audio(file_path)
                    if audio.ndim > 1:
                        audio = audio[0]

                    chops = chop(audio, sr, analysis, stem_type, chop_config, intensity_f)

                    self.progress.emit(
                        processed, total_files,
                        f"Effects on {stem_type}: {file_path.name} ({len(chops)} chops)",
                    )
                    for c in chops:
                        c.audio = effects.process(c.audio, c.sr, stem_type)
                        c.source_stem = stem_type

                    all_chops.extend(chops)
                    source_songs.append(file_path.stem)

                except Exception as e:
                    self.error.emit(f"Error processing {file_path.name}: {e}")

                processed += 1

        if not all_chops:
            self.error.emit("No chops generated. Check that stems contain audio.")
            return

        # Assign temporal groups: chops with >50% time overlap share a group
        _assign_temporal_groups(all_chops)

        # Emit generation stats
        from collections import Counter
        cls_counts = Counter(c.classification for c in all_chops)
        stats_lines = [f"Generated {len(all_chops)} chops from {total_files} stems:"]
        for cls, count in sorted(cls_counts.items(), key=lambda x: -x[1]):
            stats_lines.append(f"  {cls}: {count}")
        self.generation_stats.emit("\n".join(stats_lines))

        # Pad mapping
        self.progress.emit(processed, total_files, f"Mapping {len(all_chops)} chops to pads...")
        # Limit chops to max_samples using diversity-aware selection
        if len(all_chops) > self.max_samples:
            from ..sampler.pad_mapper import select_diverse_chops
            all_chops = select_diverse_chops(all_chops, self.max_samples)

        assignment = assign_pads(
            all_chops,
            strategy=preset.pad_mapping.strategy,
            sort_by=preset.pad_mapping.sort_by,
        )

        # Send assignment to GUI for pad grid display (audition)
        self.assignment_ready.emit(assignment)

        # Store export params for deferred export
        normalization = chop_config.get("normalization", "peak_group")
        self.awaiting_export.emit(
            assignment, self.pack_name, self.preset_name,
            self.intensity, source_songs, self.bit_depth, normalization
        )

        self.progress.emit(total_files, total_files, "Ready — click pads to audition, then Export")

    def _apply_overrides(self, effects_config: dict[str, list[dict]]):
        """Merge UI slider overrides into the preset effects config.

        Override keys map to effect types:
          reverb  -> adds/modifies reverb effect
          compress -> adds/modifies compression effect
          saturate -> adds/modifies saturation effect

        Slider value 50 = use preset default (no change).
        0 = fully remove effect. 100 = double the preset value.
        """
        override_map = {
            "reverb": {"type": "reverb", "params": {"room_size": 0.5, "wet_level": 0.3}},
            "compress": {"type": "compression", "params": {"threshold_db": -15.0, "ratio": 4.0}},
            "saturate": {"type": "saturation", "params": {"drive_db": 8.0}},
        }

        for stem_type, overrides in self.effect_overrides.items():
            stem_key = stem_type.lower()
            if stem_key not in effects_config:
                effects_config[stem_key] = []

            for override_name, slider_value in overrides.items():
                if override_name not in override_map:
                    continue

                scale = slider_value / 50.0  # 0=off, 1=default, 2=max

                if scale < 0.05:
                    # Remove this effect type from the chain
                    effects_config[stem_key] = [
                        e for e in effects_config[stem_key]
                        if e.get("type") != override_map[override_name]["type"]
                    ]
                    continue

                # Find existing effect of this type
                template = override_map[override_name]
                found = False
                for effect in effects_config[stem_key]:
                    if effect.get("type") == template["type"]:
                        # Scale existing params
                        for param_key, param_val in effect.get("params", {}).items():
                            effect["params"][param_key] = param_val * scale
                        found = True
                        break

                if not found and scale > 0.5:
                    # Add effect with scaled params
                    new_effect = {
                        "type": template["type"],
                        "params": {k: v * scale for k, v in template["params"].items()},
                        "enabled": True,
                    }
                    effects_config[stem_key].append(new_effect)


class SamplerTab(QWidget):
    """Sample Pack Generator — the core feature."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: GenerateWorker | None = None
        self._last_pack_dir: str = str(PACKS_DIR)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Three-panel splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # === LEFT PANEL: Source Selection ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)

        left_label = QLabel("Available Stems:")
        left_label.setProperty("heading", True)
        left_layout.addWidget(left_label)

        self.stem_tree = QTreeWidget()
        self.stem_tree.setHeaderLabels(["Stem"])
        self.stem_tree.setRootIsDecorated(True)
        left_layout.addWidget(self.stem_tree, 1)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setProperty("secondary", True)
        refresh_btn.clicked.connect(self.refresh_stems)
        left_layout.addWidget(refresh_btn)

        splitter.addWidget(left_panel)

        # === CENTER PANEL: Preset & Controls ===
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(10, 10, 10, 10)

        preset_label = QLabel("Artist Preset:")
        preset_label.setProperty("heading", True)
        center_layout.addWidget(preset_label)

        self.preset_combo = QComboBox()
        self.preset_combo.setMaxVisibleItems(20)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        center_layout.addWidget(self.preset_combo)

        self.preset_description = QLabel("")
        self.preset_description.setWordWrap(True)
        self.preset_description.setStyleSheet("color: #71717a; font-style: italic; padding: 4px;")
        center_layout.addWidget(self.preset_description)

        # Intensity slider
        intensity_row = QHBoxLayout()
        intensity_row.addWidget(QLabel("Intensity:"))
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue(75)
        self.intensity_slider.valueChanged.connect(self._on_intensity_changed)
        intensity_row.addWidget(self.intensity_slider, 1)
        self.intensity_label = QLabel("75%")
        self.intensity_label.setFixedWidth(40)
        intensity_row.addWidget(self.intensity_label)
        center_layout.addLayout(intensity_row)

        # Per-stem effect overrides
        effects_group = QGroupBox("Per-Stem Effect Overrides (50 = preset default)")
        effects_layout = QVBoxLayout(effects_group)

        self.stem_effect_sliders: dict[str, dict[str, QSlider]] = {}
        self._effect_value_labels: dict[str, dict[str, QLabel]] = {}
        for stem in ["Vocals", "Drums", "Bass", "Other"]:
            stem_frame = QFrame()
            stem_row = QHBoxLayout(stem_frame)
            stem_row.setContentsMargins(0, 4, 0, 4)
            stem_label = QLabel(f"{stem}:")
            stem_label.setFixedWidth(55)
            stem_label.setStyleSheet("color: #e4e4e7; font-weight: 600;")
            stem_row.addWidget(stem_label)

            sliders = {}
            value_labels = {}
            for effect_name in ["Reverb", "Compress", "Saturate"]:
                effect_col = QVBoxLayout()
                effect_col.setSpacing(1)
                lbl = QLabel(effect_name)
                lbl.setStyleSheet("font-size: 10px; color: #52525b;")
                effect_col.addWidget(lbl)
                slider_row = QHBoxLayout()
                slider_row.setSpacing(4)
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(0, 100)
                slider.setValue(50)
                slider.setFixedWidth(70)
                slider.setToolTip(f"{stem} {effect_name}: 0=off, 50=preset, 100=2x")
                val_lbl = QLabel("50")
                val_lbl.setFixedWidth(22)
                val_lbl.setStyleSheet(
                    "font-family: 'SF Mono', 'Menlo', monospace; font-size: 10px; color: #52525b;"
                )
                slider.valueChanged.connect(
                    lambda v, l=val_lbl: l.setText(str(v))
                )
                slider_row.addWidget(slider)
                slider_row.addWidget(val_lbl)
                effect_col.addLayout(slider_row)
                stem_row.addLayout(effect_col)
                sliders[effect_name.lower()] = slider
                value_labels[effect_name.lower()] = val_lbl

            self.stem_effect_sliders[stem.lower()] = sliders
            self._effect_value_labels[stem.lower()] = value_labels
            effects_layout.addWidget(stem_frame)

        scroll = QScrollArea()
        scroll.setWidget(effects_group)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(250)
        center_layout.addWidget(scroll)

        center_layout.addStretch()
        splitter.addWidget(center_panel)

        # === RIGHT PANEL: Output & Export ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        right_label = QLabel("Output:")
        right_label.setProperty("heading", True)
        right_layout.addWidget(right_label)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Pack Name:"))
        self.pack_name_input = QLineEdit()
        self.pack_name_input.setPlaceholderText("My Sample Pack")
        name_row.addWidget(self.pack_name_input, 1)
        right_layout.addLayout(name_row)

        depth_row = QHBoxLayout()
        depth_row.addWidget(QLabel("Bit Depth:"))
        self.bit_24 = QRadioButton("24-bit")
        self.bit_24.setChecked(True)
        self.bit_16 = QRadioButton("16-bit")
        depth_row.addWidget(self.bit_24)
        depth_row.addWidget(self.bit_16)
        depth_row.addStretch()
        right_layout.addLayout(depth_row)

        right_layout.addWidget(QLabel("Sample Rate: 44100 Hz"))

        # Max samples slider
        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max Samples:"))
        self.max_samples_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_samples_slider.setRange(8, 128)
        self.max_samples_slider.setValue(128)
        self.max_samples_slider.setSingleStep(8)
        self.max_samples_slider.setPageStep(16)
        self.max_samples_slider.valueChanged.connect(self._on_max_samples_changed)
        max_row.addWidget(self.max_samples_slider, 1)
        self.max_samples_label = QLabel("128")
        self.max_samples_label.setFixedWidth(30)
        self.max_samples_label.setStyleSheet(
            "font-family: 'SF Mono', 'Menlo', monospace; font-size: 12px; color: #a1a1aa;"
        )
        max_row.addWidget(self.max_samples_label)
        right_layout.addLayout(max_row)

        # Pad grid
        right_layout.addWidget(QLabel(""))
        pad_label = QLabel("Pad Mapping:")
        pad_label.setProperty("heading", True)
        right_layout.addWidget(pad_label)
        self.pad_grid = PadGrid()
        right_layout.addWidget(self.pad_grid)

        self.sample_count_label = QLabel("Samples: 0 / 128")
        self.sample_count_label.setStyleSheet(
            "font-family: 'SF Mono', 'Menlo', monospace; font-size: 12px; color: #71717a;"
        )
        right_layout.addWidget(self.sample_count_label)

        right_layout.addStretch()

        # Generate button
        self.generate_btn = QPushButton("Generate Pack")
        self.generate_btn.setFixedHeight(50)
        self.generate_btn.clicked.connect(self._start_generation)
        right_layout.addWidget(self.generate_btn)

        # Export button (shown after audition)
        self.export_btn = QPushButton("Export Pack")
        self.export_btn.setFixedHeight(40)
        self.export_btn.setStyleSheet(
            "QPushButton { background-color: #059669; color: white; font-weight: 700; "
            "border-radius: 8px; font-size: 13px; }"
            "QPushButton:hover { background-color: #047857; }"
        )
        self.export_btn.clicked.connect(self._do_export)
        self.export_btn.setVisible(False)
        right_layout.addWidget(self.export_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        right_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        right_layout.addWidget(self.status_label)

        # Stats label for generation details
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet(
            "font-family: 'SF Mono', 'Menlo', monospace; font-size: 10px; "
            "color: #71717a; padding: 4px;"
        )
        right_layout.addWidget(self.stats_label)

        self.open_output_btn = QPushButton("Open Output Folder")
        self.open_output_btn.setProperty("secondary", True)
        self.open_output_btn.clicked.connect(self._open_output)
        right_layout.addWidget(self.open_output_btn)

        splitter.addWidget(right_panel)
        splitter.setSizes([250, 400, 350])
        main_layout.addWidget(splitter, 1)

        # Load presets
        self._load_presets()

    def _load_presets(self):
        """Populate preset combo from registry, grouped by category."""
        try:
            from ..presets.registry import list_presets
            presets = list_presets()

            categories: dict[str, list[tuple]] = {}
            for name, genre, category, description in presets:
                categories.setdefault(category, []).append((name, genre, description))

            for category, items in categories.items():
                self.preset_combo.addItem(f"── {category} ──")
                idx = self.preset_combo.count() - 1
                model = self.preset_combo.model()
                item = model.item(idx)
                item.setEnabled(False)

                for name, genre, description in items:
                    self.preset_combo.addItem(f"  {name}", (name, description))

            if self.preset_combo.count() > 1:
                self.preset_combo.setCurrentIndex(1)

        except Exception as e:
            self.preset_combo.addItem("(No presets loaded)")
            self.status_label.setText(f"Failed to load presets: {e}")

    def _on_preset_changed(self, index: int):
        data = self.preset_combo.itemData(index)
        if data:
            name, desc = data
            self.preset_description.setText(f"{name}: {desc}")

    def _on_intensity_changed(self, value: int):
        self.intensity_label.setText(f"{value}%")

    def _on_max_samples_changed(self, value: int):
        # Snap to multiples of 8 (half a bank)
        snapped = round(value / 8) * 8
        if snapped < 8:
            snapped = 8
        self.max_samples_slider.blockSignals(True)
        self.max_samples_slider.setValue(snapped)
        self.max_samples_slider.blockSignals(False)
        self.max_samples_label.setText(str(snapped))

    def refresh_stems(self):
        """Rescan the Stems directory and populate the tree."""
        self.stem_tree.clear()
        session.refresh_stems()

        song_stems: dict[str, dict[str, Path]] = {}
        for stem_type, files in session.separated_stems.items():
            for f in files:
                song_name = f.stem
                for suffix in ("_vocals", "_drums", "_bass", "_other"):
                    song_name = song_name.replace(suffix, "")
                song_stems.setdefault(song_name, {})[stem_type] = f

        for song_name, stems in sorted(song_stems.items()):
            song_item = QTreeWidgetItem([song_name])
            song_item.setCheckState(0, Qt.CheckState.Checked)
            for stem_type, path in sorted(stems.items()):
                stem_item = QTreeWidgetItem([stem_type])
                stem_item.setCheckState(0, Qt.CheckState.Checked)
                stem_item.setData(0, Qt.ItemDataRole.UserRole, str(path))
                song_item.addChild(stem_item)
            self.stem_tree.addTopLevelItem(song_item)

        self.stem_tree.expandAll()
        self.status_label.setText(f"Found stems for {len(song_stems)} songs.")

    def _get_selected_stems(self) -> dict[str, list[Path]]:
        """Get selected stems organized by type."""
        result: dict[str, list[Path]] = {}
        root = self.stem_tree.invisibleRootItem()
        for i in range(root.childCount()):
            song_item = root.child(i)
            for j in range(song_item.childCount()):
                stem_item = song_item.child(j)
                if stem_item.checkState(0) == Qt.CheckState.Checked:
                    path_str = stem_item.data(0, Qt.ItemDataRole.UserRole)
                    if path_str:
                        stem_type = stem_item.text(0)
                        result.setdefault(stem_type, []).append(Path(path_str))
        return result

    def _get_effect_overrides(self) -> dict[str, dict[str, float]]:
        """Read per-stem effect override sliders. Returns {stem: {effect: value}}."""
        overrides = {}
        for stem_type, sliders in self.stem_effect_sliders.items():
            stem_overrides = {}
            for effect_name, slider in sliders.items():
                val = slider.value()
                if val != 50:  # Only include non-default values
                    stem_overrides[effect_name] = float(val)
            if stem_overrides:
                overrides[stem_type] = stem_overrides
        return overrides

    def _start_generation(self):
        # Auto-refresh stems before generating
        self.refresh_stems()

        stems = self._get_selected_stems()
        if not stems:
            self.status_label.setText("No stems found. Separate some songs first (Tab 2).")
            return

        total_files = sum(len(v) for v in stems.values())
        stem_summary = ", ".join(f"{k}: {len(v)}" for k, v in stems.items())

        preset_data = self.preset_combo.currentData()
        if not preset_data:
            self.status_label.setText("No preset selected.")
            return
        preset_text = preset_data[0]

        pack_name = self.pack_name_input.text().strip() or "Untitled Pack"
        intensity = self.intensity_slider.value()
        bit_depth = 24 if self.bit_24.isChecked() else 16
        effect_overrides = self._get_effect_overrides()

        self.generate_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(total_files)
        self.status_label.setText(
            f"Generating '{pack_name}' with {preset_text} @ {intensity}%\n"
            f"Processing {total_files} stems ({stem_summary})..."
        )

        max_samples = self.max_samples_slider.value()

        self.worker = GenerateWorker(
            stem_files=stems,
            preset_name=preset_text,
            intensity=intensity,
            pack_name=pack_name,
            bit_depth=bit_depth,
            effect_overrides=effect_overrides,
            max_samples=max_samples,
        )
        self.export_btn.setVisible(False)
        self.stats_label.setText("")
        self._pending_export = None

        self.worker.progress.connect(self._on_progress)
        self.worker.pack_done.connect(self._on_pack_done)
        self.worker.assignment_ready.connect(self._on_assignment_ready)
        self.worker.awaiting_export.connect(self._on_awaiting_export)
        self.worker.generation_stats.connect(self._on_stats)
        self.worker.error.connect(self._on_error)
        self.worker.finished_work.connect(self._on_finished)
        self.worker.start()

    @pyqtSlot(int, int, str)
    def _on_progress(self, current: int, total: int, status: str):
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
        self.status_label.setText(status)

    @pyqtSlot(object)
    def _on_assignment_ready(self, assignment):
        """Update the pad grid with the actual pad assignment."""
        self.pad_grid.set_assignment(assignment)
        self.sample_count_label.setText(
            f"Samples: {assignment.total_assigned} / 128"
        )

    @pyqtSlot(str)
    def _on_pack_done(self, pack_dir: str):
        self.status_label.setText(f"Pack exported to:\n{pack_dir}")
        self._last_pack_dir = pack_dir
        self.export_btn.setVisible(False)

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self.status_label.setText(f"Error: {msg}")

    @pyqtSlot(str)
    def _on_stats(self, stats: str):
        self.stats_label.setText(stats)

    @pyqtSlot(object, str, str, int, list, int, str)
    def _on_awaiting_export(self, assignment, pack_name, preset_name, intensity, source_songs, bit_depth, normalization):
        """Store export params and show export button for audition workflow."""
        self._pending_export = {
            "assignment": assignment,
            "pack_name": pack_name,
            "preset_name": preset_name,
            "intensity": intensity,
            "source_songs": source_songs,
            "bit_depth": bit_depth,
            "normalization": normalization,
        }
        self.export_btn.setVisible(True)
        self.status_label.setText(
            "Click pads to audition samples. Press Export when ready."
        )

    def _do_export(self):
        """Export the pack after audition."""
        if not self._pending_export:
            return

        from ..sampler.exporter import export_pack
        self.export_btn.setEnabled(False)
        self.status_label.setText("Exporting WAV files...")

        try:
            pack_dir = export_pack(
                assignment=self._pending_export["assignment"],
                pack_name=self._pending_export["pack_name"],
                preset_name=self._pending_export["preset_name"],
                intensity=self._pending_export["intensity"],
                source_songs=self._pending_export["source_songs"],
                bit_depth=self._pending_export["bit_depth"],
                normalization=self._pending_export["normalization"],
            )
            self._on_pack_done(str(pack_dir))
        except Exception as e:
            self._on_error(f"Export failed: {e}")
        finally:
            self.export_btn.setEnabled(True)
            self._pending_export = None

    @pyqtSlot()
    def _on_finished(self):
        self.generate_btn.setEnabled(True)
        if self.progress_bar.maximum() > 0:
            self.progress_bar.setValue(self.progress_bar.maximum())

    def _open_output(self):
        import platform
        if platform.system() == "Darwin":
            subprocess.Popen(["open", self._last_pack_dir])

    def cleanup(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
