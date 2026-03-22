"""python-audio-separator wrapper with QThread worker."""

import shutil
from pathlib import Path

from PyQt6.QtCore import pyqtSignal

from ..common.file_utils import sanitize_filename
from ..common.workers import BaseWorker
from ..config import STEMS_DIR, STEM_TYPES


# Models available in python-audio-separator
# load_model() expects the filename (e.g. "htdemucs_ft.yaml")
AVAILABLE_MODELS = [
    "htdemucs_ft.yaml",
    "htdemucs.yaml",
    "htdemucs_6s.yaml",
    "mdx_extra.yaml",
    "mdx_extra_q.yaml",
]


class StemSeparator:
    """Wraps python-audio-separator for stem separation."""

    # Map audio-separator output naming patterns to our stem types
    STEM_MAP = {
        "vocals": "Vocals",
        "drums": "Drums",
        "bass": "Bass",
        "other": "Other",
        "no_vocals": "Other",
        "instrumental": "Other",
        "guitar": "Other",
        "piano": "Other",
    }

    def __init__(self, model_name: str = "htdemucs_ft", output_dir: Path | None = None):
        self.model_name = model_name
        self.output_dir = output_dir or STEMS_DIR

    def separate(self, audio_path: Path) -> dict[str, Path]:
        """Separate an audio file into stems. Returns dict mapping stem type to path."""
        from audio_separator.separator import Separator

        # Use a temp dir for raw separation output
        temp_out = self.output_dir / "_temp_separation"
        temp_out.mkdir(exist_ok=True)

        separator = Separator(
            output_dir=str(temp_out),
            output_format="WAV",
        )
        # load_model expects model_filename positional arg (e.g. "htdemucs_ft.yaml")
        model_file = self.model_name
        if not model_file.endswith((".yaml", ".ckpt", ".pth", ".onnx")):
            model_file = f"{model_file}.yaml"
        separator.load_model(model_file)
        output_files = separator.separate(str(audio_path))

        song_name = sanitize_filename(audio_path.stem)

        # Resolve every returned path — separator may return absolute or relative paths
        resolved_files = []
        for f in output_files:
            p = Path(f)
            if p.is_absolute() and p.exists():
                resolved_files.append(p)
            elif (temp_out / p.name).exists():
                resolved_files.append(temp_out / p.name)
            elif (temp_out / p).exists():
                resolved_files.append(temp_out / p)

        # If separator returned nothing usable, scan the temp dir for any WAVs it produced
        if not resolved_files:
            resolved_files = list(temp_out.glob("*.wav")) + list(temp_out.glob("*.WAV"))

        # Organize into our folder structure
        result = {}
        for file_path in resolved_files:
            if not file_path.exists():
                continue

            # Determine stem type from filename
            fname_lower = file_path.stem.lower()
            matched_type = None
            for key, stem_type in self.STEM_MAP.items():
                if key in fname_lower:
                    matched_type = stem_type
                    break
            if not matched_type:
                matched_type = "Other"

            # Copy to final destination
            dest_dir = self.output_dir / matched_type
            dest_dir.mkdir(exist_ok=True)
            dest_path = dest_dir / f"{song_name}_{matched_type.lower()}.wav"

            shutil.copy2(str(file_path), str(dest_path))
            result[matched_type] = dest_path

        # Clean up temp dir only after all copies succeeded
        shutil.rmtree(temp_out, ignore_errors=True)

        return result


class SeparatorWorker(BaseWorker):
    """Batch-processes audio files through stem separation."""

    file_done = pyqtSignal(int, dict)  # index, {stem_type: path}

    def __init__(
        self,
        files: list[Path],
        model_name: str = "htdemucs_ft",
        output_dir: Path | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.files = files
        self.model_name = model_name
        self.output_dir = output_dir

    def run(self):
        separator = StemSeparator(self.model_name, self.output_dir)
        total = len(self.files)

        for i, file_path in enumerate(self.files):
            if self.is_cancelled:
                break

            self.progress.emit(i, total, f"Separating: {file_path.name}")

            try:
                stems = separator.separate(file_path)
                self.file_done.emit(i, {k: str(v) for k, v in stems.items()})
            except Exception as e:
                self.error.emit(f"Failed to separate {file_path.name}: {e}")
                self.file_done.emit(i, {})

        self.progress.emit(total, total, "Done")
        self.finished_work.emit()
