# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Crates — AI-Powered Sample Pack Factory."""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect data files that hooks might miss
torch_data = collect_data_files("torch")
torchaudio_data = collect_data_files("torchaudio")
audio_separator_data = collect_data_files("audio_separator")
librosa_data = collect_data_files("librosa")

# Collect submodules for packages with lazy/dynamic imports
audio_separator_imports = collect_submodules("audio_separator")

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[
        # Bundle standalone yt-dlp binary (download into bin/ before building)
        ("bin/yt-dlp", "."),
    ],
    datas=[
        ("crates/resources/style.qss", "crates/resources"),
        ("crates/presets/presets.json", "crates/presets"),
    ] + torch_data + torchaudio_data + audio_separator_data + librosa_data,
    hiddenimports=[
        # Audio separator / Demucs
        *audio_separator_imports,
        # MIDI
        "mido",
        "mido.backends",
        "mido.backends.rtmidi",
        "rtmidi",
        "rtmidi._rtmidi",
        # Audio I/O
        "sounddevice",
        "soundfile",
        "_soundfile_data",
        # Effects
        "pedalboard",
        # Metering
        "pyloudnorm",
        # ML
        "torch",
        "torchaudio",
        "librosa",
        "sklearn",
        "sklearn.cluster",
        "sklearn.ensemble",
        "sklearn.preprocessing",
        "joblib",
        # Data validation
        "pydantic",
        "pydantic.deprecated",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torchvision",  # not used — saves ~150MB
        "matplotlib",
        "tkinter",
        "test",
        "unittest",
        "Cython",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Crates",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # windowed mode — no terminal
    target_arch="arm64",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Crates",
)

app = BUNDLE(
    coll,
    name="Crates.app",
    icon=None,  # TODO: add assets/icon.icns when an icon is created
    bundle_identifier="com.muxician.crates",
    info_plist={
        "CFBundleName": "Crates",
        "CFBundleDisplayName": "Crates",
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "NSHighResolutionCapable": True,
        "NSMicrophoneUsageDescription": (
            "Crates needs audio access for playback and monitoring."
        ),
        "LSMinimumSystemVersion": "13.0",
    },
)
