# GrabStem

AI-powered sample pack factory. Turn any song into an MPC-ready sample pack with one click.

## What It Does

GrabStem is a desktop app (PyQt6) with four tabs:

1. **Download** — Paste a Spotify URL (track, album, or playlist) and GrabStem finds and downloads the audio via YouTube
2. **Separate** — Splits songs into 4 stems (Vocals, Drums, Bass, Other) using Demucs AI models
3. **Generate** — Chops stems into samples using one of 50 artist presets (Kanye, Madlib, J Dilla, Flying Lotus, etc.), applies effects, and maps everything to 128 MPC pads
4. **Play** — Trigger samples from a 4x4 pad grid with keyboard or MIDI controller

## Features

- **50 artist presets** — each with unique chop modes, effect chains, and pad mapping strategies
- **7 chop modes** — onset, beat grid, phrase, transient, granular, syllable, random
- **Quality gate** — auto-rejects stem bleed, silence, clipping, and spectral artifacts
- **Smart selection** — diversity-aware algorithm picks the most interesting 128 samples, not just the loudest
- **Group-aware normalization** — preserves dynamics between related samples (LUFS or peak, per-group)
- **Audition before export** — click pads to preview, then export when you're happy
- **MPC-compatible output** — 24-bit WAV files with `A01_kick_drums.wav` naming, plus MIDI (.mid) and MPC program (.xpm) files
- **Drag-and-drop** — drag samples from pads directly into your DAW
- **BPM and key tagging** — every sample tagged with source BPM, key, and beat length in the manifest
- **MIDI input** — auto-detects MPC Mini MK3 and other controllers
- **Authentic effects** — Voss-McCartney vinyl crackle, tape wobble, granular resynthesis with per-grain pitch spread, perceptual intensity curves

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+ and an internet connection for first-time model downloads.

## Run

```bash
python main.py
```

## Roadmap

- **Universal MIDI controller support** — any class-compliant MIDI controller, auto-detect, configurable pad/note mapping
- **Local file import** — skip the download step, drag in your own WAVs/MP3s/FLACs
- **DAW export formats** — Ableton Live Sets (.als), Logic Pro projects, FL Studio (.flp)
- **AU/VST plugin** — use GrabStem as a plugin inside your DAW
- **Ableton Link** — tempo-sync with other apps and devices on your network
- **Batch CLI mode** — process multiple songs from the command line without the GUI

## Disclaimer

GrabStem is provided for personal and educational use. Users are responsible for ensuring they have the rights to any audio they process. This tool does not host, distribute, or provide access to copyrighted content.
