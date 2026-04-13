#!/usr/bin/env bash
# Build Crates into a standalone macOS .app bundle.
#
# Prerequisites:
#   1. The .venv must exist with runtime deps installed (run ./launch.sh first).
#   2. Download the yt-dlp macOS binary into bin/yt-dlp:
#        mkdir -p bin
#        curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos -o bin/yt-dlp
#        chmod +x bin/yt-dlp
#
# Output: dist/Crates.app
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Activate venv
if [ ! -d ".venv" ]; then
    echo "Error: .venv not found. Run ./launch.sh first to set up the environment."
    exit 1
fi
source .venv/bin/activate

# Check yt-dlp binary
if [ ! -f "bin/yt-dlp" ]; then
    echo "Downloading yt-dlp macOS binary..."
    mkdir -p bin
    curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos -o bin/yt-dlp
    chmod +x bin/yt-dlp
fi

# Install build dependencies
pip install -r dev-requirements.txt

# Build
echo "Building Crates.app..."
pyinstaller crates.spec --clean --noconfirm

# Ad-hoc codesign (required on macOS Ventura+)
echo "Signing app bundle..."
codesign --force --deep --sign - dist/Crates.app

echo ""
echo "Done! Your app is at: dist/Crates.app"
echo "Drag it to /Applications or double-click to launch."
