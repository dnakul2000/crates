#!/usr/bin/env bash
# Crates launcher — handles venv setup and launch
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"

# Create venv if missing
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

# Install/update deps only if requirements.txt changed since last install
STAMP="$VENV/.deps-installed"
if [ ! -f "$STAMP" ] || [ "$DIR/requirements.txt" -nt "$STAMP" ]; then
    echo "Installing dependencies (this may take a while on first run)..."
    pip install -r "$DIR/requirements.txt"
    touch "$STAMP"
fi

exec python "$DIR/main.py" "$@"
