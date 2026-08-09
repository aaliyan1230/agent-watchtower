#!/bin/sh
# Renders the .excalidraw sources (assets/diagrams/*.excalidraw) to PNGs
# for the README, using @moona3k/excalidraw-export.
set -eu
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
cd "$ROOT"
python3 scripts/diagrams/build_diagrams.py
for src in assets/diagrams/*.excalidraw; do
  base="$(basename "$src" .excalidraw)"
  npx --prefix scripts/diagrams excalidraw-export "$src" -o "assets/diagrams/$base.png" --scale 2
done
