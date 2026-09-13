#!/usr/bin/env bash
set -euo pipefail

# Usage: ./detect_scenes.sh <input> <output_dir> [threshold]

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input> <output_dir> [threshold]" >&2
    exit 1
fi

INPUT="$1"
OUTPUT_DIR="$2"
THRESHOLD="${3:-27.0}"

mkdir -p "$OUTPUT_DIR"

scenedetect -i "$INPUT" detect-content -t "$THRESHOLD" list-scenes -o "$OUTPUT_DIR"

echo "{\"status\": \"success\", \"output_dir\": \"$OUTPUT_DIR\"}"
