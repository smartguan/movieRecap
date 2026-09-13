#!/usr/bin/env bash
set -euo pipefail

# Usage: ./sample_keyframes.sh <input> <timestamps_file> <output_dir>

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input> <timestamps_file> <output_dir>" >&2
    exit 1
fi

INPUT="$1"
TIMESTAMPS_FILE="$2"
OUTPUT_DIR="$3"

mkdir -p "$OUTPUT_DIR"

# Assuming timestamps_file has one timestamp per line (in seconds)
while IFS= read -r ts || [ -n "$ts" ]; do
    if [ -n "$ts" ]; then
        output_file="${OUTPUT_DIR}/frame_${ts}.jpg"
        ffmpeg -ss "$ts" -i "$INPUT" -frames:v 1 -q:v 2 "$output_file" -y 2>/dev/null
    fi
done < "$TIMESTAMPS_FILE"

echo "{\"status\": \"success\", \"output_dir\": \"$OUTPUT_DIR\"}"
