#!/usr/bin/env bash
set -euo pipefail

# Usage: ./extract_audio.sh <input> <output> [format] [sample_rate]

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input> <output> [format] [sample_rate]" >&2
    exit 1
fi

INPUT="$1"
OUTPUT="$2"
FORMAT="${3:-mp3}"
SAMPLE_RATE="${4:-44100}"

ffmpeg -i "$INPUT" -vn -ar "$SAMPLE_RATE" -f "$FORMAT" -y "$OUTPUT" 2>/dev/null

echo "{\"status\": \"success\", \"output\": \"$OUTPUT\"}"
