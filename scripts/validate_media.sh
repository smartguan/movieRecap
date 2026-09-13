#!/usr/bin/env bash
set -euo pipefail

# Usage: ./validate_media.sh <input_file>

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_file>" >&2
    exit 1
fi

INPUT="$1"

if [ ! -f "$INPUT" ]; then
    echo "{\"error\": \"File does not exist: $INPUT\"}"
    exit 1
fi

if [ ! -r "$INPUT" ]; then
    echo "{\"error\": \"File is not readable: $INPUT\"}"
    exit 1
fi

# Run ffprobe and output JSON
ffprobe -v quiet -print_format json -show_format -show_streams "$INPUT"
