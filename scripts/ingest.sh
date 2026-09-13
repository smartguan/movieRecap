#!/usr/bin/env bash
set -euo pipefail

# Usage: ./ingest.sh <input> <output_dir>

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input> <output_dir>" >&2
    exit 1
fi

INPUT="$1"
OUTPUT_DIR="$2"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting ingestion pipeline..."

echo "1. Validating media..."
"$SCRIPTS_DIR/validate_media.sh" "$INPUT" > "$OUTPUT_DIR/probe.json"

echo "2. Extracting audio..."
"$SCRIPTS_DIR/extract_audio.sh" "$INPUT" "$OUTPUT_DIR/audio.mp3"

echo "3. Detecting scenes..."
"$SCRIPTS_DIR/detect_scenes.sh" "$INPUT" "$OUTPUT_DIR/scenes"

echo "Pipeline completed successfully."
echo "{\"status\": \"success\", \"output_dir\": \"$OUTPUT_DIR\"}"
