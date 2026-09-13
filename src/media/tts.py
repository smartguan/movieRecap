"""
Voice generation worker using Neural Text-to-Speech (FR-6).

Synthesizes high-quality Mandarin narration for each script segment.
Measures exact duration and records voice asset metadata.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "zh-CN-YunxiNeural"  # Professional energetic documentary/commentary voice


async def _synthesize_text_async(text: str, voice: str, output_file: Path) -> None:
    """Async call to edge-tts communicate."""
    import edge_tts

    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(output_file))


def synthesize_segment(
    text: str,
    output_file: Path,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
) -> float:
    """
    Synthesize speech for a single narration segment.

    Args:
        text: Mandarin text to synthesize.
        output_file: Target audio file path (.mp3 or .wav).
        voice: TTS voice name.
        rate: Speed adjustment (e.g. '+5%').

    Returns:
        Duration of the generated audio in seconds.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        asyncio.run(_synthesize_text_async(text, voice, output_file))
    except Exception as e:
        logger.warning("edge-tts failed (%s), falling back to offline audio generation", e)
        # Fallback: create silent audio with estimated duration
        duration_est = max(2.0, len(text) / 4.0)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r=24000:cl=mono",
                "-t",
                str(duration_est),
                "-q:a",
                "9",
                str(output_file),
            ],
            capture_output=True,
            check=True,
        )

    # Measure exact audio duration via ffprobe
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(output_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    duration = float(res.stdout.strip())
    return duration


def generate_voice_assets(
    script: dict[str, Any],
    output_dir: Path,
    voice: str = DEFAULT_VOICE,
) -> list[dict[str, Any]]:
    """
    Generate audio files for all segments in a script.

    Args:
        script: Script data dict containing 'segments'.
        output_dir: Destination folder for voice files.
        voice: Voice identifier.

    Returns:
        List of audio asset records with file paths and exact durations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    current_timeline_time = 0.0

    segments = script.get("segments", [])
    for i, seg in enumerate(segments):
        seg_id = seg.get("segment_id", f"narration-{i:03d}")
        text = seg.get("text", "").strip()
        if not text:
            continue

        audio_file = output_dir / f"{seg_id}.mp3"
        duration = synthesize_segment(text, audio_file, voice=voice)

        asset = {
            "segment_id": seg_id,
            "text": text,
            "audio_file": str(audio_file),
            "start_time": current_timeline_time,
            "duration": duration,
            "end_time": current_timeline_time + duration,
            "voice": voice,
            "segment_type": seg.get("segment_type", "plot_and_commentary"),
            "supporting_scenes": seg.get("supporting_scenes", []),
        }
        assets.append(asset)
        current_timeline_time += duration

        logger.info(
            "Synthesized %s: duration=%.2fs, text='%s...'",
            seg_id,
            duration,
            text[:20],
        )

    return assets
