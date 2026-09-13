"""
Clip Planning and Edit Decision List (EDL) worker (FR-7).

Selects source movie scenes that correspond to each narration segment,
adjusts clip durations to match voice-over audio length, and extracts
trimmed video clips deterministically via FFmpeg.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def plan_and_extract_clips(
    source_video: Path,
    voice_assets: list[dict[str, Any]],
    scene_index: dict[str, Any],
    output_clips_dir: Path,
) -> list[dict[str, Any]]:
    """
    Plan edit decisions and extract trimmed video segments.

    Args:
        source_video: Path to source movie video file.
        voice_assets: List of voice assets with exact duration per segment.
        scene_index: Scene index with start/end timestamps.
        output_clips_dir: Directory to store trimmed clip files.

    Returns:
        List of edit decisions with clip paths and timeline positions.
    """
    output_clips_dir.mkdir(parents=True, exist_ok=True)
    scenes = scene_index.get("scenes", [])
    total_scenes = len(scenes)

    edit_decisions = []

    for i, asset in enumerate(voice_assets):
        seg_id = asset["segment_id"]
        needed_duration = asset["duration"]
        supporting = asset.get("supporting_scenes", [])

        # 1. Determine source range to sample from
        if supporting and supporting[0].get("end_seconds", 0) > supporting[0].get("start_seconds", 0):
            src_start = float(supporting[0]["start_seconds"])
            src_end = float(supporting[0]["end_seconds"])
        elif total_scenes > 0:
            # Fallback: distribute proportionately across available scenes
            scene_idx = min(int((i / max(1, len(voice_assets))) * total_scenes), total_scenes - 1)
            src_start = float(scenes[scene_idx].get("start_seconds", 0.0))
            src_end = float(scenes[scene_idx].get("end_seconds", src_start + needed_duration))
        else:
            src_start = float(i * 30.0)
            src_end = src_start + needed_duration

        # Clip length must match voiceover duration (with a small buffer or looping if needed)
        clip_file = output_clips_dir / f"clip_{seg_id}.mp4"

        # 2. Extract and re-encode clip to standard 1080p MP4 H.264
        # Handle video looping/slowing if source range is shorter than voiceover
        actual_source_duration = max(0.5, src_end - src_start)
        
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(src_start),
            "-i", str(source_video),
            "-t", str(needed_duration),
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-an",  # Strip original source audio; narration will be mixed in
            str(clip_file),
        ]

        logger.info(
            "Extracting clip %s: src=[%.1fs - %.1fs], target_duration=%.2fs",
            seg_id,
            src_start,
            src_start + needed_duration,
            needed_duration,
        )

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error("FFmpeg clip extraction failed for %s: %s", seg_id, e.stderr)
            raise

        decision = {
            "segment_id": seg_id,
            "clip_file": str(clip_file),
            "audio_file": asset["audio_file"],
            "text": asset["text"],
            "timeline_start": asset["start_time"],
            "duration": needed_duration,
            "timeline_end": asset["end_time"],
            "source_start": src_start,
            "source_end": src_start + needed_duration,
        }
        edit_decisions.append(decision)

    return edit_decisions
