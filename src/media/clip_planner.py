"""
Clip Planning and Edit Decision List (EDL) worker (FR-7).

Selects source movie scenes that correspond to each narration segment,
adjusts clip durations to match voice-over audio length, and extracts
trimmed video clips deterministically via FFmpeg.
Guarantees forward chronological progression and full timeline coverage
without repetitive or looping video clips.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_source_duration(source_video: Path, scene_index: dict[str, Any]) -> float:
    """Determine the source video duration in seconds."""
    if scene_index and scene_index.get("duration_seconds", 0) > 0:
        return float(scene_index["duration_seconds"])

    scenes = scene_index.get("scenes", []) if scene_index else []
    if scenes:
        return max(float(s.get("end_seconds", 0.0)) for s in scenes)

    if source_video.exists():
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(source_video),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return max(1.0, float(res.stdout.strip()))
        except Exception:
            pass

    return 600.0


def plan_and_extract_clips(
    source_video: Path | str | None = None,
    voice_assets: list[dict[str, Any]] | list[Any] | None = None,
    scene_index: dict[str, Any] | None = None,
    output_clips_dir: Path | str | None = None,
    script: dict[str, Any] | None = None,
    source_video_path: Path | str | None = None,
    clips_dir: Path | str | None = None,
    story_understanding: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Plan edit decisions and extract trimmed video segments.
    Enforces semantic content matching, timeline progression, full-movie coverage,
    and anti-looping deduplication (ADR 0005).

    Args:
        source_video: Path to source movie video file.
        voice_assets: List of voice assets with exact duration per segment.
        scene_index: Scene index with start/end timestamps and transcripts.
        output_clips_dir: Directory to store trimmed clip files.
        script: Optional script dict.
        source_video_path: Alias for source_video.
        clips_dir: Alias for output_clips_dir.
        story_understanding: Structured story understanding dict with local summaries.

    Returns:
        List of edit decisions with clip paths and timeline positions.
    """
    from src.media.scene_matcher import build_semantic_scene_index, find_best_scenes

    src_vid_path = Path(source_video or source_video_path or "")
    out_dir = Path(output_clips_dir or clips_dir or "clips")
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes_idx = scene_index or {}
    total_source_duration = _get_source_duration(src_vid_path, scenes_idx)

    # Build semantic scene index
    semantic_index = build_semantic_scene_index(
        scene_index=scenes_idx,
        story_understanding=story_understanding,
    )

    # Normalize voice_assets to dicts
    assets_raw = voice_assets or []
    assets: list[dict[str, Any]] = []
    for a in assets_raw:
        if hasattr(a, "model_dump"):
            assets.append(a.model_dump())
        elif isinstance(a, dict):
            assets.append(a)
        else:
            assets.append(getattr(a, "__dict__", {}))

    total_segments = len(assets)
    edit_decisions: list[dict[str, Any]] = []
    used_start_timestamps: list[float] = []
    proposed_supporting_starts: list[float] = []
    last_source_start: float = -1.0

    for i, asset in enumerate(assets):
        seg_id = asset.get("segment_id", f"narration-{i:03d}")
        needed_duration = float(asset.get("duration", 4.0))
        text = asset.get("text", "")
        supporting = asset.get("supporting_scenes", [])

        # Proportional timeline anchor across the entire movie
        target_timeline_ratio = i / max(1, total_segments)
        target_timestamp = target_timeline_ratio * max(0.0, total_source_duration - needed_duration)

        expected_phase = None
        if supporting and isinstance(supporting, list) and isinstance(supporting[0], dict):
            prop_st = float(supporting[0].get("start_seconds", 0.0))
            prop_et = float(supporting[0].get("end_seconds", prop_st + needed_duration))
            is_dup = any(abs(prop_st - prev) < 15.0 for prev in used_start_timestamps)
            is_severe_loop = (last_source_start > 0 and prop_st < last_source_start - 120.0)
            is_repeated_proposal = (prop_st in proposed_supporting_starts)
            proposed_supporting_starts.append(prop_st)

            if 0.0 <= prop_st <= total_source_duration and not is_dup and not is_severe_loop and not is_repeated_proposal:
                expected_phase = (prop_st - 40.0, prop_et + 40.0)

        is_spine_aligned = bool(
            script
            and str(script.get("version", "")).lower() not in ("v1", "v2")
            and supporting
            and isinstance(supporting, list)
            and isinstance(supporting[0], dict)
        )
        matched_scene_id = ""
        best_score = 1.0

        if is_spine_aligned:
            chosen_start = float(supporting[0].get("start_seconds", 0.0))
            matched_scene_id = asset.get("sequence_id") or f"{script.get('version', 'v4')}-seq-{i:03d}"
        else:
            # Query semantic scene matcher for best content-matched candidate (V2)
            candidates = find_best_scenes(
                semantic_index=semantic_index,
                narration_text=text,
                target_timestamp=target_timestamp,
                needed_duration=needed_duration,
                used_start_timestamps=used_start_timestamps,
                expected_phase=expected_phase,
                last_source_start=last_source_start,
                min_spacing=8.0,
                top_k=1,
            )

            if candidates:
                best_scene, best_score = candidates[0]
                chosen_start = best_scene.start_seconds
                matched_scene_id = best_scene.scene_id
            else:
                chosen_start = target_timestamp
                best_score = 0.0

        # Final clamping to ensure valid video boundaries
        src_start = max(0.0, min(chosen_start, max(0.0, total_source_duration - needed_duration)))
        src_end = src_start + needed_duration

        used_start_timestamps.append(src_start)
        last_source_start = src_start

        clip_file = out_dir / f"clip_{seg_id}.mp4"

        # 4. Extract and re-encode clip to standard 1080p MP4 H.264
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(src_start),
            "-i", str(src_vid_path),
            "-t", str(needed_duration),
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-an",  # Strip original source audio
            str(clip_file),
        ]

        logger.info(
            "Extracting clip %s: src=[%.1fs - %.1fs] (target_ratio=%.1f%%, needed_dur=%.2fs)",
            seg_id,
            src_start,
            src_end,
            (src_start / max(1.0, total_source_duration)) * 100.0,
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
            "audio_file": asset.get("audio_file", ""),
            "text": asset.get("text", ""),
            "timeline_start": asset.get("start_time", 0.0),
            "duration": needed_duration,
            "timeline_end": asset.get("end_time", needed_duration),
            "source_start": src_start,
            "source_end": src_end,
            "scene_id": matched_scene_id,
            "semantic_score": best_score,
        }
        edit_decisions.append(decision)

    return edit_decisions
