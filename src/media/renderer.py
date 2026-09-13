"""
Video Assembly and Rendering worker (FR-8).

Concatenates edit clips, synchronizes narration audio tracks,
generates burned-in/styled subtitles, and produces a final 1080p MP4 recap video.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from src.utils.timing import seconds_to_srt_time

logger = logging.getLogger(__name__)


def generate_recap_subtitles(edit_decisions: list[dict[str, Any]], srt_path: Path) -> Path:
    """
    Generate an SRT subtitle file from timeline edit decisions.

    Args:
        edit_decisions: List of edit decision dicts with timing and text.
        srt_path: Output .srt file destination.

    Returns:
        Path to the written SRT file.
    """
    lines = []
    for i, dec in enumerate(edit_decisions, start=1):
        start_str = seconds_to_srt_time(dec["timeline_start"])
        end_str = seconds_to_srt_time(dec["timeline_end"])
        text = dec.get("text", "").strip()

        lines.append(str(i))
        lines.append(f"{start_str} --> {end_str}")
        lines.append(text)
        lines.append("")

    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return srt_path


def render_recap_video(
    edit_decisions: list[dict[str, Any]],
    output_video_path: Path,
    burn_subtitles: bool = True,
) -> Path:
    """
    Render complete 1080p recap video with synchronized voice-over and subtitles.

    Args:
        edit_decisions: List of edit decision dicts.
        output_video_path: Final output MP4 destination.
        burn_subtitles: Whether to burn subtitles directly into video frames.

    Returns:
        Path to the rendered video file.
    """
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    render_dir = output_video_path.parent

    # 1. Create video concat manifest
    concat_list_file = render_dir / "concat_clips.txt"
    concat_content = []
    for dec in edit_decisions:
        clip_path = Path(dec["clip_file"]).resolve()
        concat_content.append(f"file '{clip_path}'")
    concat_list_file.write_text("\n".join(concat_content), encoding="utf-8")

    # 2. Create audio concat manifest
    audio_concat_file = render_dir / "concat_audio.txt"
    audio_content = []
    for dec in edit_decisions:
        audio_path = Path(dec["audio_file"]).resolve()
        audio_content.append(f"file '{audio_path}'")
    audio_concat_file.write_text("\n".join(audio_content), encoding="utf-8")

    # 3. Concatenate video clips into a single video stream
    raw_video = render_dir / "raw_concat_video.mp4"
    cmd_vconcat = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_file),
        "-c", "copy",
        str(raw_video),
    ]
    subprocess.run(cmd_vconcat, capture_output=True, text=True, check=True)

    # 4. Concatenate voice-over audio files into a single audio track
    raw_audio = render_dir / "raw_concat_audio.mp3"
    cmd_aconcat = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(audio_concat_file),
        "-c", "copy",
        str(raw_audio),
    ]
    subprocess.run(cmd_aconcat, capture_output=True, text=True, check=True)

    # 5. Generate subtitles
    srt_file = render_dir / "recap_subtitles.srt"
    generate_recap_subtitles(edit_decisions, srt_file)

    # 6. Final multiplexing and subtitle integration
    # Try burn-in using relative filename in cwd=render_dir
    rendered = False
    if burn_subtitles:
        try:
            cmd_burn = [
                "ffmpeg",
                "-y",
                "-i", "raw_concat_video.mp4",
                "-i", "raw_concat_audio.mp3",
                "-vf", "subtitles=recap_subtitles.srt:force_style='FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=2,Shadow=0,MarginV=35,Alignment=2'",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "20",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                str(output_video_path.resolve()),
            ]
            subprocess.run(cmd_burn, cwd=render_dir, capture_output=True, text=True, check=True)
            rendered = True
        except Exception as e:
            logger.warning("Burn-in subtitle filter failed (%s), embedding subtitles as soft track", e)

    if not rendered:
        # Multiplex video + audio + embedded mov_text subtitle track
        cmd_soft = [
            "ffmpeg",
            "-y",
            "-i", str(raw_video),
            "-i", str(raw_audio),
            "-i", str(srt_file),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "192k",
            "-c:s", "mov_text",
            "-metadata:s:s:0", "language=chi",
            "-shortest",
            str(output_video_path),
        ]
        subprocess.run(cmd_soft, capture_output=True, text=True, check=True)

    # Clean up intermediate files
    for temp in [raw_video, raw_audio, concat_list_file, audio_concat_file]:
        if temp.exists():
            temp.unlink()

    return output_video_path
