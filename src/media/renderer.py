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
    # Render to an atomic temporary staging file first so readers never see partial/truncated streams
    temp_render_output = render_dir / f".tmp_{output_video_path.name}"
    if temp_render_output.exists():
        temp_render_output.unlink()

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
                str(temp_render_output.resolve()),
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
            str(temp_render_output),
        ]
        subprocess.run(cmd_soft, capture_output=True, text=True, check=True)

    # 7. Validate that temp render file exists, is non-empty, and finalized
    if not temp_render_output.exists() or temp_render_output.stat().st_size == 0:
        raise RuntimeError(f"Rendering failed: staging file {temp_render_output} was not created")

    # 8. Atomically replace destination path
    temp_render_output.replace(output_video_path)
    logger.info("Atomically finalized rendered recap to %s (%d bytes)", output_video_path, output_video_path.stat().st_size)

    # Clean up intermediate files
    for temp in [raw_video, raw_audio, concat_list_file, audio_concat_file]:
        if temp.exists():
            temp.unlink()

    return output_video_path


def export_recap_assets(
    project_dir: Path,
    movie_title: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Atomically export finalized recap video and subtitles to the public output folder.

    Guarantees no external process sees a partially copied/truncated video.

    Args:
        project_dir: Path to project directory.
        movie_title: Name of the movie for clean naming.
        output_dir: Public output directory (e.g. data/output or output/).

    Returns:
        Tuple of (exported_video_path, exported_subtitle_path).
    """
    import shutil
    output_dir.mkdir(parents=True, exist_ok=True)

    src_video = project_dir / "renders" / "recap_final.mp4"
    src_srt = project_dir / "renders" / "recap_subtitles.srt"

    if not src_video.exists():
        raise FileNotFoundError(f"Source recap video not found for export: {src_video}")

    safe_title = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in movie_title).strip("_") or "Recap"
    dest_video = output_dir / f"{safe_title}_Recap.mp4"
    dest_srt = output_dir / f"{safe_title}_Recap.srt"

    # Atomic copy for video: copy to .tmp first, then atomic replace
    tmp_video = output_dir / f".tmp_{dest_video.name}"
    shutil.copy2(src_video, tmp_video)
    tmp_video.replace(dest_video)

    # Copy subtitles if present
    if src_srt.exists():
        tmp_srt = output_dir / f".tmp_{dest_srt.name}"
        shutil.copy2(src_srt, tmp_srt)
        tmp_srt.replace(dest_srt)

    logger.info("Atomically exported recap video to %s (%d bytes)", dest_video, dest_video.stat().st_size)
    return dest_video, dest_srt
