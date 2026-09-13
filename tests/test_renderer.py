from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.media.renderer import generate_recap_subtitles, export_recap_assets


def test_generate_recap_subtitles(tmp_path):
    srt_out = tmp_path / "test.srt"
    decisions = [
        {"timeline_start": 0.0, "timeline_end": 5.0, "text": "第一句字幕"},
        {"timeline_start": 5.5, "timeline_end": 10.0, "text": "第二句字幕"},
    ]
    res = generate_recap_subtitles(decisions, srt_out)
    assert res.exists()
    content = res.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:05,000" in content
    assert "第一句字幕" in content
    assert "00:00:05,500 --> 00:00:10,000" in content
    assert "第二句字幕" in content


def test_atomic_export_recap_assets(tmp_path):
    project_dir = tmp_path / "project_1"
    renders_dir = project_dir / "renders"
    renders_dir.mkdir(parents=True)

    fake_video = renders_dir / "recap_final.mp4"
    fake_video.write_bytes(b"complete video data 123456789")

    fake_srt = renders_dir / "recap_subtitles.srt"
    fake_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")

    public_output = tmp_path / "public_output"

    dest_video, dest_srt = export_recap_assets(
        project_dir=project_dir,
        movie_title="Test Action Movie!",
        output_dir=public_output,
    )

    assert dest_video.exists()
    assert dest_video.name == "Test_Action_Movie_Recap.mp4"
    assert dest_video.read_bytes() == b"complete video data 123456789"

    assert dest_srt.exists()
    assert dest_srt.name == "Test_Action_Movie_Recap.srt"
    assert "Hi" in dest_srt.read_text(encoding="utf-8")

    # Verify no dangling temporary files
    tmp_files = list(public_output.glob(".tmp_*"))
    assert len(tmp_files) == 0
