"""
Unit tests for Multi-Platform Export Packager and Chapter Generation.
"""

from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.media.platform_exporter import (
    PlatformExporter,
    generate_cover_thumbnail,
    generate_vtt_subtitles,
    generate_youtube_chapters,
)


def test_generate_youtube_chapters() -> None:
    edit_decisions = [
        {"timeline_start": 0.0, "timeline_end": 15.0, "text": "精彩的故事从此开始，李度遭遇人生变故。"},
        {"timeline_start": 35.5, "timeline_end": 50.0, "text": "妻子杨策鼓励他振作起来，二人共同奋斗。"},
        {"timeline_start": 82.0, "timeline_end": 100.0, "text": "最终真相大白，两人收获了圆满的结局。"},
    ]

    chapters = generate_youtube_chapters(edit_decisions)
    lines = chapters.splitlines()

    assert len(lines) == 3
    assert lines[0].startswith("00:00")
    assert lines[1].startswith("00:35")
    assert lines[2].startswith("01:22")
    assert "精彩的故事从此" in lines[0] or "精彩开场" in lines[0]


def test_generate_vtt_subtitles(tmp_path: Path) -> None:
    edit_decisions = [
        {"timeline_start": 0.0, "timeline_end": 5.25, "text": "第一句字幕"},
        {"timeline_start": 5.25, "timeline_end": 10.5, "text": "第二句字幕"},
    ]

    vtt_file = tmp_path / "subtitles.vtt"
    generate_vtt_subtitles(edit_decisions, vtt_file)

    assert vtt_file.exists()
    content = vtt_file.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:05.250" in content
    assert "第一句字幕" in content


@patch("src.media.platform_exporter.subprocess.run")
def test_platform_exporter_package(mock_subproc, tmp_path: Path) -> None:
    mock_subproc.return_value = MagicMock(returncode=0)

    project_dir = tmp_path / "proj_123"
    project_dir.mkdir()
    (project_dir / "renders").mkdir()
    (project_dir / "assets").mkdir()
    (project_dir / "keyframes").mkdir()

    # Create mock render files
    (project_dir / "renders" / "recap_final.mp4").write_bytes(b"dummy mp4")
    (project_dir / "renders" / "recap_subtitles.srt").write_text("1\n00:00:00,000 --> 00:00:05,000\n字幕\n", encoding="utf-8")

    edit_decisions = [
        {"timeline_start": 0.0, "timeline_end": 5.0, "text": "开场白", "clip_file": "/tmp/c.mp4", "audio_file": "/tmp/a.mp3"}
    ]
    (project_dir / "assets" / "edit_decisions.json").write_text(json.dumps(edit_decisions), encoding="utf-8")

    # Mock quality & token reports
    (project_dir / "recap_quality_report.md").write_text("# QA Passed", encoding="utf-8")
    (project_dir / "token_profile.md").write_text("# Token Profile", encoding="utf-8")

    output_root = tmp_path / "output"
    exporter = PlatformExporter(output_root=output_root)

    metadata = {
        "title": "测试电影",
        "synopsis": "这是一部精彩的电影",
        "stars": ["演员A", "演员B"],
        "genre": "动作,科幻",
    }

    result = exporter.export_package(
        project_dir=project_dir,
        movie_title="测试电影",
        metadata=metadata,
    )

    package_folder = output_root / "测试电影"
    assert package_folder.exists()
    assert (package_folder / "recap_youtube.mp4").exists()
    assert (package_folder / "recap_bilibili.mp4").exists()
    assert (package_folder / "metadata_youtube.json").exists()
    assert (package_folder / "metadata_bilibili.json").exists()
    assert (package_folder / "subtitles.srt").exists()
    assert (package_folder / "subtitles.vtt").exists()
    assert (package_folder / "cover.jpg").exists()

    # Check YouTube metadata
    with open(package_folder / "metadata_youtube.json", "r", encoding="utf-8") as f:
        yt_meta = json.load(f)
        assert "测试电影" in yt_meta["title"]
        assert "演员A" in yt_meta["tags"]
        assert "章节导航" in yt_meta["description"]
        assert yt_meta["categoryId"] == "1"

    # Check Bilibili metadata
    with open(package_folder / "metadata_bilibili.json", "r", encoding="utf-8") as f:
        bili_meta = json.load(f)
        assert "测试电影" in bili_meta["title"]
        assert bili_meta["tid"] == 182
        assert "演员A" in bili_meta["tag"]
