from __future__ import annotations
import pytest
from pathlib import Path
from src.media.subtitles import SubtitleEntry, subtitle_time_to_seconds, parse_srt, parse_vtt, search_subtitles, get_subtitles_in_range

def test_subtitle_time_to_seconds():
    assert subtitle_time_to_seconds("00:01:02,000") == 62.0
    assert subtitle_time_to_seconds("01:00:00,500") == 3600.5

def test_parse_srt(tmp_path):
    srt_content = "1\n00:00:01,000 --> 00:00:02,000\nHello world\n\n"
    srt_file = tmp_path / "test.srt"
    srt_file.write_text(srt_content, encoding="utf-8")
    subs = parse_srt(srt_file)
    assert len(subs) == 1
    assert subs[0].text == "Hello world"
    assert subs[0].start_seconds == 1.0
    assert subs[0].end_seconds == 2.0

def test_search_subtitles():
    subs = [
        SubtitleEntry(1, 1.0, 2.0, "Hello world"),
        SubtitleEntry(2, 3.0, 4.0, "Goodbye"),
    ]
    results = search_subtitles(subs, "Hello")
    assert len(results) == 1
    assert results[0].index == 1

def test_get_subtitles_in_range():
    subs = [
        SubtitleEntry(1, 1.0, 2.0, "Hello"),
        SubtitleEntry(2, 3.0, 4.0, "World"),
    ]
    results = get_subtitles_in_range(subs, 1.0, 3.5)
    assert len(results) == 2
