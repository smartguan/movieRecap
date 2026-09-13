from __future__ import annotations
import pytest
from src.utils.timing import seconds_to_srt_time, seconds_to_vtt_time, format_duration, time_overlap

def test_seconds_to_srt_time():
    assert seconds_to_srt_time(62.5) == "00:01:02,500"

def test_seconds_to_vtt_time():
    assert seconds_to_vtt_time(62.5) == "00:01:02.500"

def test_format_duration():
    assert format_duration(3661) == "1h 1m 1s"
    assert format_duration(65) == "1m 5s"
    assert format_duration(30) == "30s"

def test_time_overlap():
    assert time_overlap(0, 10, 5, 15) == 5.0
    assert time_overlap(0, 5, 5, 10) == 0.0
    assert time_overlap(0, 10, 2, 8) == 6.0
    assert time_overlap(10, 20, 0, 5) == 0.0
