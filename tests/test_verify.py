from __future__ import annotations
import pytest
from src.ai.verify import _check_schema, _check_length, _check_phrase_frequency, _check_segment_types

def test_check_schema():
    valid = {"segments": [{"segment_id": "seg-1", "type": "narration", "text": "hello"}]}
    findings = _check_schema(valid)
    assert any(f["severity"] == "pass" for f in findings)
    assert not any(f["severity"] == "fail" for f in findings)

    invalid = {"segments": []}
    findings_inv = _check_schema(invalid)
    assert any(f["severity"] == "fail" for f in findings_inv)

def test_check_length():
    short_script = {
        "segments": [{"text": "短文本"}],
        "target_speaking_rate": 250.0,
        "target_range_minutes": [20, 30],
    }
    findings = _check_length(short_script, {})
    assert any(f["check"] == "length_too_short" and f["severity"] == "fail" for f in findings)

    # 6000 chars / 250 = 24 minutes (within 20-30 min range)
    normal_script = {
        "segments": [{"text": "正常文本内容" * 1000}],
        "target_speaking_rate": 250.0,
        "target_range_minutes": [20, 30],
    }
    findings_normal = _check_length(normal_script, {})
    assert any(f["check"] == "length_ok" and f["severity"] == "pass" for f in findings_normal)

def test_check_phrase_frequency():
    script_rep = {"segments": [{"text": "重复模式短语重复模式短语重复模式短语" * 10}]}
    findings = _check_phrase_frequency(script_rep)
    assert any(f["check"] == "phrase_repetition" and f["severity"] == "warning" for f in findings)

    script_normal = {"segments": [{"text": "这是一个完全不重复的正常中文句子，叙述了电影的基本情节。"}]}
    findings_norm = _check_phrase_frequency(script_normal)
    assert any(f["check"] == "phrase_repetition" and f["severity"] == "pass" for f in findings_norm)

def test_check_segment_types():
    script = {
        "segments": [
            {"segment_id": "1", "segment_type": "hook", "text": "hook"},
            {"segment_id": "2", "segment_type": "plot_and_commentary", "text": "commentary"},
            {"segment_id": "3", "segment_type": "conclusion", "text": "conclusion"},
        ]
    }
    findings = _check_segment_types(script)
    assert not any(f["severity"] == "warning" for f in findings)
