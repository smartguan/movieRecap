"""
Tests for Evaluation & Anti-AI-Slop Framework.
"""

from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.eval.cliches import (
    calculate_lexical_diversity,
    find_repeated_phrases,
    scan_for_cliches,
)
from src.eval.deterministic_eval import evaluate_deterministic
from src.eval.models import EvalGrade
from src.eval.slop_detector import SlopDetector


def test_scan_for_cliches():
    text_with_cliches = "在这个充满科技感的世界中，不得不说，男主做出了一个选择。究竟会发生什么呢？让我们拭目以待。总的来说这是一部好电影。"
    cliches = scan_for_cliches(text_with_cliches)
    assert len(cliches) >= 3

    clean_text = "阿姆斯特丹教堂的残垣断壁在末世余晖下格外肃穆。汤姆握紧机械义肢，在记忆的回廊里寻找着解救世界的密码。"
    clean_cliches = scan_for_cliches(clean_text)
    assert len(clean_cliches) == 0


def test_calculate_lexical_diversity():
    repetitive_text = "电影电影电影电影电影电影"
    rep_metrics = calculate_lexical_diversity(repetitive_text)
    assert rep_metrics["ttr"] < 0.35

    diverse_text = "阿姆斯特丹的黄昏沉浸在赛博机械巨兽的阴影中，古老教堂成为抵抗军最后的避难所。"
    div_metrics = calculate_lexical_diversity(diverse_text)
    assert div_metrics["ttr"] > 0.70
    assert div_metrics["distinct_2"] > 0.85


def test_find_repeated_phrases():
    text = "机械飞升的执念机械飞升的执念机械飞升的执念"
    repeated = find_repeated_phrases(text, ngram_size=4, min_count=3)
    assert len(repeated) > 0
    assert repeated[0][0] == "机械飞升"


def test_deterministic_eval_clean_script():
    clean_script = {
        "title": "Tears of Steel",
        "segments": [
            {
                "segment_id": "narration-000",
                "segment_type": "hook",
                "text": "如果一段四十年前无疾而终的恋情引发浩劫，你会如何面对？",
                "supporting_scenes": [{"start_seconds": 35.0, "end_seconds": 90.0}],
            },
            {
                "segment_id": "narration-001",
                "segment_type": "plot_and_commentary",
                "text": "阿姆斯特丹的古老教堂如今成了抵抗军的庇护所，机械巨兽在上空盘旋。",
                "supporting_scenes": [{"start_seconds": 95.0, "end_seconds": 120.0}],
            },
        ],
    }

    metrics, dims, telemetry = evaluate_deterministic(clean_script)
    assert metrics.cleanliness_score == 100.0
    assert len(metrics.hard_failures) == 0
    assert metrics.evidence_grounded_ratio == 1.0
    assert metrics.av_coupling.coupling_score >= 80.0
    assert telemetry.is_deterministic is True
    assert telemetry.total_tokens == 0
    assert telemetry.cost_usd == 0.0


def test_deterministic_eval_slop_leak():
    leaky_script = {
        "title": "Tears of Steel",
        "segments": [
            {
                "segment_id": "narration-000",
                "segment_type": "hook",
                "text": '{"text": "如果一段恋情引发浩劫...", "supporting_scenes": [{"start_seconds": 35.0}]}',
            }
        ],
    }

    metrics, dims, telemetry = evaluate_deterministic(leaky_script)
    assert metrics.cleanliness_score == 0.0
    assert len(metrics.hard_failures) > 0


def test_av_coupling_evaluation():
    script = {
        "title": "Tears of Steel",
        "segments": [
            {
                "segment_id": "narration-000",
                "text": "汤姆在实验室中与西莉亚对峙。",
                "supporting_scenes": [{"start_seconds": 35.0, "end_seconds": 55.0}],
            }
        ]
    }
    story = {
        "characters": [
            {"name": "汤姆", "description": "男主角"},
            {"name": "西莉亚", "description": "女主角"}
        ]
    }
    scene_index = {
        "scenes": [
            {"scene_id": "scene-0", "start_seconds": 35.0, "end_seconds": 55.0, "characters": ["汤姆", "西莉亚"]}
        ]
    }
    edit_decisions = [
        {"segment_id": "narration-000", "duration": 20.0, "source_start": 35.0, "source_end": 55.0}
    ]

    metrics, dims, telemetry = evaluate_deterministic(
        script=script,
        story=story,
        scene_index=scene_index,
        edit_decisions=edit_decisions,
    )
    assert metrics.av_coupling.coupling_score == 100.0
    assert metrics.av_coupling.av_duration_drift_seconds == 0.0


def test_slop_detector_evaluation():
    detector = SlopDetector()
    good_script = {
        "title": "Tears of Steel",
        "project_id": "test-good",
        "estimated_duration_minutes": 2.0,
        "segments": [
            {
                "segment_id": "narration-000",
                "segment_type": "hook",
                "text": "如果一段四十年前无疾而终的恋情引发浩劫，你会如何抉择？这部电影用极具张力的末世图景探讨了和解与遗憾。",
                "supporting_scenes": [{"start_seconds": 35.0, "end_seconds": 90.0}],
            },
            {
                "segment_id": "narration-001",
                "segment_type": "plot_and_commentary",
                "text": "未来的阿姆斯特丹教堂成为抵抗军庇护所，汤姆必须面对全息投影中西莉亚的执念。",
                "supporting_scenes": [{"start_seconds": 100.0, "end_seconds": 150.0}],
            },
        ],
    }

    story = {"themes": ["Love and Technology", "Regret and Redemption"]}
    report = detector.evaluate_script(good_script, story=story)

    assert report.total_score >= 80.0
    assert report.passed is True
    assert report.is_slop is False
    assert report.grade in (EvalGrade.TIER_S, EvalGrade.TIER_A, EvalGrade.TIER_B)
    assert len(report.eval_telemetry) >= 2
    assert any(t.is_deterministic and t.total_tokens == 0 for t in report.eval_telemetry)

    # Markdown format check
    md = detector.format_markdown_report(report)
    assert "# 🎬 Movie Recap Quality & Anti-Slop Scorecard" in md
    assert "Cross-Modal Audio-Visual Coupling Telemetry" in md
    assert "Evaluator LLM Token & Cost Telemetry" in md
