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


def test_deterministic_eval_catches_looping_clips():
    script = {
        "title": "诅咒",
        "segments": [
            {"segment_id": "narration-000", "text": "故事开端，神秘诅咒正在蔓延。", "supporting_scenes": [{"start_seconds": 35.0, "end_seconds": 55.0}]},
            {"segment_id": "narration-001", "text": "真相逐渐浮现，主角深入调查。", "supporting_scenes": [{"start_seconds": 35.0, "end_seconds": 55.0}]},
            {"segment_id": "narration-002", "text": "高潮对决时刻，怨灵彻底爆发。", "supporting_scenes": [{"start_seconds": 35.0, "end_seconds": 55.0}]},
        ]
    }
    scene_index = {"duration_seconds": 5000.0, "scenes": [{"start_seconds": float(i * 10), "end_seconds": float(i * 10 + 10)} for i in range(500)]}
    edit_decisions = [
        {"segment_id": "narration-000", "duration": 10.0, "source_start": 35.0, "source_end": 45.0},
        {"segment_id": "narration-001", "duration": 10.0, "source_start": 35.0, "source_end": 45.0},
        {"segment_id": "narration-002", "duration": 10.0, "source_start": 35.0, "source_end": 45.0},
    ]

    metrics, dims, telemetry = evaluate_deterministic(
        script=script,
        scene_index=scene_index,
        edit_decisions=edit_decisions,
    )

    assert any("looping/duplicate" in fail for fail in metrics.hard_failures)
    av_dim = next(d for d in dims if d.name == "Cross-Modal Audio-Visual Coupling")
    assert av_dim.passed is False


@patch("subprocess.run")
def test_clip_planner_timeline_progression_and_anti_looping(mock_run, tmp_path):
    from src.media.clip_planner import plan_and_extract_clips
    
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    voice_assets = [
        {"segment_id": f"narration-{i:03d}", "duration": 10.0, "start_time": float(i * 10), "end_time": float(i * 10 + 10), "text": f"段落{i}", "supporting_scenes": [{"start_seconds": 35.0, "end_seconds": 55.0}]}
        for i in range(10)
    ]
    scene_index = {
        "duration_seconds": 5000.0,
        "scenes": [
            {"scene_id": f"s-{i}", "start_seconds": float(i * 50), "end_seconds": float(i * 50 + 40)}
            for i in range(100)
        ]
    }

    decisions = plan_and_extract_clips(
        source_video=tmp_path / "source.mp4",
        voice_assets=voice_assets,
        scene_index=scene_index,
        output_clips_dir=tmp_path / "clips",
    )

    assert len(decisions) == 10
    start_times = [d["source_start"] for d in decisions]
    # Check that timestamps advance and do not loop or repeat the same 35s
    for i in range(1, len(start_times)):
        assert start_times[i] != start_times[i - 1]
        assert abs(start_times[i] - start_times[i - 1]) >= 8.0
    
    # Check that decisions span a significant portion of the movie timeline
    assert max(start_times) > 3500.0


def test_av_semantic_phase_alignment_passes_when_synchronized():
    """Verify that synchronized narration and footage pass AV semantic alignment."""
    script = {
        "title": "诅咒",
        "segments": [
            {"segment_id": "narration-000", "text": "故事从东京静谧的理发店修剪发丝开始。", "supporting_scenes": [{"start_seconds": 10.0, "end_seconds": 30.0}]},
            {"segment_id": "narration-001", "text": "跨海抵达台北，在老旧街巷香烛道铺调查七日绝魂煞。", "supporting_scenes": [{"start_seconds": 3200.0, "end_seconds": 3250.0}]},
            {"segment_id": "narration-002", "text": "午夜在荒废神庙与红衣怨灵决战，打火机点燃母偶。", "supporting_scenes": [{"start_seconds": 4700.0, "end_seconds": 4750.0}]},
            {"segment_id": "narration-003", "text": "晨光穿透残破庙顶走出密林，重回东京街头反思。", "supporting_scenes": [{"start_seconds": 5200.0, "end_seconds": 5500.0}]},
        ]
    }
    edit_decisions = [
        {"segment_id": "narration-000", "duration": 10.0, "source_start": 20.0, "source_end": 30.0, "text": script["segments"][0]["text"]},
        {"segment_id": "narration-001", "duration": 10.0, "source_start": 3250.0, "source_end": 3260.0, "text": script["segments"][1]["text"]},
        {"segment_id": "narration-002", "duration": 10.0, "source_start": 4720.0, "source_end": 4730.0, "text": script["segments"][2]["text"]},
        {"segment_id": "narration-003", "duration": 10.0, "source_start": 5300.0, "source_end": 5310.0, "text": script["segments"][3]["text"]},
    ]
    scene_index = {"duration_seconds": 5600.0, "scenes": []}

    metrics, dims, _ = evaluate_deterministic(
        script=script,
        scene_index=scene_index,
        edit_decisions=edit_decisions,
    )

    assert metrics.av_coupling.av_semantic_alignment_score == 100.0
    assert metrics.av_coupling.phase_mismatch_count == 0
    av_dim = next(d for d in dims if d.name == "Cross-Modal Audio-Visual Coupling")
    assert av_dim.passed is True


def test_av_semantic_phase_alignment_fails_when_disjointed():
    """Verify that severe phase mismatches (e.g. Taiwan audio with Tokyo video) trigger QA failure."""
    script = {
        "title": "诅咒",
        "segments": [
            # Taiwan audio paired with Tokyo salon footage (50s)
            {"segment_id": "narration-000", "text": "跨海抵达台北，在老旧街巷香烛道铺调查七日绝魂煞。", "supporting_scenes": [{"start_seconds": 3200.0, "end_seconds": 3250.0}]},
            # Temple fight audio paired with Tokyo apartment footage (800s)
            {"segment_id": "narration-001", "text": "午夜在荒废神庙与红衣怨灵决战，防风打火机点燃母偶。", "supporting_scenes": [{"start_seconds": 4700.0, "end_seconds": 4750.0}]},
            # Tokyo salon audio paired with Temple ghost footage (4800s)
            {"segment_id": "narration-002", "text": "故事从东京理发店修剪发丝开始收音机播放轻缓音乐。", "supporting_scenes": [{"start_seconds": 10.0, "end_seconds": 30.0}]},
        ]
    }
    edit_decisions = [
        {"segment_id": "narration-000", "duration": 10.0, "source_start": 50.0, "source_end": 60.0, "text": script["segments"][0]["text"]},
        {"segment_id": "narration-001", "duration": 10.0, "source_start": 800.0, "source_end": 810.0, "text": script["segments"][1]["text"]},
        {"segment_id": "narration-002", "duration": 10.0, "source_start": 4800.0, "source_end": 4810.0, "text": script["segments"][2]["text"]},
    ]
    scene_index = {"duration_seconds": 5600.0, "scenes": []}

    metrics, dims, _ = evaluate_deterministic(
        script=script,
        scene_index=scene_index,
        edit_decisions=edit_decisions,
    )

    assert metrics.av_coupling.phase_mismatch_count >= 3
    assert metrics.av_coupling.av_semantic_alignment_score < 50.0
    assert any("Severe AV semantic misalignment" in fail for fail in metrics.hard_failures)
    av_dim = next(d for d in dims if d.name == "Cross-Modal Audio-Visual Coupling")
    assert av_dim.passed is False


def test_av_scene_content_matching_eval():
    """Verify that evaluate_deterministic evaluates content-level semantic alignment."""
    script = {
        "title": "诅咒",
        "segments": [
            {"segment_id": "narration-000", "text": "故事从东京理发店修剪发丝开始。", "supporting_scenes": [{"start_seconds": 200.0, "end_seconds": 230.0}]},
            {"segment_id": "narration-001", "text": "跨海抵达台北，在老旧街巷调查。", "supporting_scenes": [{"start_seconds": 2600.0, "end_seconds": 2630.0}]},
        ]
    }
    scene_index = {
        "duration_seconds": 5000.0,
        "scenes": [
            {"scene_id": "sc-001", "start_seconds": 200.0, "end_seconds": 230.0, "transcript_text": "カットとカラーでお願いします"},
            {"scene_id": "sc-002", "start_seconds": 2600.0, "end_seconds": 2630.0, "transcript_text": "私も台湾に行く"},
        ]
    }
    edit_decisions = [
        {"segment_id": "narration-000", "duration": 10.0, "source_start": 200.0, "source_end": 210.0, "text": script["segments"][0]["text"]},
        {"segment_id": "narration-001", "duration": 10.0, "source_start": 2600.0, "source_end": 2610.0, "text": script["segments"][1]["text"]},
    ]

    metrics, dims, _ = evaluate_deterministic(
        script=script,
        scene_index=scene_index,
        edit_decisions=edit_decisions,
    )

    assert metrics.av_coupling.scene_content_match_score >= 80.0
    assert len(metrics.av_coupling.low_match_segments) == 0


def test_storyteller_continuity_ideal_script():
    """Verify that a smoothly transitioned, chronologically progressing script earns high continuity score."""
    from src.eval.deterministic_eval import calculate_storyteller_continuity

    segments = [
        {
            "segment_id": "narration-000",
            "text": "如果一段早已离世的挚友账号突然更新，你会怎么做？故事从东京一家静谧的理发店拉开序幕。",
            "supporting_scenes": [{"start_seconds": 30.0, "end_seconds": 80.0}],
        },
        {
            "segment_id": "narration-001",
            "text": "随后，同伴神色慌张地展示手机上弹出的诡异纸人咒符，恐怖阴影开始蔓延。",
            "supporting_scenes": [{"start_seconds": 150.0, "end_seconds": 210.0}],
        },
        {
            "segment_id": "narration-002",
            "text": "为了弄清真相并救下同伴，女主毅然跨海来到台北，深入老旧道铺寻找破解之法。",
            "supporting_scenes": [{"start_seconds": 2800.0, "end_seconds": 2860.0}],
        },
        {
            "segment_id": "narration-003",
            "text": "在惨烈搏杀过后，夜幕终被晨曦撕破，伤痕累累的两人终于走出了密林。",
            "supporting_scenes": [{"start_seconds": 4800.0, "end_seconds": 4850.0}],
        },
    ]
    story = {
        "characters": [
            {"name": "女主", "description": "主角"},
            {"name": "同伴", "description": "好友"},
        ]
    }

    metrics, dim = calculate_storyteller_continuity(segments, story=story, total_source_duration=5000.0)

    assert metrics.backward_jump_count == 0
    assert metrics.temporal_monotonicity_score == 100.0
    assert metrics.transition_coherence_ratio == 1.0
    assert metrics.unbridged_jump_count == 0
    assert metrics.continuity_score >= 85.0
    assert dim.passed is True
    assert dim.name == "Storyteller Narrative Continuity"


def test_storyteller_continuity_detects_backward_timeline_regressions():
    """Verify that severe backward jumps are detected and penalize the storyteller continuity score."""
    from src.eval.deterministic_eval import calculate_storyteller_continuity

    segments = [
        {
            "segment_id": "narration-000",
            "text": "故事在东京平静地展开。",
            "supporting_scenes": [{"start_seconds": 100.0, "end_seconds": 150.0}],
        },
        {
            "segment_id": "narration-001",
            "text": "随后在台北的神庙与恶灵生死决战。",
            "supporting_scenes": [{"start_seconds": 4500.0, "end_seconds": 4550.0}],
        },
        {
            "segment_id": "narration-002",
            "text": "突然又回到了理发店洗头剪发。",
            "supporting_scenes": [{"start_seconds": 200.0, "end_seconds": 250.0}],
        },
    ]

    metrics, dim = calculate_storyteller_continuity(segments)

    assert metrics.backward_jump_count == 1
    assert metrics.temporal_monotonicity_score < 80.0
    assert any("Major backward timeline regression" in e for e in metrics.discontinuity_events)
    assert dim.passed is False


def test_storyteller_continuity_detects_unbridged_scene_jumps():
    """Verify that jumping forward hundreds of seconds without transition words lowers transition coherence."""
    from src.eval.deterministic_eval import calculate_storyteller_continuity

    segments = [
        {
            "segment_id": "narration-000",
            "text": "故事在东京展开。",
            "supporting_scenes": [{"start_seconds": 50.0, "end_seconds": 100.0}],
        },
        {
            "segment_id": "narration-001",
            # Abrupt jump from 100s to 3000s without any transition markers ("随后", "为了", "来到", etc.)
            "text": "香烛店的老板看着符文冷笑，店里的香炉升起青烟。",
            "supporting_scenes": [{"start_seconds": 3000.0, "end_seconds": 3060.0}],
        },
    ]

    metrics, dim = calculate_storyteller_continuity(segments)

    assert metrics.scene_jump_count == 1
    assert metrics.unbridged_jump_count == 1
    assert metrics.transition_coherence_ratio == 0.0
    assert any("without connective transition phrasing" in e for e in metrics.discontinuity_events)


def test_check_av_semantic_alignment_detects_activity_conflict():
    """Verify that claiming salon haircut during a kitchen dining scene triggers an AV conflict."""
    from src.eval.deterministic_eval import check_av_semantic_alignment

    segments = [
        {
            "segment_id": "narration-002",
            "text": "午休时分，理发师正在为顾客修剪发丝，交流发型细节。",
            "supporting_scenes": [{"start_seconds": 332.0, "end_seconds": 380.0}],
        }
    ]
    edit_decisions = [
        {
            "segment_id": "narration-002",
            "source_start": 332.0,
            "source_end": 380.0,
            "text": "午休时分，理发师正在为顾客修剪发丝，交流发型细节。",
        }
    ]
    scene_index = {
        "scenes": [
            {
                "scene_id": "scene-0037",
                "start_seconds": 332.0,
                "end_seconds": 380.0,
                "transcript_text": "サイコーンでしょ食べる 今日はいい感じで出来ました いただきます",
                "characters": ["女主"],
            }
        ]
    }

    score, mismatches, findings = check_av_semantic_alignment(
        segments=segments,
        edit_decisions=edit_decisions,
        total_source_duration=1000.0,
        movie_title="Generic Movie",
        scene_index=scene_index,
    )

    assert mismatches >= 1
    assert score < 100.0
    assert any("AV Activity Conflict" in f for f in findings)


def test_check_av_semantic_alignment_passes_when_grounded():
    """Verify that matching cooking narration to dining dialogue passes with 100%."""
    from src.eval.deterministic_eval import check_av_semantic_alignment

    segments = [
        {
            "segment_id": "narration-002",
            "text": "夜幕降临，两人回到公寓厨房一起下厨烹饪家常晚餐，享用热气腾腾的美食。",
            "supporting_scenes": [{"start_seconds": 332.0, "end_seconds": 380.0}],
        }
    ]
    edit_decisions = [
        {
            "segment_id": "narration-002",
            "source_start": 332.0,
            "source_end": 380.0,
            "text": "夜幕降临，两人回到公寓厨房一起下厨烹饪家常晚餐，享用热气腾腾的美食。",
        }
    ]
    scene_index = {
        "scenes": [
            {
                "scene_id": "scene-0037",
                "start_seconds": 332.0,
                "end_seconds": 380.0,
                "transcript_text": "サイコーンでしょ食べる 今日はいい感じで出来ました いただきます",
                "characters": ["女主"],
            }
        ]
    }

    score, mismatches, findings = check_av_semantic_alignment(
        segments=segments,
        edit_decisions=edit_decisions,
        total_source_duration=1000.0,
        movie_title="Generic Movie",
        scene_index=scene_index,
    )

    assert mismatches == 0
    assert score == 100.0


def test_slop_detector_blocks_grade_s_when_critical_dimension_fails():
    """Verify that an AV coupling failure blocks Grade S and marks the project as failed (Tier F)."""
    detector = SlopDetector()
    mismatched_script = {
        "title": "Generic Movie",
        "project_id": "test-mismatch",
        "estimated_duration_minutes": 2.0,
        "segments": [
            {
                "segment_id": "narration-000",
                "segment_type": "hook",
                "text": "如果一个离奇的秘密被揭晓，你会如何选择？深度解说高能电影《Generic Movie》。",
                "supporting_scenes": [{"start_seconds": 10.0, "end_seconds": 50.0}],
            },
            {
                "segment_id": "narration-001",
                "segment_type": "plot_and_commentary",
                "text": "女主在理发店修剪发丝，理发师正在给顾客洗头设计发型。",
                "supporting_scenes": [{"start_seconds": 332.0, "end_seconds": 380.0}],
            },
        ],
    }
    scene_index = {
        "scenes": [
            {"scene_id": "s1", "start_seconds": 10.0, "end_seconds": 50.0, "transcript_text": "こんにちは"},
            {"scene_id": "s2", "start_seconds": 332.0, "end_seconds": 380.0, "transcript_text": "サイコーンでしょ食べる いただきます 料理"},
        ]
    }
    edit_decisions = [
        {"segment_id": "narration-000", "source_start": 10.0, "source_end": 50.0, "duration": 40.0, "text": "如果一个离奇的秘密被揭晓，你会如何选择？深度解说高能电影《Generic Movie》。"},
        {"segment_id": "narration-001", "source_start": 332.0, "source_end": 380.0, "duration": 48.0, "text": "女主在理发店修剪发丝，理发师正在给顾客洗头设计发型。"},
    ]

    report = detector.evaluate_script(
        script=mismatched_script,
        scene_index=scene_index,
        edit_decisions=edit_decisions,
    )

    # AV coupling dimension must fail due to salon vs cooking conflict
    av_dim = next((d for d in report.dimensions if d.name == "Cross-Modal Audio-Visual Coupling"), None)
    assert av_dim is not None
    assert av_dim.passed is False

    # The entire report MUST fail and receive TIER_F, NOT TIER_S
    assert report.passed is False
    assert report.grade == EvalGrade.TIER_F
    assert "FAILED" in report.summary_verdict


def test_duplicate_sentence_detection_fails_continuity():
    """Verify that repeating duplicate sentences across segments causes continuity failure."""
    from src.eval.deterministic_eval import calculate_storyteller_continuity
    duplicate_script_segments = [
        {
            "segment_id": "seg-00",
            "text": "随着时间推移，暗处的危机已然扩散至全新的场景。突发事件的连环冲击使得原本脆弱的平衡分崩离析，人物被动卷入未知的漩涡。",
            "supporting_scenes": [{"start_seconds": 10.0, "end_seconds": 60.0}],
        },
        {
            "segment_id": "seg-01",
            "text": "未等众人喘息，局势在周围悄然加剧。突发事件的连环冲击使得原本脆弱的平衡分崩离析，人物被动卷入未知的漩涡。",
            "supporting_scenes": [{"start_seconds": 100.0, "end_seconds": 150.0}],
        },
    ]

    metrics, dim = calculate_storyteller_continuity(duplicate_script_segments)
    assert dim.passed is False
    assert any("Repetitive duplicate sentence" in ev for ev in metrics.discontinuity_events)


def test_hospital_medical_av_conflict():
    """Verify that narrating combat/burning dolls over a hospital exam scene fails AV coupling."""
    detector = SlopDetector()
    script = {
        "title": "Medical Test Movie",
        "estimated_duration_minutes": 2.0,
        "segments": [
            {
                "segment_id": "narration-000",
                "text": "生死一线之际，手指终于触碰到了黑色母偶！烈焰腾空而起，将母偶连同滔天的怨念一同卷入熊熊火海，厉鬼惨死在烈焰中。",
                "supporting_scenes": [{"start_seconds": 100.0, "end_seconds": 160.0}],
            },
        ],
    }
    scene_index = {
        "scenes": [
            {"scene_id": "s1", "start_seconds": 100.0, "end_seconds": 160.0, "transcript_text": "今回の検査では原因まで特定できませんでした。もっと大きな病院で詳しい検査が必要です。"},
        ]
    }
    edit_decisions = [
        {"segment_id": "narration-000", "source_start": 100.0, "source_end": 160.0, "duration": 60.0, "text": script["segments"][0]["text"]},
    ]

    report = detector.evaluate_script(
        script=script,
        scene_index=scene_index,
        edit_decisions=edit_decisions,
    )

    av_dim = next((d for d in report.dimensions if d.name == "Cross-Modal Audio-Visual Coupling"), None)
    assert av_dim is not None
    assert av_dim.passed is False
    assert any("hospital_medical" in f for f in av_dim.findings)


def test_offline_eval_detects_slop_boilerplate():
    """Verify that offline evaluation detects AI slop boilerplate and fails the script."""
    detector = SlopDetector()
    slop_script = {
        "title": "Slop Test",
        "segments": [
            {"segment_id": "s0", "text": "在这个充满悬疑的世界中，不得不说，男主做出了一个选择。究竟会发生什么呢？让我们拭目以待。突发事件的连环冲击使得原本脆弱的平衡分崩离析。"},
            {"segment_id": "s1", "text": "突发事件的连环冲击使得原本脆弱的平衡分崩离析。命运的齿轮已然无可逆转地开始转动。"},
            {"segment_id": "s2", "text": "突发事件的连环冲击使得原本脆弱的平衡分崩离析。总的来说这是一部悬疑电影。"},
        ],
    }
    report = detector.evaluate_script(script=slop_script)
    assert report.passed is False
    assert report.grade == EvalGrade.TIER_F
    assert report.semantic.commentary_depth_score < 60.0





