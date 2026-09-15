"""
Tests for V3 Video-First Narrative-Spine Recap Pipeline and A/B Swappability.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.ai.story_filter import filter_main_story_sequences, score_sequence_importance
from src.ai.video_grounded_script import VideoGroundedScriptSynthesizer
from src.eval.deterministic_eval import calculate_storyteller_continuity, evaluate_deterministic
from src.media.sequence_clusterer import NarrativeSequence, cluster_scenes_into_sequences


def test_sequence_clusterer_continuous_grouping():
    """Verify that cluster_scenes_into_sequences merges shots into contiguous macro-scenes."""
    scenes = [
        {"scene_id": "s-001", "start_seconds": 0.0, "end_seconds": 15.0, "transcript_text": "おはよう", "characters": ["女主"]},
        {"scene_id": "s-002", "start_seconds": 15.0, "end_seconds": 35.0, "transcript_text": "カットをお願いします", "characters": ["女主"]},
        {"scene_id": "s-003", "start_seconds": 35.0, "end_seconds": 55.0, "transcript_text": "髪型変えるの？", "characters": ["女主", "同伴"]},
        {"scene_id": "s-004", "start_seconds": 55.0, "end_seconds": 110.0, "transcript_text": "そうなんだ", "characters": ["同伴"]},
    ]
    subtitles = [
        {"start": 10.0, "end": 14.0, "text": "おはよう"},
        {"start": 16.0, "end": 20.0, "text": "カットをお願いします"},
        {"start": 36.0, "end": 40.0, "text": "髪型変えるの？"},
    ]

    sequences = cluster_scenes_into_sequences(
        scenes=scenes,
        subtitles=subtitles,
        min_sequence_duration=40.0,
        max_sequence_duration=120.0,
    )

    assert len(sequences) >= 1
    for seq in sequences:
        assert seq.duration_seconds >= 40.0 or seq == sequences[-1]
        assert seq.end_seconds > seq.start_seconds
        assert isinstance(seq.characters, list)


def test_story_filter_budget_and_phases():
    """Verify that story_filter picks balanced dramatic scenes hitting target duration."""
    sequences = [
        NarrativeSequence(
            sequence_id=f"seq-{i:03d}",
            sequence_index=i,
            start_seconds=float(i * 100),
            end_seconds=float(i * 100 + 60),
            duration_seconds=60.0,
            transcript_text="投稿された動画に変な画像が写っている呪い死ぬ",
            characters=["女主", "同伴"],
            dialogue_count=4,
        )
        for i in range(20)
    ]
    story_understanding = {
        "movie_title": "诅咒",
        "characters": [{"name": "女主"}, {"name": "同伴"}],
        "events": [
            {"timestamp_seconds": 100.0, "event_type": "setup"},
            {"timestamp_seconds": 500.0, "event_type": "inciting_incident"},
            {"timestamp_seconds": 1000.0, "event_type": "investigation"},
            {"timestamp_seconds": 1600.0, "event_type": "climax"},
            {"timestamp_seconds": 1900.0, "event_type": "resolution"},
        ],
    }

    # Request ~300 seconds budget (5 sequences of 60s)
    selected = filter_main_story_sequences(
        sequences=sequences,
        story_understanding=story_understanding,
        target_duration_sec=300.0,
    )

    assert len(selected) >= 3
    total_dur = sum(s.duration_seconds for s in selected)
    assert 200.0 <= total_dur <= 450.0

    # Verify chronological ordering
    for j in range(1, len(selected)):
        assert selected[j].start_seconds >= selected[j - 1].start_seconds


def test_video_grounded_script_synthesis_and_continuity():
    """Verify that VideoGroundedScriptSynthesizer produces commentary with transitions and continuity."""
    sequences = [
        NarrativeSequence(
            sequence_id="seq-000",
            sequence_index=0,
            start_seconds=10.0,
            end_seconds=70.0,
            duration_seconds=60.0,
            transcript_text="カットをお願いします。髪型を変えたいです。",
            characters=["女主"],
            event_type="setup",
        ),
        NarrativeSequence(
            sequence_id="seq-001",
            sequence_index=1,
            start_seconds=250.0,
            end_seconds=310.0,
            duration_seconds=60.0,
            transcript_text="投稿を見た？死んだはずのあの子のアカウントから変な画像が。",
            characters=["女主", "同伴"],
            event_type="inciting_incident",
        ),
        NarrativeSequence(
            sequence_id="seq-002",
            sequence_index=2,
            start_seconds=1500.0,
            end_seconds=1560.0,
            duration_seconds=60.0,
            transcript_text="台湾に行くしかない。台北の店で呪いを解く方法を聞こう。",
            characters=["女主"],
            event_type="investigation",
        ),
        NarrativeSequence(
            sequence_id="seq-003",
            sequence_index=3,
            start_seconds=3200.0,
            end_seconds=3260.0,
            duration_seconds=60.0,
            transcript_text="神廟の奥に母偶がある。早く火をつけろ！",
            characters=["女主", "同伴"],
            event_type="climax",
        ),
    ]

    synthesizer = VideoGroundedScriptSynthesizer()
    segments = synthesizer.synthesize_script(
        title="诅咒",
        synopsis="女主调查好友离奇死亡诅咒的悬疑恐怖故事",
        cast=["女主", "同伴"],
        genre="惊悚",
        selected_sequences=sequences,
    )

    assert len(segments) == len(sequences)
    # Check that each segment has matching supporting_scenes
    for idx, seg in enumerate(segments):
        assert seg["supporting_scenes"][0]["start_seconds"] == sequences[idx].start_seconds
        assert seg["supporting_scenes"][0]["end_seconds"] == sequences[idx].end_seconds
        assert len(seg["text"]) >= 40

    # Evaluate continuity of the V3 synthesized script
    story = {"characters": [{"name": "女主"}, {"name": "同伴"}]}
    cont_metrics, cont_dim = calculate_storyteller_continuity(
        segments=segments,
        story=story,
        total_source_duration=3600.0,
    )

    assert cont_metrics.temporal_monotonicity_score == 100.0
    assert cont_metrics.backward_jump_count == 0
    assert cont_metrics.continuity_score >= 80.0
    assert cont_dim.passed is True


def test_video_grounded_script_zero_meta_commentary():
    """Verify that synthesized commentary contains zero ungrounded meta-commentary or fourth-wall breaks."""
    from src.eval.cliches import scan_for_meta_commentary

    sequences = [
        NarrativeSequence(
            sequence_id=f"seq-{i}",
            sequence_index=i,
            start_seconds=float(i * 300),
            end_seconds=float(i * 300 + 60),
            duration_seconds=60.0,
            transcript_text=f"调查证据线索第{i}幕",
            characters=["女主"],
            event_type="setup" if i == 0 else ("climax" if i == 4 else "investigation"),
        )
        for i in range(5)
    ]

    synthesizer = VideoGroundedScriptSynthesizer()
    segments = synthesizer.synthesize_script(
        title="诅咒",
        synopsis="女主调查好友离奇死亡诅咒的悬疑恐怖故事",
        cast=["女主", "同伴"],
        genre="惊悚",
        selected_sequences=sequences,
    )

    full_text = "".join(seg["text"] for seg in segments)
    meta_matches = scan_for_meta_commentary(full_text)
    assert meta_matches == [], f"Found ungrounded meta-commentary: {meta_matches}"


@patch("src.stages.generate.render_recap_video")
@patch("src.stages.generate.generate_voice_assets")
@patch("src.stages.generate.plan_and_extract_clips")
@patch("src.stages.generate.SlopDetector")
@patch("src.stages.generate.PlatformExporter")
@patch("src.stages.generate.generate_script")
def test_stage3_ab_swapping_v2_vs_v3(
    mock_v2_script,
    mock_platform_exporter_cls,
    mock_slop_detector_cls,
    mock_plan_clips,
    mock_tts,
    mock_render,
    tmp_path: Path,
):
    """Verify that Stage 3 supports clean A/B swappability between v2 and v3 pipelines."""
    import json
    from src.stages.generate import run_stage_generate

    stage2_dir = tmp_path / "2_analyzed" / "test_movie"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    (stage2_dir / "source.mp4").write_bytes(b"dummy video")

    story_data = {
        "movie_title": "Test Movie",
        "characters": [{"name": "主角"}],
        "events": [],
    }
    (stage2_dir / "story_understanding.json").write_text(json.dumps(story_data), encoding="utf-8")

    scene_data = {
        "movie_title": "Test Movie",
        "duration_seconds": 600.0,
        "scenes": [
            {"scene_id": "s-01", "start_seconds": 0.0, "end_seconds": 60.0, "transcript_text": "hello"},
            {"scene_id": "s-02", "start_seconds": 60.0, "end_seconds": 120.0, "transcript_text": "world"},
        ],
    }
    (stage2_dir / "scene_index.json").write_text(json.dumps(scene_data), encoding="utf-8")
    (stage2_dir / "subtitles.json").write_text(json.dumps([]), encoding="utf-8")

    mock_v2_script.return_value = {
        "title": "Test Movie",
        "version": "v2",
        "segments": [{"segment_id": "seg-1", "segment_type": "hook", "text": "V2 script"}],
    }

    mock_voice = MagicMock()
    mock_voice.model_dump.return_value = {"segment_id": "seg-1", "duration_seconds": 5.0}
    mock_tts.return_value = [mock_voice]

    mock_clip = MagicMock()
    mock_clip.model_dump.return_value = {"clip_id": "c-1", "duration": 5.0}
    mock_plan_clips.return_value = [mock_clip]

    mock_render_res = MagicMock()
    mock_render_res.recap_video_path = stage2_dir / "recap.mp4"
    mock_render_res.subtitles_path = stage2_dir / "subtitles.srt"
    mock_render_res.audio_duration_seconds = 5.0
    mock_render.return_value = mock_render_res

    mock_report = MagicMock()
    mock_report.grade.value = "S"
    mock_report.total_score = 95.0
    mock_detector = MagicMock()
    mock_detector.evaluate_script.return_value = mock_report
    mock_slop_detector_cls.return_value = mock_detector

    mock_exporter = MagicMock()
    mock_exporter.export_package.return_value = {}
    mock_platform_exporter_cls.return_value = mock_exporter

    # 1. Run V2
    res_v2 = run_stage_generate(
        stage2_input=stage2_dir,
        stage3_dir=tmp_path / "stage3_v2",
        output_dir=tmp_path / "output_v2",
        force=True,
        algo_version="v2",
    )
    assert mock_v2_script.called

    # 2. Run V3
    mock_v2_script.reset_mock()
    res_v3 = run_stage_generate(
        stage2_input=stage2_dir,
        stage3_dir=tmp_path / "stage3_v3",
        output_dir=tmp_path / "output_v3",
        force=True,
        algo_version="v3",
    )
    # In V3, mock_v2_script must NOT be called
    assert not mock_v2_script.called
    v3_script = json.loads((tmp_path / "stage3_v3" / "script.json").read_text(encoding="utf-8"))
    assert v3_script.get("version") == "v3"

