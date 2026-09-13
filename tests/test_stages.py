from __future__ import annotations
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.stages.analyze import Stage2Result, run_stage_analyze
from src.stages.download import Stage1Result, run_stage_download
from src.stages.generate import Stage3Result, run_stage_generate
from src.stages.pipeline import run_staged_pipeline
from src.models.project import MediaInfo


@patch("src.acquirer.agent.extract_media_info")
def test_stage1_download_and_checkpoint(mock_extract_info, tmp_path: Path):
    mock_extract_info.return_value = MediaInfo(
        duration_seconds=180.0,
        resolution=(1920, 1080),
        frame_rate=24.0,
        video_codec="h264",
        audio_codec="aac",
        audio_tracks=2,
        has_subtitles=False,
    )

    # Create dummy video
    source_file = tmp_path / "incoming" / "TestMovie" / "source.mp4"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_bytes(b"mock video data")

    meta_file = source_file.parent / "metadata.json"
    meta_file.write_text(json.dumps({"title": "Test Movie"}), encoding="utf-8")

    res = run_stage_download(
        url=str(source_file.parent),
        output_dir=tmp_path / "stages" / "1_download",
    )

    assert res.slug == "TestMovie"
    assert res.movie_title == "Test Movie"
    assert res.source_video_path.exists()
    assert (res.stage_dir / "stage1_checkpoint.json").exists()


@patch("src.stages.analyze.detect_scenes")
@patch("src.stages.analyze.extract_scene_keyframes")
@patch("src.stages.analyze.extract_audio")
@patch("src.stages.analyze.understand_story")
@patch("src.stages.analyze.extract_media_info")
def test_stage2_analyze_and_checkpoint(
    mock_media_info,
    mock_story,
    mock_audio,
    mock_keyframes,
    mock_scenes,
    tmp_path: Path,
):
    mock_media_info.return_value = MediaInfo(
        duration_seconds=300.0,
        resolution=(1920, 1080),
        frame_rate=24.0,
        video_codec="h264",
        audio_codec="aac",
        audio_tracks=2,
        has_subtitles=False,
    )
    mock_scenes.return_value = [(0.0, 30.0), (30.0, 90.0), (90.0, 150.0)]

    mock_story.return_value = {
        "movie_title": "Mock Film",
        "characters": [{"name": "主角A", "aliases": ["小A"], "importance": "main"}],
        "events": [],
        "themes": ["Courage"],
        "tone": "Dramatic",
    }

    # Prepare mock stage 1 directory
    stage1_dir = tmp_path / "1_download" / "mock_slug"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    (stage1_dir / "source.mp4").write_bytes(b"dummy video data")
    (stage1_dir / "metadata.json").write_text(json.dumps({"title": "Mock Film"}), encoding="utf-8")

    # Run Stage 2
    res2 = run_stage_analyze(
        stage1_input=stage1_dir,
        output_dir=tmp_path / "2_analyzed",
        force=True,
    )

    assert res2.slug == "mock_slug"
    assert res2.movie_title == "Mock Film"
    assert res2.scene_count == 3
    assert res2.scene_index_path.exists()
    assert res2.story_understanding_path.exists()
    assert (res2.stage_dir / "stage2_checkpoint.json").exists()
    assert res2.was_cached is False

    # Second run without force should hit checkpoint and skip
    res2_cached = run_stage_analyze(
        stage1_input=stage1_dir,
        output_dir=tmp_path / "2_analyzed",
        force=False,
    )
    assert res2_cached.was_cached is True
    assert res2_cached.scene_count == 3


@patch("src.stages.generate.generate_script")
@patch("src.stages.generate.generate_voice_assets")
@patch("src.stages.generate.plan_and_extract_clips")
@patch("src.stages.generate.render_recap_video")
@patch("src.stages.generate.SlopDetector")
@patch("src.stages.generate.PlatformExporter")
def test_stage3_generate_multiple_cuts(
    mock_platform_exporter_cls,
    mock_slop_detector_cls,
    mock_render,
    mock_plan_clips,
    mock_tts,
    mock_generate_script,
    tmp_path: Path,
):
    # Setup mock stage 2 directory
    stage2_dir = tmp_path / "2_analyzed" / "my_movie"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    (stage2_dir / "source.mp4").write_bytes(b"dummy video content")

    story_data = {
        "movie_title": "My Movie",
        "characters": [{"name": "Hero"}],
        "events": [],
    }
    (stage2_dir / "story_understanding.json").write_text(json.dumps(story_data), encoding="utf-8")

    scene_data = {
        "movie_title": "My Movie",
        "duration_seconds": 600.0,
        "scenes": [{"scene_id": "scene-001", "start_seconds": 0.0, "end_seconds": 30.0}],
    }
    (stage2_dir / "scene_index.json").write_text(json.dumps(scene_data), encoding="utf-8")

    mock_generate_script.return_value = {
        "title": "My Movie",
        "segments": [{"segment_id": "seg-1", "segment_type": "hook", "text": "精彩开场"}],
    }

    mock_voice = MagicMock()
    mock_voice.model_dump.return_value = {"segment_id": "seg-1", "duration_seconds": 5.0}
    mock_tts.return_value = [mock_voice]

    mock_clip = MagicMock()
    mock_clip.model_dump.return_value = {"clip_id": "clip-1", "duration": 5.0}
    mock_plan_clips.return_value = [mock_clip]

    mock_render_res = MagicMock()
    mock_render_res.recap_video_path = stage2_dir / "recap_final.mp4"
    mock_render_res.subtitles_path = stage2_dir / "subtitles.srt"
    mock_render_res.audio_duration_seconds = 5.0
    mock_render.return_value = mock_render_res

    mock_qa_report = MagicMock()
    mock_qa_report.grade.value = "A"
    mock_qa_report.total_score = 88.5
    mock_detector_instance = MagicMock()
    mock_detector_instance.evaluate_script.return_value = mock_qa_report
    mock_slop_detector_cls.return_value = mock_detector_instance

    mock_exporter_instance = MagicMock()
    mock_exporter_instance.export_package.return_value = {
        "youtube_video": stage2_dir / "recap_youtube.mp4",
        "metadata_youtube": stage2_dir / "metadata_youtube.json",
    }
    mock_platform_exporter_cls.return_value = mock_exporter_instance

    # Generate 2-minute recap cut
    res_cut1 = run_stage_generate(
        stage2_input=stage2_dir,
        output_dir=tmp_path / "output_1",
        stage3_dir=tmp_path / "stage3_1",
        target_duration=2.0,
    )
    assert res_cut1.target_recap_minutes == 2.0
    assert res_cut1.quality_grade == "A"

    # Generate 5-minute recap cut (without touching Stage 1 or Stage 2)
    res_cut2 = run_stage_generate(
        stage2_input=stage2_dir,
        output_dir=tmp_path / "output_2",
        stage3_dir=tmp_path / "stage3_2",
        target_duration=5.0,
    )
    assert res_cut2.target_recap_minutes == 5.0
    assert res_cut2.quality_grade == "A"


@patch("src.stages.pipeline.run_stage_analyze")
@patch("src.stages.pipeline.run_stage_generate")
def test_pipeline_slices(mock_generate, mock_analyze, tmp_path: Path):
    mock_res2 = MagicMock()
    mock_res2.slug = "my_film"
    mock_res2.to_dict.return_value = {"slug": "my_film", "scene_count": 5}
    mock_analyze.return_value = mock_res2

    mock_res3 = MagicMock()
    mock_res3.to_dict.return_value = {"slug": "my_film", "quality_grade": "S"}
    mock_generate.return_value = mock_res3

    # Run from Stage 2 to Stage 3 directly
    results = run_staged_pipeline(
        input_target="my_film",
        from_stage=2,
        to_stage=3,
        target_duration=3.0,
    )

    assert "stage1" not in results
    assert "stage2" in results
    assert "stage3" in results
    assert mock_analyze.call_count == 1
    assert mock_generate.call_count == 1

