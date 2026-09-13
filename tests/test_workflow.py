from __future__ import annotations
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.orchestrator.workflow import WorkflowRunner
from src.orchestrator.states import ProjectState

@patch("src.media.ingest.ingest_movie")
def test_workflow_process_incoming(mock_ingest, tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    projects = tmp_path / "projects"
    projects.mkdir()
    
    movie_folder = incoming / "movie_1"
    movie_folder.mkdir()
    (movie_folder / "source.mp4").write_bytes(b"data")
    
    mock_ingest.return_value = {
        "project_id": "proj-1",
        "title": "movie_1",
        "state": ProjectState.RECEIVED.value
    }
    
    runner = WorkflowRunner(projects_dir=projects, incoming_dir=incoming)
    new_projects = runner.process_incoming()
    
    assert len(new_projects) == 1
    assert new_projects[0]["project_id"] == "proj-1"
    assert (movie_folder / ".ingested").exists()
    
    # Second run should skip already ingested
    new_projects_second = runner.process_incoming()
    assert len(new_projects_second) == 0

@patch("src.media.probe.extract_media_info")
@patch("src.media.audio.extract_audio")
@patch("src.media.scene_detect.detect_scenes")
@patch("src.media.frames.extract_scene_keyframes")
@patch("src.ai.story.understand_story")
@patch("src.ai.script.generate_script")
@patch("src.ai.verify.verify_script")
def test_workflow_run_project_milestone1(
    mock_verify, mock_script, mock_story, mock_keyframes, mock_detect, mock_audio, mock_probe, tmp_path
):
    projects = tmp_path / "projects"
    incoming = tmp_path / "incoming"
    
    runner = WorkflowRunner(projects_dir=projects, incoming_dir=incoming)
    
    source_file = tmp_path / "test.mp4"
    source_file.write_bytes(b"dummy")
    
    project = runner.project_manager.create_project(
        source_path=source_file,
        source_hash="sha256_mock",
        title="Spike Movie"
    )
    
    mock_probe.return_value = {
        "duration_seconds": 100.0,
        "resolution": (1920, 1080),
        "frame_rate": 24.0,
        "video_codec": "h264",
        "audio_codec": "aac",
        "audio_tracks": 1,
        "has_subtitles": False
    }
    mock_detect.return_value = [(0.0, 10.0), (10.0, 20.0)]
    mock_story.return_value = {
        "title": "Spike Movie",
        "events": [],
        "characters": [],
        "telemetry": {}
    }
    mock_script.return_value = {
        "project_id": project["project_id"],
        "segments": [{"segment_id": "seg-1", "text": "开场白", "segment_type": "hook"}],
        "telemetry": {}
    }
    mock_verify.return_value = {
        "status": "pass",
        "fails": 0,
        "warnings": 0,
        "findings": []
    }
    
    # Run project through Milestone 1 pipeline
    result = runner.run_project(project["project_id"])
    
    # Should progress through INGESTING -> ANALYZING -> SCRIPTING -> GENERATING_AUDIO -> etc.
    assert result["state"] in [ProjectState.AWAITING_APPROVAL.value, ProjectState.GENERATING_AUDIO.value, ProjectState.APPROVED.value, ProjectState.YOUTUBE_PRIVATE.value]
