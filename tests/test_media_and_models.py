from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.media.probe import probe_media, extract_media_info
from src.media.audio import extract_audio, extract_audio_segment
from src.media.frames import extract_keyframes, extract_scene_keyframes
from src.media.scene_detect import detect_scenes, detect_scenes_adaptive
from src.media.ingest import validate_incoming, ingest_movie, compute_file_hash
from src.models.project import Project, ProjectState as ModelProjectState, MediaInfo
from src.models.scene import Scene, SceneIndex as ModelSceneIndex
from src.models.story import Character, CharacterRelationship, StoryEvent, StoryUnderstanding
from src.models.script import ScriptSegment, SceneReference, Script
from src.models.invocation import ModelInvocation, InvocationLedger
from src.utils.validation import validate_project_dir, validate_media_file, validate_json_schema

def test_models_roundtrip():
    proj = Project(
        project_id="p-123",
        source_path=Path("/tmp/test.mp4"),
        source_hash="sha256_hash",
        title="Test Movie",
        state=ModelProjectState.RECEIVED
    )
    assert proj.title == "Test Movie"
    assert proj.state == ModelProjectState.RECEIVED

    scene = Scene(
        scene_id="s-1",
        start_seconds=0.0,
        end_seconds=10.0,
        characters=["Protagonist"]
    )
    assert scene.duration_seconds == 10.0

    script_seg = ScriptSegment(
        segment_id="seg-1",
        text="开场白叙述",
        supporting_scenes=[SceneReference(start_seconds=0.0, end_seconds=10.0)],
        confidence=0.95,
        segment_type="hook"
    )
    script = Script(
        project_id="p-123",
        segments=[script_seg]
    )
    assert script.total_characters == 5
    assert script.estimated_duration_minutes > 0

def test_validation_utils(tmp_path):
    valid_file = tmp_path / "test.mp4"
    valid_file.write_bytes(b"dummy video content")
    
    is_valid, errors = validate_media_file(valid_file)
    assert is_valid is True
    assert len(errors) == 0

    non_existent = tmp_path / "missing.mp4"
    is_valid, errors = validate_media_file(non_existent)
    assert is_valid is False
    assert len(errors) > 0

def test_validate_incoming(tmp_path):
    incoming = tmp_path / "incoming_movie"
    incoming.mkdir()
    
    # Missing source file
    valid, errors = validate_incoming(incoming)
    assert valid is False

    # Add source file
    (incoming / "source.mp4").write_bytes(b"data")
    valid, errors = validate_incoming(incoming)
    assert valid is True

@patch("subprocess.run")
def test_probe_media(mock_run, tmp_path):
    fake_file = tmp_path / "source.mp4"
    fake_file.write_bytes(b"dummy")
    
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout='{"format": {"duration": "120.5", "format_name": "mov,mp4"}, "streams": [{"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080, "r_frame_rate": "24/1"}, {"codec_type": "audio", "codec_name": "aac"}]}',
        stderr=""
    )
    
    info = probe_media(fake_file)
    assert "format" in info
    assert "streams" in info

    media_info = extract_media_info(fake_file)
    assert media_info.duration_seconds == 120.5
    assert media_info.resolution == (1920, 1080)
    assert media_info.video_codec == "h264"
