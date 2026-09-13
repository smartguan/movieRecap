"""Tests for SceneIndex."""
from __future__ import annotations

from src.index.scene_index import SceneIndex, SceneEntry


def _make_index():
    """Create a test scene index."""
    scenes = [
        SceneEntry(
            scene_id="scene-0001",
            start_seconds=0.0,
            end_seconds=30.0,
            duration_seconds=30.0,
            transcript_text="Hello, welcome to the story.",
            characters=["Alice", "Bob"],
        ),
        SceneEntry(
            scene_id="scene-0002",
            start_seconds=30.0,
            end_seconds=60.0,
            duration_seconds=30.0,
            transcript_text="The conflict begins here.",
            characters=["Alice"],
        ),
        SceneEntry(
            scene_id="scene-0003",
            start_seconds=60.0,
            end_seconds=90.0,
            duration_seconds=30.0,
            transcript_text="Resolution and ending.",
            characters=["Bob", "Charlie"],
            sensitivity_labels=["mild_violence"],
        ),
    ]
    return SceneIndex(project_id="test-project", scenes=scenes)


def test_search_transcript():
    """Test searching scenes by transcript text."""
    index = _make_index()
    results = index.search_transcript("conflict")
    assert len(results) == 1
    assert results[0].scene_id == "scene-0002"


def test_search_no_results():
    """Test searching with no matches."""
    index = _make_index()
    results = index.search_transcript("nonexistent")
    assert len(results) == 0


def test_get_scenes_in_range():
    """Test getting scenes that overlap with a time range."""
    index = _make_index()
    results = index.get_scenes_in_range(25.0, 55.0)
    assert len(results) == 2
    scene_ids = {s.scene_id for s in results}
    assert "scene-0001" in scene_ids
    assert "scene-0002" in scene_ids

    # Range covering all 3
    all_results = index.get_scenes_in_range(25.0, 65.0)
    assert len(all_results) == 3


def test_get_scene_by_id():
    """Test getting a scene by ID."""
    index = _make_index()
    scene = index.get_scene_by_id("scene-0002")
    assert scene is not None
    assert scene.transcript_text == "The conflict begins here."

    assert index.get_scene_by_id("nonexistent") is None


def test_get_scenes_by_character():
    """Test filtering scenes by character name."""
    index = _make_index()
    results = index.get_scenes_by_character("Alice")
    assert len(results) == 2

    results = index.get_scenes_by_character("Charlie")
    assert len(results) == 1


def test_get_sensitive_scenes():
    """Test getting scenes with sensitivity labels."""
    index = _make_index()
    results = index.get_sensitive_scenes()
    assert len(results) == 1
    assert results[0].scene_id == "scene-0003"


def test_save_load_roundtrip(tmp_path):
    """Test saving and loading the scene index."""
    index = _make_index()
    path = tmp_path / "scene_index.json"
    index.save(path)

    loaded = SceneIndex.load(path)
    assert loaded.project_id == "test-project"
    assert len(loaded.scenes) == 3
    assert loaded.scenes[0].scene_id == "scene-0001"
    assert loaded.scenes[2].sensitivity_labels == ["mild_violence"]
