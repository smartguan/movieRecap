from __future__ import annotations

import json
import pytest
from pathlib import Path

from src.orchestrator.project import ProjectManager
from src.orchestrator.states import ProjectState, InvalidTransitionError


def test_create_project(tmp_path):
    """Test that create_project creates directory structure and project.json."""
    pm = ProjectManager(tmp_path)
    project = pm.create_project(
        source_path=Path("/fake/source.mp4"),
        source_hash="abc123",
        title="Test Movie",
    )
    assert project["project_id"]
    assert project["state"] == ProjectState.RECEIVED.value
    assert project["title"] == "Test Movie"
    assert project["source_hash"] == "abc123"

    project_dir = tmp_path / project["project_id"]
    assert project_dir.exists()
    assert (project_dir / "project.json").exists()
    assert (project_dir / "assets").is_dir()
    assert (project_dir / "keyframes").is_dir()


def test_transition_updates_state(tmp_path):
    """Test that transition updates state and appends history."""
    pm = ProjectManager(tmp_path)
    project = pm.create_project(
        source_path=Path("/fake/source.mp4"),
        source_hash="abc123",
        title="Test Movie",
    )
    project = pm.transition(project, ProjectState.INGESTING, details="Starting")
    assert project["state"] == ProjectState.INGESTING.value
    assert len(project["state_history"]) == 2
    assert project["state_history"][-1]["state"] == "ingesting"


def test_invalid_transition_raises(tmp_path):
    """Test that invalid transitions raise InvalidTransitionError."""
    pm = ProjectManager(tmp_path)
    project = pm.create_project(
        source_path=Path("/fake/source.mp4"),
        source_hash="abc123",
        title="Test Movie",
    )
    # RECEIVED cannot transition directly to PUBLISHED
    with pytest.raises(InvalidTransitionError):
        pm.transition(project, ProjectState.PUBLISHED)


def test_save_load_roundtrip(tmp_path):
    """Test that save/load preserves project data."""
    pm = ProjectManager(tmp_path)
    project = pm.create_project(
        source_path=Path("/fake/source.mp4"),
        source_hash="abc123",
        title="Test Movie",
        metadata={"release_year": 2025},
    )
    loaded = pm.load_project(project["project_id"])
    assert loaded["project_id"] == project["project_id"]
    assert loaded["title"] == project["title"]
    assert loaded["source_hash"] == project["source_hash"]


def test_list_projects(tmp_path):
    """Test listing projects with and without state filter."""
    pm = ProjectManager(tmp_path)
    pm.create_project(Path("/a.mp4"), "hash1", "Movie A")
    pm.create_project(Path("/b.mp4"), "hash2", "Movie B")

    all_projects = pm.list_projects()
    assert len(all_projects) == 2

    received = pm.list_projects(state_filter=ProjectState.RECEIVED)
    assert len(received) == 2

    ingesting = pm.list_projects(state_filter=ProjectState.INGESTING)
    assert len(ingesting) == 0


def test_add_cost(tmp_path):
    """Test that add_cost accumulates cost."""
    pm = ProjectManager(tmp_path)
    project = pm.create_project(Path("/a.mp4"), "hash1", "Movie A")
    project = pm.add_cost(project, 1.50)
    assert project["cost_usd"] == 1.50
    project = pm.add_cost(project, 0.75)
    assert project["cost_usd"] == 2.25


def test_load_nonexistent_raises(tmp_path):
    """Test that loading a nonexistent project raises FileNotFoundError."""
    pm = ProjectManager(tmp_path)
    with pytest.raises(FileNotFoundError):
        pm.load_project("nonexistent-id")
