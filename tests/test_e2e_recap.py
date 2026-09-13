"""
Unit & Integration tests for End-to-End Recap Flow and 1/5 Proportional Duration Policy.
"""

from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.media.ingest import ingest_movie
from src.orchestrator.project import ProjectManager
from src.orchestrator.states import ProjectState


def test_proportional_duration_calculation(tmp_path: Path) -> None:
    pm = ProjectManager(projects_dir=tmp_path / "projects")
    source_file = tmp_path / "dummy_movie.mp4"
    source_file.write_bytes(b"data")

    # Test 1: 10-minute movie (600s) -> 1/5 duration = 2.0 min, range ~[1.7, 2.3]
    proj1 = pm.create_project(
        source_path=source_file,
        source_hash="hash1",
        title="10min Movie",
        media_info={"duration_seconds": 600.0},
        duration_ratio=0.20,
    )
    dur_range = proj1["target_duration_range"]
    assert dur_range[0] == pytest.approx(1.70, 0.05)
    assert dur_range[1] == pytest.approx(2.30, 0.05)

    # Test 2: 100-minute movie (6000s) -> 1/5 duration = 20.0 min, range ~[17.0, 23.0]
    proj2 = pm.create_project(
        source_path=source_file,
        source_hash="hash2",
        title="100min Movie",
        media_info={"duration_seconds": 6000.0},
        duration_ratio=0.20,
    )
    dur_range2 = proj2["target_duration_range"]
    assert dur_range2[0] == pytest.approx(17.0, 0.5)
    assert dur_range2[1] == pytest.approx(23.0, 0.5)

    # Test 3: User specifies explicit target duration (e.g. 5.0 min)
    proj3 = pm.create_project(
        source_path=source_file,
        source_hash="hash3",
        title="Custom Duration Movie",
        media_info={"duration_seconds": 6000.0},
        target_duration_range=[4.5, 5.5],
    )
    assert proj3["target_duration_range"] == [4.5, 5.5]
