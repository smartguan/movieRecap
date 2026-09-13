from __future__ import annotations

import json
import pytest
from pathlib import Path
from src.ai.profiler import TokenCostProfiler, TokenProfile, TaskCostMetrics


def test_profiler_with_mock_project(tmp_path):
    proj_dir = tmp_path / "proj-123"
    proj_dir.mkdir(parents=True)

    # 1. Mock project.json
    project_data = {
        "project_id": "proj-123",
        "title": "Profiling Test Movie",
        "media_info": {"duration_seconds": 600.0},
    }
    (proj_dir / "project.json").write_text(json.dumps(project_data), encoding="utf-8")

    # 2. Mock script.json
    script_data = {
        "project_id": "proj-123",
        "total_characters": 500,
        "telemetry": {
            "per_task": {
                "script_generation": {
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "cost_usd": 0.001,
                    "cache_hits": 1,
                }
            }
        },
    }
    (proj_dir / "script.json").write_text(json.dumps(script_data), encoding="utf-8")

    # 3. Mock story_understanding.json
    story_data = {
        "title": "Profiling Test Movie",
        "telemetry": {
            "per_task": {
                "story_synthesis": {
                    "input_tokens": 500,
                    "output_tokens": 100,
                    "cost_usd": 0.0005,
                    "cache_hits": 0,
                }
            }
        },
    }
    (proj_dir / "story_understanding.json").write_text(json.dumps(story_data), encoding="utf-8")

    profiler = TokenCostProfiler(projects_dir=tmp_path)
    profile = profiler.profile_project("proj-123")

    assert profile.project_id == "proj-123"
    assert profile.movie_title == "Profiling Test Movie"
    assert profile.source_duration_minutes == 10.0
    assert profile.total_input_tokens == 1500
    assert profile.total_output_tokens == 300
    assert profile.total_tokens == 1800
    assert profile.total_cost_usd > 0
    assert "script_generation" in profile.by_task
    assert "story_synthesis" in profile.by_task
    assert profile.tokens_per_source_minute == 180.0
    assert profile.tokens_per_script_char == 3.6

    # Test report export
    json_path, md_path = profiler.save_profile_report(profile, proj_dir)
    assert json_path.exists()
    assert md_path.exists()

    md_content = md_path.read_text(encoding="utf-8")
    assert "Cost Breakdown by Semantic Task" in md_content
    assert "script_generation" in md_content


def test_profiler_nonexistent_project(tmp_path):
    profiler = TokenCostProfiler(projects_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        profiler.profile_project("missing-id")
