"""
Project lifecycle management.

Handles creation, state transitions, persistence, and loading of projects.
All operations are deterministic — no LLM calls.
"""

from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from src.orchestrator.states import (
    InvalidTransitionError,
    ProjectState,
    validate_transition,
)


class ProjectManager:
    """Manages project lifecycle: creation, state transitions, and persistence."""

    def __init__(self, projects_dir: Path) -> None:
        """
        Initialize the project manager.

        Args:
            projects_dir: Root directory for all project data.
        """
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def create_project(
        self,
        source_path: Path,
        source_hash: str,
        title: str,
        media_info: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        target_duration_range: Optional[Union[List[float], Tuple[float, float]]] = None,
        duration_ratio: float = 0.20,
    ) -> dict[str, Any]:
        """
        Create a new project from an incoming source.

        Calculates dynamic 1/5 duration target range based on source media duration.

        Args:
            source_path: Path to the source media file.
            source_hash: SHA-256 hash of the source file.
            title: Movie title.
            media_info: Extracted media information.
            metadata: Optional user-supplied metadata.
            target_duration_range: Optional explicit target duration range in minutes [min, max].
            duration_ratio: Proportional duration ratio (default: 0.20 = 1/5).

        Returns:
            Project data dict with all fields initialized.
        """
        project_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Compute dynamic duration range
        if target_duration_range is not None:
            final_duration_range = list(target_duration_range)
        else:
            duration_sec = 0.0
            if media_info:
                duration_sec = media_info.get("duration_seconds", 0.0) or media_info.get("duration", 0.0)
            elif metadata:
                duration_sec = metadata.get("duration_seconds", 0.0)

            if duration_sec > 0:
                source_min = duration_sec / 60.0
                target_recap_min = max(0.5, round(source_min * duration_ratio, 2))
                final_duration_range = [
                    round(target_recap_min * 0.85, 2),
                    round(target_recap_min * 1.15, 2),
                ]
            else:
                final_duration_range = [20.0, 30.0]

        project = {
            "project_id": project_id,
            "source_path": str(source_path),
            "source_hash": source_hash,
            "title": title,
            "release_year": (metadata or {}).get("release_year") or (metadata or {}).get("year"),
            "source_language": (metadata or {}).get("source_language", "zh-CN"),
            "target_language": (metadata or {}).get("target_language", "zh-CN"),
            "state": ProjectState.RECEIVED.value,
            "target_duration_range": final_duration_range,
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {},
            "media_info": media_info,
            "error_info": None,
            "cost_usd": 0.0,
            "state_history": [
                {
                    "state": ProjectState.RECEIVED.value,
                    "timestamp": now,
                    "details": f"Project created with target duration {final_duration_range[0]}-{final_duration_range[1]} min (ratio: {duration_ratio:.2f})",
                }
            ],
        }

        # Create project directory
        project_dir = self.projects_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "assets").mkdir(exist_ok=True)
        (project_dir / "keyframes").mkdir(exist_ok=True)
        (project_dir / "audio").mkdir(exist_ok=True)
        (project_dir / "renders").mkdir(exist_ok=True)

        # Save project state
        self.save_project(project)

        return project

    def transition(
        self,
        project: dict[str, Any],
        to_state: ProjectState,
        details: str = "",
    ) -> dict[str, Any]:
        """Transition a project to a new state."""
        from_state = ProjectState(project["state"])
        validate_transition(from_state, to_state)

        now = datetime.now(timezone.utc).isoformat()
        project["state"] = to_state.value
        project["updated_at"] = now
        project["state_history"].append(
            {
                "state": to_state.value,
                "timestamp": now,
                "details": details,
            }
        )

        if to_state == ProjectState.FAILED:
            project["error_info"] = details

        self.save_project(project)
        return project

    def save_project(self, project: dict[str, Any]) -> Path:
        """Save project state to disk as JSON."""
        project_dir = self.projects_dir / project["project_id"]
        project_dir.mkdir(parents=True, exist_ok=True)
        project_file = project_dir / "project.json"
        project_file.write_text(
            json.dumps(project, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return project_file

    def load_project(self, project_id: str) -> dict[str, Any]:
        """Load a project from disk."""
        project_file = self.projects_dir / project_id / "project.json"
        if not project_file.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")
        return json.loads(project_file.read_text(encoding="utf-8"))

    def list_projects(
        self,
        state_filter: Optional[ProjectState] = None,
    ) -> list[dict[str, Any]]:
        """List all projects, optionally filtered by state."""
        projects = []
        for project_dir in self.projects_dir.iterdir():
            if not project_dir.is_dir():
                continue
            project_file = project_dir / "project.json"
            if not project_file.exists():
                continue
            try:
                project = json.loads(project_file.read_text(encoding="utf-8"))
                if state_filter is None or project.get("state") == state_filter.value:
                    projects.append(project)
            except (json.JSONDecodeError, KeyError):
                continue
        return projects

    def get_project_dir(self, project_id: str) -> Path:
        """Get the directory for a project."""
        return self.projects_dir / project_id

    def add_cost(self, project: dict[str, Any], cost_usd: float) -> dict[str, Any]:
        """Add cost to a project's running total."""
        project["cost_usd"] = project.get("cost_usd", 0.0) + cost_usd
        project["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save_project(project)
        return project
