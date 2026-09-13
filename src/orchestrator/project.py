"""
Project lifecycle management.

Handles creation, state transitions, persistence, and loading of projects.
All operations are deterministic — no LLM calls.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
        media_info: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new project from an incoming source.

        Args:
            source_path: Path to the source media file.
            source_hash: SHA-256 hash of the source file.
            title: Movie title.
            media_info: Extracted media information.
            metadata: Optional user-supplied metadata.

        Returns:
            Project data dict with all fields initialized.
        """
        project_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        project = {
            "project_id": project_id,
            "source_path": str(source_path),
            "source_hash": source_hash,
            "title": title,
            "release_year": metadata.get("release_year") if metadata else None,
            "source_language": (metadata or {}).get("source_language", "ko"),
            "target_language": (metadata or {}).get("target_language", "zh-CN"),
            "state": ProjectState.RECEIVED.value,
            "target_duration_range": [20, 30],
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
                    "details": "Project created",
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
        """
        Transition a project to a new state.

        Args:
            project: Project data dict.
            to_state: Target state.
            details: Optional details about the transition.

        Returns:
            Updated project dict.

        Raises:
            InvalidTransitionError: If the transition is not valid.
        """
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
        """
        Save project state to disk as JSON.

        Args:
            project: Project data dict.

        Returns:
            Path to the saved project file.
        """
        project_dir = self.projects_dir / project["project_id"]
        project_dir.mkdir(parents=True, exist_ok=True)
        project_file = project_dir / "project.json"
        project_file.write_text(
            json.dumps(project, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return project_file

    def load_project(self, project_id: str) -> dict[str, Any]:
        """
        Load a project from disk.

        Args:
            project_id: UUID of the project.

        Returns:
            Project data dict.

        Raises:
            FileNotFoundError: If the project does not exist.
        """
        project_file = self.projects_dir / project_id / "project.json"
        if not project_file.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")
        return json.loads(project_file.read_text(encoding="utf-8"))

    def list_projects(
        self,
        state_filter: ProjectState | None = None,
    ) -> list[dict[str, Any]]:
        """
        List all projects, optionally filtered by state.

        Args:
            state_filter: If provided, only return projects in this state.

        Returns:
            List of project data dicts.
        """
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
        """
        Add cost to a project's running total.

        Args:
            project: Project data dict.
            cost_usd: Cost to add in USD.

        Returns:
            Updated project dict.
        """
        project["cost_usd"] = project.get("cost_usd", 0.0) + cost_usd
        project["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.save_project(project)
        return project
