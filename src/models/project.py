"""
Project data models.
"""
from __future__ import annotations
from enum import Enum
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, Field


class ProjectState(str, Enum):
    RECEIVED = "RECEIVED"
    INGESTING = "INGESTING"
    ANALYZING = "ANALYZING"
    SCRIPTING = "SCRIPTING"
    GENERATING_AUDIO = "GENERATING_AUDIO"
    PLANNING_EDIT = "PLANNING_EDIT"
    RENDERING = "RENDERING"
    AUTOMATED_QA = "AUTOMATED_QA"
    YOUTUBE_PRIVATE = "YOUTUBE_PRIVATE"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    SCHEDULED = "SCHEDULED"
    PUBLISHED = "PUBLISHED"
    RETRYING = "RETRYING"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class MediaInfo(BaseModel):
    """Information about the source media file."""
    duration_seconds: float
    resolution: Tuple[int, int]
    frame_rate: float
    video_codec: str
    audio_codec: str
    audio_tracks: int
    has_subtitles: bool


class Project(BaseModel):
    """Project entity representing a single recap video generation task."""
    project_id: str
    source_path: Path
    source_hash: str
    title: str
    release_year: Optional[int] = None
    source_language: str = 'ko'
    target_language: str = 'zh-CN'
    state: ProjectState = ProjectState.RECEIVED
    target_duration_range: Tuple[int, int] = (20, 30)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict)
    error_info: Optional[str] = None
    cost_usd: float = 0.0
    media_info: Optional[MediaInfo] = None

    def transition_to(self, new_state: ProjectState) -> None:
        """Transitions the project to a new state and updates timestamp."""
        self.state = new_state
        self.updated_at = datetime.utcnow()

    def to_json(self) -> str:
        """Serializes the project to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "Project":
        """Deserializes a project from a JSON string."""
        return cls.model_validate_json(json_str)

    def save(self, path: Path) -> None:
        """Saves the project data to a file."""
        with path.open("w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: Path) -> "Project":
        """Loads the project data from a file."""
        with path.open("r", encoding="utf-8") as f:
            return cls.from_json(f.read())
