"""
Scene data models.
"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, computed_field, Field


class Scene(BaseModel):
    """A logical scene extracted from the source media."""
    scene_id: str
    start_seconds: float
    end_seconds: float
    transcript_text: str = ''
    keyframe_paths: List[Path] = Field(default_factory=list)
    visual_description: str = ''
    characters: List[str] = Field(default_factory=list)
    location: str = ''
    actions: List[str] = Field(default_factory=list)
    emotional_tone: str = ''
    sensitivity_labels: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None

    @computed_field
    @property
    def duration_seconds(self) -> float:
        """The computed duration of the scene in seconds."""
        return max(0.0, self.end_seconds - self.start_seconds)


class SceneIndex(BaseModel):
    """An index of all scenes in a project."""
    project_id: str
    scenes: List[Scene]
    created_at: datetime
