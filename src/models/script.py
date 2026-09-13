"""
Script generation data models.
"""
from datetime import datetime
from typing import List

from pydantic import BaseModel, computed_field, Field


class SceneReference(BaseModel):
    """Reference to a specific time range in a scene."""
    start_seconds: float
    end_seconds: float


class ScriptSegment(BaseModel):
    """A segment of the generated script."""
    segment_id: str
    text: str
    supporting_scenes: List[SceneReference] = Field(default_factory=list)
    confidence: float
    segment_type: str
    revision_count: int = 0
    pronunciation_notes: List[str] = Field(default_factory=list)


class Script(BaseModel):
    """The complete script for the recap video."""
    project_id: str
    segments: List[ScriptSegment] = Field(default_factory=list)
    target_speaking_rate: float = 250.0  # characters per minute for Mandarin
    created_at: datetime
    version: int = 1

    @computed_field
    @property
    def total_characters(self) -> int:
        """The total number of characters in the script."""
        return sum(len(segment.text) for segment in self.segments)

    @computed_field
    @property
    def estimated_duration_minutes(self) -> float:
        """The estimated spoken duration of the script in minutes."""
        if self.target_speaking_rate <= 0:
            return 0.0
        return self.total_characters / self.target_speaking_rate
