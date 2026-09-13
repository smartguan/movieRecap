"""
Story understanding data models.
"""
from datetime import datetime
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class Character(BaseModel):
    """A character identified in the story."""
    name: str
    aliases: List[str] = Field(default_factory=list)
    description: str
    first_appearance_seconds: Optional[float] = None


class CharacterRelationship(BaseModel):
    """A relationship between two characters."""
    character_a: str
    character_b: str
    relationship_type: str
    description: str


class StoryEvent(BaseModel):
    """An important event in the story timeline."""
    event_id: str
    description: str
    characters: List[str] = Field(default_factory=list)
    timestamp_seconds: float
    evidence_timestamps: List[Tuple[float, float]] = Field(default_factory=list)
    confidence: float
    event_type: str  # e.g., 'conflict', 'turning_point', 'climax', 'resolution'


class StoryUnderstanding(BaseModel):
    """Comprehensive understanding of the story's narrative."""
    project_id: str
    characters: List[Character] = Field(default_factory=list)
    relationships: List[CharacterRelationship] = Field(default_factory=list)
    events: List[StoryEvent] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    ambiguities: List[str] = Field(default_factory=list)
    created_at: datetime
