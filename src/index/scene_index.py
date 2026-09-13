"""
Searchable scene index.

DETERMINISTIC: Builds and queries a structured index of scenes.
LLM-generated descriptions may be added later but the index itself
is purely deterministic.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SceneEntry:
    """An entry in the scene index."""

    scene_id: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    transcript_text: str = ""
    keyframe_paths: list[str] = field(default_factory=list)
    visual_description: str = ""
    characters: list[str] = field(default_factory=list)
    location: str = ""
    actions: list[str] = field(default_factory=list)
    emotional_tone: str = ""
    sensitivity_labels: list[str] = field(default_factory=list)


class SceneIndex:
    """
    Searchable index of movie scenes.

    All query operations are deterministic. The index can be enriched
    with LLM-generated descriptions after construction.
    """

    def __init__(self, project_id: str, scenes: list[SceneEntry] | None = None) -> None:
        self.project_id = project_id
        self.scenes: list[SceneEntry] = scenes or []

    def add_scene(self, scene: SceneEntry) -> None:
        """Add a scene to the index."""
        self.scenes.append(scene)

    def search_transcript(self, query: str) -> list[SceneEntry]:
        """Search scenes by transcript text. DETERMINISTIC."""
        query_lower = query.lower()
        return [s for s in self.scenes if query_lower in s.transcript_text.lower()]

    def get_scenes_in_range(
        self, start_seconds: float, end_seconds: float
    ) -> list[SceneEntry]:
        """Get scenes that overlap with a time range. DETERMINISTIC."""
        return [
            s
            for s in self.scenes
            if s.start_seconds < end_seconds and s.end_seconds > start_seconds
        ]

    def get_scene_by_id(self, scene_id: str) -> SceneEntry | None:
        """Get a scene by its ID. DETERMINISTIC."""
        for s in self.scenes:
            if s.scene_id == scene_id:
                return s
        return None

    def get_scenes_by_character(self, character_name: str) -> list[SceneEntry]:
        """Get scenes featuring a specific character. DETERMINISTIC."""
        name_lower = character_name.lower()
        return [
            s for s in self.scenes if any(name_lower in c.lower() for c in s.characters)
        ]

    def get_sensitive_scenes(self) -> list[SceneEntry]:
        """Get scenes with sensitivity labels. DETERMINISTIC."""
        return [s for s in self.scenes if s.sensitivity_labels]

    def save(self, path: Path) -> None:
        """Save the scene index to JSON."""
        data = {
            "project_id": self.project_id,
            "scene_count": len(self.scenes),
            "scenes": [
                {
                    "scene_id": s.scene_id,
                    "start_seconds": s.start_seconds,
                    "end_seconds": s.end_seconds,
                    "duration_seconds": s.duration_seconds,
                    "transcript_text": s.transcript_text,
                    "keyframe_paths": s.keyframe_paths,
                    "visual_description": s.visual_description,
                    "characters": s.characters,
                    "location": s.location,
                    "actions": s.actions,
                    "emotional_tone": s.emotional_tone,
                    "sensitivity_labels": s.sensitivity_labels,
                }
                for s in self.scenes
            ],
        }
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "SceneIndex":
        """Load a scene index from JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        scenes = [
            SceneEntry(
                scene_id=s["scene_id"],
                start_seconds=s["start_seconds"],
                end_seconds=s["end_seconds"],
                duration_seconds=s["duration_seconds"],
                transcript_text=s.get("transcript_text", ""),
                keyframe_paths=s.get("keyframe_paths", []),
                visual_description=s.get("visual_description", ""),
                characters=s.get("characters", []),
                location=s.get("location", ""),
                actions=s.get("actions", []),
                emotional_tone=s.get("emotional_tone", ""),
                sensitivity_labels=s.get("sensitivity_labels", []),
            )
            for s in data.get("scenes", [])
        ]
        return cls(project_id=data.get("project_id", ""), scenes=scenes)
