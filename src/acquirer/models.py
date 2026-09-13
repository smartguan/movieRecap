"""
Data models for Video Acquisition and Media Preprocessing.
"""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StreamInfo(BaseModel):
    stream_url: str
    stream_type: str = "hls"  # "hls", "mp4", "dash"
    headers: Dict[str, str] = Field(default_factory=dict)
    bitrate: int = 0
    resolution: str = ""
    duration_seconds: float = 0.0


class VideoMetadata(BaseModel):
    title: str
    source_url: str
    video_id: str = ""
    episode_id: str = ""
    channel: str = ""
    genre: str = ""
    year: str = ""
    stars: List[str] = Field(default_factory=list)
    directors: List[str] = Field(default_factory=list)
    synopsis: str = ""
    cover_image_url: str = ""
    duration_seconds: float = 0.0
    resolution: str = ""
    acquired_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extra: Dict[str, Any] = Field(default_factory=dict)


class AcquisitionResult(BaseModel):
    success: bool
    movie_title: str
    incoming_dir: str
    source_video_path: str
    metadata_path: str
    subtitles_path: Optional[str] = None
    duration_seconds: float = 0.0
    file_size_bytes: int = 0
    is_ready_for_recap: bool = True
    error_message: Optional[str] = None
