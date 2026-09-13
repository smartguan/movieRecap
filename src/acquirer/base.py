"""
Base Video Extractor Protocol.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from src.acquirer.models import StreamInfo, VideoMetadata


class BaseVideoExtractor(ABC):
    """Abstract base class for platform-specific video extractors."""

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """Check if this extractor can process the provided URL."""
        pass

    @abstractmethod
    def extract_metadata(self, url: str) -> VideoMetadata:
        """Extract movie/video metadata from the source URL."""
        pass

    @abstractmethod
    def resolve_stream(self, url: str) -> StreamInfo:
        """Resolve direct stream URL (HLS / MP4) and required HTTP headers."""
        pass
