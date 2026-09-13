"""
Video Acquirer Module.
Provides autonomous acquisition and packaging of online media streams.
"""

from __future__ import annotations
from src.acquirer.agent import VideoAcquirerAgent, slugify_title
from src.acquirer.base import BaseVideoExtractor
from src.acquirer.generic import GenericVideoExtractor
from src.acquirer.models import AcquisitionResult, StreamInfo, VideoMetadata
from src.acquirer.yfsp import YfspExtractor

__all__ = [
    "VideoAcquirerAgent",
    "BaseVideoExtractor",
    "GenericVideoExtractor",
    "YfspExtractor",
    "VideoMetadata",
    "StreamInfo",
    "AcquisitionResult",
    "slugify_title",
]
