"""
Generic Video Extractor for direct stream/media URLs (.mp4, .m3u8, .mkv, etc.).
"""

from __future__ import annotations
import urllib.parse
from pathlib import Path

from src.acquirer.base import BaseVideoExtractor
from src.acquirer.models import StreamInfo, VideoMetadata


class GenericVideoExtractor(BaseVideoExtractor):
    """
    Extractor for direct media URLs (m3u8 playlists, mp4/mkv files).
    """

    EXTENSIONS = {".m3u8", ".mp4", ".mkv", ".webm", ".mov", ".flv", ".ts"}

    def can_handle(self, url: str) -> bool:
        """Check if URL ends with a known media extension or is direct stream."""
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.lower()
        return any(path.endswith(ext) or ext in path for ext in self.EXTENSIONS)

    def extract_metadata(self, url: str) -> VideoMetadata:
        """Extract basic metadata from URL path."""
        parsed = urllib.parse.urlparse(url)
        filename = Path(parsed.path).stem or "video"
        title = filename.replace("_", " ").replace("-", " ").strip()

        return VideoMetadata(
            title=title or "Generic Video",
            source_url=url,
        )

    def resolve_stream(self, url: str) -> StreamInfo:
        """Resolve stream info from direct URL."""
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.lower()
        stream_type = "hls" if ".m3u8" in path else "mp4"

        return StreamInfo(
            stream_url=url,
            stream_type=stream_type,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
