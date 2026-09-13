"""
Video Acquirer Agent Orchestrator.
Dispatches video acquisition requests to extractors, downloads media deterministically,
and packages assets into the standard data/incoming/<slug>/ structure.
"""

from __future__ import annotations
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Union

from src.acquirer.base import BaseVideoExtractor
from src.acquirer.generic import GenericVideoExtractor
from src.acquirer.models import AcquisitionResult, StreamInfo, VideoMetadata
from src.acquirer.yfsp import YfspExtractor
from src.media.ingest import validate_incoming
from src.media.probe import extract_media_info

logger = logging.getLogger(__name__)


def slugify_title(title: str) -> str:
    """Generate a clean filesystem-safe folder slug from title."""
    # Remove filesystem unsafe characters
    clean = re.sub(r'[\\/*?:"<>|]', "", title)
    clean = re.sub(r"\s+", "_", clean).strip("_")
    return clean or "movie"


class VideoAcquirerAgent:
    """
    Autonomous agent for acquiring and packaging video assets.
    Adheres strictly to Deterministic-First execution (0 LLM tokens).
    """

    def __init__(self, extractors: Optional[List[BaseVideoExtractor]] = None) -> None:
        self.extractors: List[BaseVideoExtractor] = extractors or [
            YfspExtractor(),
            GenericVideoExtractor(),
        ]

    def register_extractor(self, extractor: BaseVideoExtractor) -> None:
        """Register a new platform extractor with highest priority."""
        self.extractors.insert(0, extractor)

    def get_extractor(self, url: str) -> BaseVideoExtractor:
        """Find the matching extractor for the URL."""
        for ext in self.extractors:
            if ext.can_handle(url):
                return ext
        raise ValueError(f"No registered extractor can handle URL: {url}")

    def download_stream(
        self,
        stream_info: StreamInfo,
        output_path: Path,
        max_duration: Optional[float] = None,
    ) -> None:
        """
        Download video stream using deterministic FFmpeg stream copy.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]

        # Add custom headers if required by CDN
        if stream_info.headers:
            header_str = "".join(
                f"{k}: {v}\r\n" for k, v in stream_info.headers.items()
            )
            cmd.extend(["-headers", header_str])

        cmd.extend(["-i", stream_info.stream_url])

        if max_duration and max_duration > 0:
            cmd.extend(["-t", str(max_duration)])

        # Stream copy video & audio without re-encoding
        cmd.extend(["-c", "copy", "-bsf:a", "aac_adtstoasc", str(output_path)])

        logger.info("Running deterministic stream copy: %s", " ".join(cmd[:6]))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg stream download failed (code {result.returncode}): {result.stderr}"
            )

    def acquire(
        self,
        url: str,
        incoming_base_dir: Optional[Union[Path, str]] = None,
        max_duration: Optional[float] = None,
    ) -> AcquisitionResult:
        """
        Acquire video from URL and prepare incoming package.

        Args:
            url: Webpage or direct stream URL.
            incoming_base_dir: Optional custom base directory (defaults to data/incoming).
            max_duration: Optional duration limit in seconds.

        Returns:
            AcquisitionResult with file paths and metadata.
        """
        base_dir = (
            Path(incoming_base_dir)
            if incoming_base_dir
            else Path("data/incoming")
        )

        try:
            # 1. Resolve extractor
            extractor = self.get_extractor(url)
            logger.info("Using extractor: %s for %s", extractor.__class__.__name__, url)

            # 2. Extract metadata
            metadata = extractor.extract_metadata(url)
            logger.info("Extracted metadata for: %s", metadata.title)

            # 3. Resolve media stream
            stream_info = extractor.resolve_stream(url)
            logger.info("Resolved stream URL: %s", stream_info.stream_url[:60] + "...")

            # 4. Prepare incoming directory
            slug = slugify_title(metadata.title)
            incoming_dir = base_dir / slug
            incoming_dir.mkdir(parents=True, exist_ok=True)

            source_video_path = incoming_dir / "source.mp4"
            metadata_path = incoming_dir / "metadata.json"

            # 5. Download media stream deterministically
            self.download_stream(
                stream_info=stream_info,
                output_path=source_video_path,
                max_duration=max_duration,
            )

            # 6. Extract media info & update metadata
            media_info = extract_media_info(source_video_path)
            metadata.duration_seconds = media_info.duration_seconds
            metadata.resolution = f"{media_info.resolution[0]}x{media_info.resolution[1]}"

            # 7. Write metadata.json
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.model_dump(), f, ensure_ascii=False, indent=2)

            # 8. Validate incoming directory against Movie Recap standard
            is_valid, errors = validate_incoming(incoming_dir)
            if not is_valid:
                raise ValueError(
                    f"Incoming validation failed: {', '.join(errors)}"
                )

            return AcquisitionResult(
                success=True,
                movie_title=metadata.title,
                incoming_dir=str(incoming_dir),
                source_video_path=str(source_video_path),
                metadata_path=str(metadata_path),
                duration_seconds=media_info.duration_seconds,
                file_size_bytes=source_video_path.stat().st_size,
                is_ready_for_recap=True,
            )

        except Exception as e:
            logger.error("Video acquisition failed: %s", e, exc_info=True)
            return AcquisitionResult(
                success=False,
                movie_title="",
                incoming_dir="",
                source_video_path="",
                metadata_path="",
                is_ready_for_recap=False,
                error_message=str(e),
            )
