"""
Stage 1: Downloader & Media Ingest (FR-Stage-1).

Acquires and validates the raw movie stream, extracting container metadata
and storing the artifacts in data/stages/1_download/<slug>/.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.acquirer.agent import VideoAcquirerAgent, slugify_title
from src.media.probe import extract_media_info

logger = logging.getLogger(__name__)


@dataclass
class Stage1Result:
    """Artifacts produced by Stage 1 (Downloader)."""
    slug: str
    stage_dir: Path
    source_video_path: Path
    metadata_path: Path
    movie_title: str
    duration_seconds: float
    file_size_bytes: int
    was_cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slug": self.slug,
            "stage_dir": str(self.stage_dir),
            "source_video_path": str(self.source_video_path),
            "metadata_path": str(self.metadata_path),
            "movie_title": self.movie_title,
            "duration_seconds": self.duration_seconds,
            "file_size_bytes": self.file_size_bytes,
            "was_cached": self.was_cached,
        }


def run_stage_download(
    url: str,
    output_dir: Optional[Path | str] = None,
    force: bool = False,
    max_duration: Optional[float] = None,
) -> Stage1Result:
    """
    Execute Stage 1: Download / Cache media stream and generate metadata.

    Args:
        url: Webpage URL, media stream, or existing local directory.
        output_dir: Base directory for Stage 1 artifacts (default: data/stages/1_download).
        force: If True, re-download even if already cached.
        max_duration: Optional limit on download runtime.

    Returns:
        Stage1Result with persistent file paths.
    """
    base_dir = Path(output_dir) if output_dir else Path("data/stages/1_download")
    base_dir.mkdir(parents=True, exist_ok=True)

    agent = VideoAcquirerAgent()
    acq = agent.acquire(
        url=url,
        incoming_base_dir=base_dir,
        max_duration=max_duration,
        force_download=force,
    )

    if not acq.success:
        raise RuntimeError(f"Stage 1 Download failed: {acq.error_message}")

    stage_dir = Path(acq.incoming_dir)
    source_path = Path(acq.source_video_path)
    meta_path = Path(acq.metadata_path)

    was_cached = not force and source_path.exists()

    result = Stage1Result(
        slug=stage_dir.name,
        stage_dir=stage_dir,
        source_video_path=source_path,
        metadata_path=meta_path,
        movie_title=acq.movie_title,
        duration_seconds=acq.duration_seconds,
        file_size_bytes=acq.file_size_bytes,
        was_cached=was_cached,
    )

    # Save checkpoint summary
    (stage_dir / "stage1_checkpoint.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Stage 1 complete: '%s' saved to %s", result.movie_title, stage_dir)
    return result
