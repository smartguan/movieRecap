"""
Deterministic Folder-Based Queue Manager (FR-Queue).

Implements atomic state transitions across:
  data/queue/pending/ -> data/queue/processing/ -> data/queue/completed/ (or failed/)
Adheres strictly to Deterministic-First execution (0 LLM tokens).
"""

from __future__ import annotations
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QueueItem:
    """Represents a movie item enqueued for recap generation."""
    slug: str
    job_path: Path
    movie_title: str
    duration_seconds: float
    enqueued_at: str
    custom_target_duration: Optional[float] = None
    duration_ratio: float = 0.20
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], job_path: Path) -> QueueItem:
        return cls(
            slug=data.get("slug", job_path.name),
            job_path=job_path,
            movie_title=data.get("movie_title", job_path.name),
            duration_seconds=data.get("duration_seconds", 0.0),
            enqueued_at=data.get("enqueued_at", datetime.now(timezone.utc).isoformat()),
            custom_target_duration=data.get("custom_target_duration"),
            duration_ratio=data.get("duration_ratio", 0.20),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slug": self.slug,
            "movie_title": self.movie_title,
            "duration_seconds": self.duration_seconds,
            "enqueued_at": self.enqueued_at,
            "custom_target_duration": self.custom_target_duration,
            "duration_ratio": self.duration_ratio,
            "metadata": self.metadata,
        }


class FolderQueueManager:
    """
    Manages a resilient, file-system based queue for detached video processing.
    """

    def __init__(self, queue_root: Optional[Path | str] = None) -> None:
        self.queue_root = Path(queue_root) if queue_root else Path("data/queue")
        self.pending_dir = self.queue_root / "pending"
        self.processing_dir = self.queue_root / "processing"
        self.completed_dir = self.queue_root / "completed"
        self.failed_dir = self.queue_root / "failed"
        self.staging_dir = self.queue_root / ".staging"

        # Ensure all subdirectories exist
        for d in (
            self.pending_dir,
            self.processing_dir,
            self.completed_dir,
            self.failed_dir,
            self.staging_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        slug: str,
        source_video_path: Path | str,
        metadata: Dict[str, Any],
        custom_target_duration: Optional[float] = None,
        duration_ratio: float = 0.20,
    ) -> Path:
        """
        Atomically enqueue a downloaded movie for recap processing.

        Args:
            slug: Safe movie folder identifier.
            source_video_path: Path to downloaded source video (source.mp4).
            metadata: Movie metadata dict.
            custom_target_duration: Optional target recap duration in minutes.
            duration_ratio: Proportional duration ratio (default: 0.20 = 1/5).

        Returns:
            Path to the enqueued directory in pending/.
        """
        source_path = Path(source_video_path)
        if not source_path.exists() or source_path.stat().st_size == 0:
            raise FileNotFoundError(f"Source video missing or empty: {source_path}")

        # 1. Prepare in staging folder
        stage_item = self.staging_dir / slug
        if stage_item.exists():
            shutil.rmtree(stage_item)
        stage_item.mkdir(parents=True, exist_ok=True)

        target_source = stage_item / "source.mp4"
        # If source is already in incoming/, we can copy or hardlink
        if source_path.resolve() != target_source.resolve():
            try:
                os.link(source_path, target_source)
            except (OSError, AttributeError):
                shutil.copy2(source_path, target_source)

        # Write metadata.json
        meta_target = stage_item / "metadata.json"
        meta_target.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Write job payload
        duration_sec = metadata.get("duration_seconds", 0.0)
        job_item = QueueItem(
            slug=slug,
            job_path=stage_item,
            movie_title=metadata.get("title", slug),
            duration_seconds=duration_sec,
            enqueued_at=datetime.now(timezone.utc).isoformat(),
            custom_target_duration=custom_target_duration,
            duration_ratio=duration_ratio,
            metadata=metadata,
        )
        (stage_item / "job.json").write_text(
            json.dumps(job_item.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 2. Atomic move from staging to pending
        dest_pending = self.pending_dir / slug
        if dest_pending.exists():
            shutil.rmtree(dest_pending)

        shutil.move(str(stage_item), str(dest_pending))
        logger.info("Enqueued movie '%s' to %s", slug, dest_pending)
        return dest_pending

    def dequeue(self, worker_id: str = "worker-default") -> Optional[QueueItem]:
        """
        Atomically dequeue the oldest pending movie.

        Returns:
            QueueItem claimed by worker, or None if pending queue is empty.
        """
        pending_items = [
            p for p in self.pending_dir.iterdir() if p.is_dir() and (p / "source.mp4").exists()
        ]
        if not pending_items:
            return None

        # Sort by creation time (FIFO)
        pending_items.sort(key=lambda p: p.stat().st_ctime)

        for candidate in pending_items:
            slug = candidate.name
            target_proc = self.processing_dir / slug

            # Attempt atomic move to processing
            try:
                shutil.move(str(candidate), str(target_proc))
            except (OSError, shutil.Error):
                # Another concurrent worker claimed this candidate
                continue

            # Record worker claim lock
            lock_data = {
                "worker_id": worker_id,
                "pid": os.getpid(),
                "claimed_at": datetime.now(timezone.utc).isoformat(),
            }
            (target_proc / "worker.lock").write_text(
                json.dumps(lock_data, indent=2), encoding="utf-8"
            )

            # Read job.json or reconstruct from metadata.json
            job_file = target_proc / "job.json"
            if job_file.exists():
                try:
                    data = json.loads(job_file.read_text(encoding="utf-8"))
                    return QueueItem.from_dict(data, target_proc)
                except Exception as e:
                    logger.warning("Error reading job.json (%s), fallback to metadata", e)

            meta_file = target_proc / "metadata.json"
            metadata = {}
            if meta_file.exists():
                try:
                    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            item = QueueItem(
                slug=slug,
                job_path=target_proc,
                movie_title=metadata.get("title", slug),
                duration_seconds=metadata.get("duration_seconds", 0.0),
                enqueued_at=datetime.now(timezone.utc).isoformat(),
                metadata=metadata,
            )
            return item

        return None

    def mark_completed(
        self, slug: str, summary: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Move movie package from processing to completed."""
        proc_path = self.processing_dir / slug
        dest_path = self.completed_dir / slug

        if not proc_path.exists():
            raise FileNotFoundError(f"Cannot complete job not in processing: {proc_path}")

        if dest_path.exists():
            shutil.rmtree(dest_path)

        # Write completion summary
        if summary:
            (proc_path / "completion_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        shutil.move(str(proc_path), str(dest_path))
        logger.info("Marked movie '%s' as COMPLETED at %s", slug, dest_path)
        return dest_path

    def mark_failed(self, slug: str, error_message: str) -> Path:
        """Move movie package from processing to failed with error diagnostic log."""
        proc_path = self.processing_dir / slug
        dest_path = self.failed_dir / slug

        if not proc_path.exists():
            # Check if it was in pending
            pending_path = self.pending_dir / slug
            if pending_path.exists():
                proc_path = pending_path

        if not proc_path.exists():
            raise FileNotFoundError(f"Cannot fail job not found: {slug}")

        if dest_path.exists():
            shutil.rmtree(dest_path)

        # Write error log
        err_data = {
            "slug": slug,
            "error": error_message,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        (proc_path / "error.json").write_text(
            json.dumps(err_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        shutil.move(str(proc_path), str(dest_path))
        logger.warning("Marked movie '%s' as FAILED at %s (error: %s)", slug, dest_path, error_message)
        return dest_path

    def get_status(self) -> Dict[str, Any]:
        """Get summary status of the queue."""
        def list_items(d: Path) -> List[str]:
            return [p.name for p in d.iterdir() if p.is_dir()]

        pending = list_items(self.pending_dir)
        processing = list_items(self.processing_dir)
        completed = list_items(self.completed_dir)
        failed = list_items(self.failed_dir)

        return {
            "pending_count": len(pending),
            "processing_count": len(processing),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "pending_jobs": pending,
            "processing_jobs": processing,
            "completed_jobs": completed,
            "failed_jobs": failed,
        }
