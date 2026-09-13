"""
Recap Queue Consumer Worker (FR-Queue).

Monitors data/queue/pending/, claims movies atomically, executes the full
AI recap workflow, generates platform packages, and transitions queue state.
"""

from __future__ import annotations
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.media.ingest import ingest_movie
from src.orchestrator.workflow import WorkflowRunner
from src.queue.manager import FolderQueueManager, QueueItem

logger = logging.getLogger(__name__)


class RecapQueueWorker:
    """
    Consumer worker that picks up queued movies and generates recaps.
    """

    def __init__(
        self,
        queue_manager: Optional[FolderQueueManager] = None,
        projects_dir: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
        worker_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.queue_manager = queue_manager or FolderQueueManager()
        self.projects_dir = Path(projects_dir) if projects_dir else Path("data/projects")
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.worker_id = worker_id or f"worker-{os.getpid()}"
        self.config = config or {
            "paths": {
                "output_dir": str(self.output_dir),
                "projects_dir": str(self.projects_dir),
            }
        }

        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_item(self, item: QueueItem) -> Dict[str, Any]:
        """
        Process a single claimed queue item.

        Args:
            item: Claimed QueueItem.

        Returns:
            Dict of execution results.
        """
        logger.info(
            "[%s] Starting recap generation for '%s' (Source duration: %.1fs)...",
            self.worker_id,
            item.movie_title,
            item.duration_seconds,
        )

        source_duration_min = item.duration_seconds / 60.0

        # Calculate proportional target recap duration
        if item.custom_target_duration is not None and item.custom_target_duration > 0:
            target_recap_min = item.custom_target_duration
        else:
            target_recap_min = max(
                0.5, round(source_duration_min * item.duration_ratio, 2)
            )

        duration_range = [
            round(target_recap_min * 0.85, 2),
            round(target_recap_min * 1.15, 2),
        ]

        logger.info(
            "[%s] Target duration: ~%.2f min (Range: %.2f - %.2f min)",
            self.worker_id,
            target_recap_min,
            duration_range[0],
            duration_range[1],
        )

        try:
            # 1. Ingest movie into projects directory
            project_dict = ingest_movie(
                incoming_dir=item.job_path,
                projects_dir=self.projects_dir,
                target_duration_range=duration_range,
                duration_ratio=item.duration_ratio,
            )
            project_id = project_dict["project_id"]

            # 2. Run AI recap workflow pipeline
            runner = WorkflowRunner(
                projects_dir=self.projects_dir,
                incoming_dir=item.job_path.parent,
                config=self.config,
            )
            final_project = runner.run_project(project_id)

            # 3. Verify output bundle
            package_path = self.output_dir / item.slug
            summary = {
                "slug": item.slug,
                "project_id": project_id,
                "movie_title": item.movie_title,
                "status": final_project.get("state"),
                "package_path": str(package_path),
                "target_recap_minutes": target_recap_min,
                "completed_by": self.worker_id,
            }

            # 4. Mark job completed
            self.queue_manager.mark_completed(item.slug, summary=summary)
            logger.info(
                "[%s] Successfully completed recap for '%s' -> %s",
                self.worker_id,
                item.movie_title,
                package_path,
            )
            return summary

        except Exception as e:
            logger.error(
                "[%s] Failed to process recap for '%s': %s",
                self.worker_id,
                item.movie_title,
                e,
                exc_info=True,
            )
            self.queue_manager.mark_failed(item.slug, error_message=str(e))
            raise

    def run_once(self) -> int:
        """
        Process all currently pending items in the queue once and exit.

        Returns:
            Number of items processed.
        """
        count = 0
        while True:
            item = self.queue_manager.dequeue(worker_id=self.worker_id)
            if item is None:
                break
            try:
                self.process_item(item)
                count += 1
            except Exception as e:
                logger.error("Error processing queue item %s: %s", item.slug, e)
        return count

    def run_daemon(self, poll_interval: float = 5.0) -> None:
        """
        Run continuous daemon worker polling the queue.

        Args:
            poll_interval: Seconds to wait between polling attempts when idle.
        """
        logger.info(
            "[%s] Starting RecapQueueWorker daemon (polling %s every %.1fs)...",
            self.worker_id,
            self.queue_manager.pending_dir,
            poll_interval,
        )

        try:
            while True:
                item = self.queue_manager.dequeue(worker_id=self.worker_id)
                if item is not None:
                    try:
                        self.process_item(item)
                    except Exception as e:
                        logger.error("Job failed for %s: %s", item.slug, e)
                else:
                    time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("[%s] Worker interrupted. Shutting down gracefully.", self.worker_id)
