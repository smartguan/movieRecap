from __future__ import annotations
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.queue.manager import FolderQueueManager, QueueItem
from src.queue.worker import RecapQueueWorker


def test_queue_enqueue_dequeue_lifecycle(tmp_path: Path):
    qm = FolderQueueManager(tmp_path / "queue")

    # Create dummy source video
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"dummy video data")

    metadata = {
        "title": "测试电影",
        "duration_seconds": 120.0,
        "genre": "Suspense",
    }

    # 1. Enqueue
    enqueued_path = qm.enqueue(
        slug="test_movie",
        source_video_path=source_video,
        metadata=metadata,
        custom_target_duration=3.0,
        duration_ratio=0.20,
    )
    assert enqueued_path.exists()
    assert (enqueued_path / "source.mp4").exists()
    assert (enqueued_path / "job.json").exists()

    status = qm.get_status()
    assert status["pending_count"] == 1
    assert "test_movie" in status["pending_jobs"]

    # 2. Dequeue
    item = qm.dequeue(worker_id="worker-test-1")
    assert item is not None
    assert item.slug == "test_movie"
    assert item.movie_title == "测试电影"
    assert item.custom_target_duration == 3.0
    assert (item.job_path / "worker.lock").exists()

    status_after_dequeue = qm.get_status()
    assert status_after_dequeue["pending_count"] == 0
    assert status_after_dequeue["processing_count"] == 1

    # 3. Mark completed
    comp_path = qm.mark_completed(
        slug="test_movie", summary={"status": "APPROVED", "recap_duration": 180.0}
    )
    assert comp_path.exists()
    assert (comp_path / "completion_summary.json").exists()

    status_final = qm.get_status()
    assert status_final["processing_count"] == 0
    assert status_final["completed_count"] == 1


def test_queue_failure_handling(tmp_path: Path):
    qm = FolderQueueManager(tmp_path / "queue")

    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"dummy video data")

    qm.enqueue(
        slug="failed_movie",
        source_video_path=source_video,
        metadata={"title": "Failed Film", "duration_seconds": 60.0},
    )

    item = qm.dequeue(worker_id="worker-test-fail")
    assert item is not None

    fail_path = qm.mark_failed(
        slug="failed_movie", error_message="Simulated pipeline render crash"
    )
    assert fail_path.exists()
    assert (fail_path / "error.json").exists()

    err_data = json.loads((fail_path / "error.json").read_text(encoding="utf-8"))
    assert err_data["error"] == "Simulated pipeline render crash"

    status = qm.get_status()
    assert status["failed_count"] == 1
    assert status["processing_count"] == 0


def test_queue_empty_dequeue_returns_none(tmp_path: Path):
    qm = FolderQueueManager(tmp_path / "queue")
    item = qm.dequeue(worker_id="worker-1")
    assert item is None


@patch("src.queue.worker.ingest_movie")
@patch("src.queue.worker.WorkflowRunner")
def test_recap_worker_run_once(mock_workflow_runner, mock_ingest, tmp_path: Path):
    qm = FolderQueueManager(tmp_path / "queue")

    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"dummy video data")

    qm.enqueue(
        slug="worker_movie",
        source_video_path=source_video,
        metadata={"title": "Worker Movie", "duration_seconds": 600.0},
        custom_target_duration=2.0,
    )

    mock_ingest.return_value = {"project_id": "proj-worker-test"}
    mock_runner_instance = MagicMock()
    mock_runner_instance.run_project.return_value = {"state": "EXPORTED"}
    mock_workflow_runner.return_value = mock_runner_instance

    worker = RecapQueueWorker(
        queue_manager=qm,
        projects_dir=tmp_path / "projects",
        output_dir=tmp_path / "output",
        worker_id="test-worker-unit",
    )

    processed_count = worker.run_once()
    assert processed_count == 1

    status = qm.get_status()
    assert status["completed_count"] == 1
    assert status["pending_count"] == 0
    assert status["processing_count"] == 0
