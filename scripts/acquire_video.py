#!/usr/bin/env python3
"""
CLI script to acquire video from URL, download stream deterministically,
and package it into data/incoming/<slug>/.
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acquirer.agent import VideoAcquirerAgent
from src.media.ingest import ingest_movie
from src.orchestrator.workflow import WorkflowRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("acquire_video")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acquire video from online stream/URL and prepare standard incoming package."
    )
    parser.add_argument("url", help="URL of the video to acquire (e.g., yfsp.tv play link or direct stream)")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/incoming",
        help="Base directory for incoming packages (default: data/incoming)",
    )
    parser.add_argument(
        "--max-duration",
        "-t",
        type=float,
        default=None,
        help="Limit downloaded video duration in seconds (optional)",
    )
    parser.add_argument(
        "--run-recap",
        action="store_true",
        help="Automatically trigger recap workflow after acquisition",
    )

    args = parser.parse_args()

    agent = VideoAcquirerAgent()
    print(f"🎬 Starting video acquisition for: {args.url}")
    
    result = agent.acquire(
        url=args.url,
        incoming_base_dir=args.output_dir,
        max_duration=args.max_duration,
    )

    if not result.success:
        print(f"❌ Acquisition failed: {result.error_message}", file=sys.stderr)
        return 1

    print("\n✅ Video Acquisition Successful!")
    print(f"   • Movie Title: {result.movie_title}")
    print(f"   • Incoming Dir: {result.incoming_dir}")
    print(f"   • Source Video: {result.source_video_path}")
    print(f"   • Metadata File: {result.metadata_path}")
    print(f"   • Duration: {result.duration_seconds:.2f} seconds")
    print(f"   • File Size: {result.file_size_bytes / (1024 * 1024):.2f} MB")
    print(f"   • Ready for Recap: {result.is_ready_for_recap}")

    if args.run_recap:
        print("\n🚀 Ingesting and triggering Movie Recap workflow...")
        incoming_dir = Path(result.incoming_dir)
        projects_dir = Path("data/projects")
        project = ingest_movie(incoming_dir, projects_dir)
        project_id = project["project_id"]
        print(f"   • Created Project: {project_id}")

        runner = WorkflowRunner(projects_dir=projects_dir, incoming_dir=incoming_dir.parent)
        completed_project = runner.run_project(project_id)
        print(f"🎉 Recap finished with status: {completed_project['state']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
