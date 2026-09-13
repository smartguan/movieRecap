#!/usr/bin/env python3
"""
End-to-End Movie Recap Autopilot CLI.

Inputs:
- Online video URL (IYF / 爱壹帆 / Generic stream)
- Optional target duration in minutes (defaults to 1/5 of source movie length)

Outputs:
- Upload-ready multi-platform package for YouTube and Bilibili in output/<slug>/
"""

from __future__ import annotations

import argparse
import json
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
logger = logging.getLogger("produce_recap")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-End Movie Recap Autopilot: From URL to YouTube & Bilibili Ready Packages."
    )
    parser.add_argument("url", help="Movie URL (e.g., IYF play link or direct stream)")
    parser.add_argument(
        "--target-duration",
        "-t",
        type=float,
        default=None,
        help="Target recap duration in minutes (defaults to 1/5 of source movie runtime)",
    )
    parser.add_argument(
        "--duration-ratio",
        "-r",
        type=float,
        default=0.20,
        help="Proportional duration ratio (default: 0.20 = 1/5 of source runtime)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Output directory for platform-ready packages (default: output)",
    )
    parser.add_argument(
        "--projects-dir",
        "-p",
        default="data/projects",
        help="Workspace directory for project state and intermediate renders (default: data/projects)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download video stream even if movie is already cached in data/incoming/",
    )
    parser.add_argument(
        "--queue-only",
        action="store_true",
        help="Only fetch and enqueue into data/queue/pending/ without running recap pipeline immediately",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🎬  MOVIE RECAP AUTOPILOT — END-TO-END WORKFLOW")
    print("=" * 70)
    print(f"🔗 Source URL: {args.url}")
    if args.target_duration:
        print(f"⏱️  Target Recap Duration: {args.target_duration:.2f} minutes (Explicit)")
    else:
        print(f"⏱️  Duration Policy: 1/5 of source runtime (Ratio: {args.duration_ratio:.2f})")

    # Step 1: Acquire video and metadata deterministically
    print("\n[Step 1/5] Acquiring video stream & movie metadata (0 LLM tokens)...")
    acquirer = VideoAcquirerAgent()
    acq_result = acquirer.acquire(url=args.url, force_download=args.force_download)

    if not acq_result.success:
        print(f"❌ Acquisition failed: {acq_result.error_message}", file=sys.stderr)
        return 1

    source_duration_min = acq_result.duration_seconds / 60.0
    print(f"   • Movie Title: {acq_result.movie_title}")
    print(f"   • Source Duration: {source_duration_min:.2f} minutes ({acq_result.duration_seconds:.1f}s)")
    print(f"   • Source Video: {acq_result.source_video_path}")

    if args.queue_only:
        from src.queue.manager import FolderQueueManager
        qm = FolderQueueManager()
        safe_slug = Path(acq_result.incoming_dir).name
        meta_dict = {}
        if Path(acq_result.metadata_path).exists():
            meta_dict = json.loads(Path(acq_result.metadata_path).read_text(encoding="utf-8"))
        p_path = qm.enqueue(
            slug=safe_slug,
            source_video_path=acq_result.source_video_path,
            metadata=meta_dict,
            custom_target_duration=args.target_duration,
            duration_ratio=args.duration_ratio,
        )
        print(f"\n✅ Movie enqueued to {p_path} (--queue-only). Recap worker can process it asynchronously.")
        return 0

    # Calculate dynamic target duration range
    if args.target_duration is not None:
        target_recap_min = args.target_duration
    else:
        target_recap_min = max(0.5, round(source_duration_min * args.duration_ratio, 2))

    duration_range = [
        round(target_recap_min * 0.85, 2),
        round(target_recap_min * 1.15, 2),
    ]
    print(f"   • Computed Recap Target: ~{target_recap_min:.2f} min (Range: {duration_range[0]}-{duration_range[1]} min)")

    # Step 2: Ingest into projects directory
    print("\n[Step 2/5] Ingesting media and initializing project state...")
    incoming_dir = Path(acq_result.incoming_dir)
    projects_dir = Path(args.projects_dir)
    project_dict = ingest_movie(
        incoming_dir=incoming_dir,
        projects_dir=projects_dir,
        target_duration_range=duration_range,
        duration_ratio=args.duration_ratio,
    )
    project_id = project_dict["project_id"]
    print(f"   • Initialized Project: {project_id}")

    # Step 3: Run Workflow Pipeline
    print("\n[Step 3/5] Executing AI Recap Pipeline (Analyzing -> Scripting -> TTS -> Editing -> Rendering -> QA)...")
    config = {
        "paths": {
            "output_dir": args.output_dir,
            "projects_dir": args.projects_dir,
        }
    }
    runner = WorkflowRunner(
        projects_dir=projects_dir,
        incoming_dir=incoming_dir.parent,
        config=config,
    )

    final_project = runner.run_project(project_id)
    print(f"   • Workflow State: {final_project.get('state')}")

    # Step 4: Verification and Asset Summary
    print("\n[Step 4/5] Verifying Generated Assets & Multi-Platform Packages...")
    project_path = projects_dir / project_id
    safe_slug = incoming_dir.name
    package_path = Path(args.output_dir) / safe_slug

    if not package_path.exists():
        package_path = Path(args.output_dir)

    print("\n" + "=" * 70)
    print(f"🎉  RECAP GENERATION COMPLETE: {acq_result.movie_title}")
    print("=" * 70)
    print(f"📁 Platform Output Bundle: {package_path.resolve()}")
    print("\n📦 Generated Multi-Platform Assets:")
    print(f"   🎥 YouTube Video (1080p faststart): {package_path / 'recap_youtube.mp4'}")
    print(f"   📺 Bilibili Video (1080p MP4):       {package_path / 'recap_bilibili.mp4'}")
    print(f"   📝 YouTube Metadata (with Chapters): {package_path / 'metadata_youtube.json'}")
    print(f"   📑 Bilibili Metadata (with Tags):    {package_path / 'metadata_bilibili.json'}")
    print(f"   💬 Subtitles (.srt & .vtt):          {package_path / 'subtitles.srt'} | {package_path / 'subtitles.vtt'}")
    print(f"   🖼️  Thumbnail / Cover Art (16:9):     {package_path / 'cover.jpg'}")
    print(f"   🛡️  Anti-Slop Quality Assessment:     {package_path / 'recap_quality_report.md'}")
    print(f"   📊 Token & Cost Breakdown:           {package_path / 'token_profile.md'}")

    # Print summary of YouTube chapters if available
    yt_meta_file = package_path / "metadata_youtube.json"
    if yt_meta_file.exists():
        try:
            yt_meta = json.loads(yt_meta_file.read_text(encoding="utf-8"))
            print("\n⏰ Generated Chapter Timestamps:")
            for line in yt_meta.get("chapters", "").splitlines():
                print(f"      {line}")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
