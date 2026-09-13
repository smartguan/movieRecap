#!/usr/bin/env python3
"""
Detached Movie Fetcher CLI (Producer).

Fetches movies from online URLs (IYF / Generic streams) and enqueues them
into the folder queue (data/queue/pending/) for autonomous recap workers.
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acquirer.agent import VideoAcquirerAgent, slugify_title
from src.queue.manager import FolderQueueManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fetch_movie")


def parse_urls(input_args: list[str]) -> list[str]:
    """Parse single/multiple URLs or read lines from a file."""
    urls = []
    for arg in input_args:
        p = Path(arg)
        if p.exists() and p.is_file():
            # Read lines from file
            lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]
            urls.extend([line for line in lines if line and not line.startswith("#")])
        else:
            urls.append(arg)
    return urls


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Movie Fetcher: Download/cache movies and enqueue into data/queue/pending/"
    )
    parser.add_argument(
        "urls",
        nargs="+",
        help="One or more movie URLs or path to a text file containing URLs",
    )
    parser.add_argument(
        "--target-duration",
        "-t",
        type=float,
        default=None,
        help="Optional custom target recap duration in minutes",
    )
    parser.add_argument(
        "--duration-ratio",
        "-r",
        type=float,
        default=0.20,
        help="Proportional duration ratio (default: 0.20 = 1/5 runtime)",
    )
    parser.add_argument(
        "--queue-dir",
        "-q",
        default="data/queue",
        help="Queue base directory (default: data/queue)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Limit downloaded video duration in seconds (optional)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download video even if cached locally in data/incoming/",
    )

    args = parser.parse_args()

    urls_to_fetch = parse_urls(args.urls)
    if not urls_to_fetch:
        print("❌ No valid URLs provided.", file=sys.stderr)
        return 1

    print("\n" + "=" * 70)
    print("📥  MOVIE FETCHER & QUEUE PRODUCER")
    print("=" * 70)
    print(f"📋 Total URLs to fetch: {len(urls_to_fetch)}")
    print(f"📁 Target Queue: {Path(args.queue_dir).resolve()}")

    acquirer = VideoAcquirerAgent()
    queue_manager = FolderQueueManager(args.queue_dir)

    success_count = 0

    for i, url in enumerate(urls_to_fetch, 1):
        print(f"\n[{i}/{len(urls_to_fetch)}] Fetching movie: {url}")
        res = acquirer.acquire(
            url=url,
            max_duration=args.max_duration,
            force_download=args.force_download,
        )

        if not res.success:
            print(f"   ❌ Failed to acquire: {res.error_message}", file=sys.stderr)
            continue

        slug = Path(res.incoming_dir).name
        print(f"   🎬 Title: {res.movie_title}")
        print(f"   ⏱️  Runtime: {res.duration_seconds / 60.0:.2f} mins ({res.duration_seconds:.1f}s)")
        print(f"   💾 Staged at: {res.incoming_dir}")

        import json
        meta_dict = {}
        if Path(res.metadata_path).exists():
            meta_dict = json.loads(Path(res.metadata_path).read_text(encoding="utf-8"))

        pending_path = queue_manager.enqueue(
            slug=slug,
            source_video_path=res.source_video_path,
            metadata=meta_dict,
            custom_target_duration=args.target_duration,
            duration_ratio=args.duration_ratio,
        )

        print(f"   ✅ Enqueued to: {pending_path}")
        success_count += 1

    status = queue_manager.get_status()
    print("\n" + "=" * 70)
    print(f"🎉  FETCH & ENQUEUE COMPLETE: {success_count}/{len(urls_to_fetch)} movies enqueued.")
    print(f"📊 Queue Status: {status['pending_count']} pending | {status['processing_count']} processing | {status['completed_count']} completed")
    print("=" * 70)

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
