#!/usr/bin/env python3
"""
Detached Recap Queue Worker CLI (Consumer).

Continuously monitors data/queue/pending/, dequeues movies atomically,
generates full AI recaps, and exports platform packages to output/<slug>/.
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.queue.manager import FolderQueueManager
from src.queue.worker import RecapQueueWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("recap_worker")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recap Queue Worker: Dequeues movies and generates autonomous recaps."
    )
    parser.add_argument(
        "--queue-dir",
        "-q",
        default="data/queue",
        help="Queue directory (default: data/queue)",
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
        help="Workspace directory for project state (default: data/projects)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds when queue is empty (default: 5.0s)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process all currently pending movies and exit (single-pass mode)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current queue statistics and exit",
    )

    args = parser.parse_args()

    queue_manager = FolderQueueManager(args.queue_dir)

    if args.status:
        st = queue_manager.get_status()
        print("\n" + "=" * 60)
        print("📊  MOVIE RECAP QUEUE STATUS")
        print("=" * 60)
        print(f"📥 Pending:    {st['pending_count']:2d} {st['pending_jobs']}")
        print(f"⚙️  Processing: {st['processing_count']:2d} {st['processing_jobs']}")
        print(f"✅ Completed:  {st['completed_count']:2d} {st['completed_jobs']}")
        print(f"❌ Failed:     {st['failed_count']:2d} {st['failed_jobs']}")
        print("=" * 60 + "\n")
        return 0

    print("\n" + "=" * 70)
    print("⚙️   MOVIE RECAP QUEUE WORKER (CONSUMER)")
    print("=" * 70)
    print(f"📁 Queue Dir:    {Path(args.queue_dir).resolve()}")
    print(f"📁 Projects Dir: {Path(args.projects_dir).resolve()}")
    print(f"📁 Output Dir:   {Path(args.output_dir).resolve()}")

    worker = RecapQueueWorker(
        queue_manager=queue_manager,
        projects_dir=args.projects_dir,
        output_dir=args.output_dir,
    )

    if args.once:
        print("\n🚀 Running in single-pass mode (--once)...")
        processed = worker.run_once()
        print(f"\n🎉 Finished single-pass queue processing. Total processed: {processed}")
        return 0
    else:
        print(f"\n🔄 Running daemon mode (polling every {args.poll_interval:.1f}s)... Press Ctrl+C to exit.")
        worker.run_daemon(poll_interval=args.poll_interval)
        return 0


if __name__ == "__main__":
    sys.exit(main())
