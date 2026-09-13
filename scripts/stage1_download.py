#!/usr/bin/env python3
"""
Stage 1 CLI: Downloader & Media Ingest.

Fetches video stream and extracts metadata into data/stages/1_download/<slug>/.
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stages.download import run_stage_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stage1_download")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 1: Acquire movie stream and metadata into data/stages/1_download/<slug>/"
    )
    parser.add_argument("url", help="Movie URL (IYF play link or generic stream)")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/stages/1_download",
        help="Output directory (default: data/stages/1_download)",
    )
    parser.add_argument(
        "--max-duration",
        "-t",
        type=float,
        default=None,
        help="Optional download duration limit in seconds",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if already cached",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("📥  STAGE 1: MOVIE DOWNLOADER")
    print("=" * 70)
    print(f"🔗 URL: {args.url}")

    try:
        res = run_stage_download(
            url=args.url,
            output_dir=args.output_dir,
            force=args.force,
            max_duration=args.max_duration,
        )

        print("\n✅ Stage 1 Complete!")
        print(f"   • Movie Title: {res.movie_title}")
        print(f"   • Duration:    {res.duration_seconds / 60.0:.2f} mins ({res.duration_seconds:.1f}s)")
        print(f"   • Stored in:   {res.stage_dir.resolve()}")
        print(f"   • Video File:  {res.source_video_path}")
        print(f"   • Cache Hit:   {res.was_cached}")
        return 0
    except Exception as e:
        print(f"\n❌ Stage 1 Failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
