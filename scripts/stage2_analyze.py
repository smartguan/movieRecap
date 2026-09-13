#!/usr/bin/env python3
"""
Stage 2 CLI: Movie Analyzer.

Extracts scenes, keyframes, audio, subtitles, and AI story understanding
into data/stages/2_analyzed/<slug>/.
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stages.analyze import run_stage_analyze

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stage2_analyze")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2: Analyze movie media, detect scenes, extract keyframes, and understand story"
    )
    parser.add_argument(
        "target",
        help="Movie slug (e.g. 诅咒) or path to Stage 1 downloaded directory",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/stages/2_analyzed",
        help="Base directory for analysis artifacts (default: data/stages/2_analyzed)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-analysis even if checkpoint exists",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🔍  STAGE 2: MOVIE ANALYZER")
    print("=" * 70)
    print(f"🎬 Target: {args.target}")

    try:
        res = run_stage_analyze(
            stage1_input=args.target,
            output_dir=args.output_dir,
            force=args.force,
        )

        print("\n✅ Stage 2 Complete!")
        print(f"   • Movie Title:   {res.movie_title}")
        print(f"   • Scene Cuts:    {res.scene_count} scenes")
        print(f"   • Keyframes:     {res.keyframe_count} frames")
        print(f"   • Characters:    {', '.join(res.characters) if res.characters else 'None identified'}")
        print(f"   • Analysis Dir:  {res.stage_dir.resolve()}")
        print(f"   • Checkpoint:    {res.stage_dir / 'stage2_checkpoint.json'}")
        print(f"   • Cache Hit:     {res.was_cached}")
        return 0
    except Exception as e:
        print(f"\n❌ Stage 2 Failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
