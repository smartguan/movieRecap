#!/usr/bin/env python3
"""
Stage 3 CLI: Recap Generator.

Generates commentary script, TTS audio, video assembly, QA scorecard,
and multi-platform packages in output/<slug>/ using Stage 2 analyzed artifacts.
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stages.generate import run_stage_generate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stage3_generate")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 3: Generate commentary script, audio, video, and multi-platform packages"
    )
    parser.add_argument(
        "target",
        help="Movie slug (e.g. 诅咒) or path to Stage 2 analyzed directory",
    )
    parser.add_argument(
        "--target-duration",
        "-t",
        type=float,
        default=None,
        help="Target recap duration in minutes (defaults to 1/5 runtime)",
    )
    parser.add_argument(
        "--duration-ratio",
        "-r",
        type=float,
        default=0.20,
        help="Proportional duration ratio (default: 0.20 = 1/5 runtime)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Base directory for platform-ready packages (default: output)",
    )
    parser.add_argument(
        "--algo-version",
        default="v4",
        help="Algorithm version ('v4' for grounded narrative spine, 'v3', 'v2')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-generation of script and renders",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🎬  STAGE 3: RECAP GENERATOR")
    print("=" * 70)
    print(f"🎬 Target: {args.target}")
    if args.target_duration:
        print(f"⏱️  Duration: ~{args.target_duration:.2f} mins (Explicit)")
    else:
        print(f"⏱️  Duration Ratio: {args.duration_ratio * 100:.0f}% of source runtime")

    try:
        res = run_stage_generate(
            stage2_input=args.target,
            output_dir=args.output_dir,
            target_duration=args.target_duration,
            duration_ratio=args.duration_ratio,
            force=args.force,
            algo_version=args.algo_version,
        )

        print("\n✅ Stage 3 Complete!")
        print(f"   • Movie Title:      {res.movie_title}")
        print(f"   • Recap Duration:   ~{res.target_recap_minutes:.2f} minutes")
        print(f"   • Anti-Slop QA:     Grade {res.quality_grade} ({res.quality_score:.1f}/100)")
        print(f"   • Platform Package: {res.platform_output_dir.resolve()}")
        print("\n📦 Exported Files:")
        for f in res.exported_files:
            print(f"      - {Path(f).name}")
        return 0
    except Exception as e:
        print(f"\n❌ Stage 3 Failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
