#!/usr/bin/env python3
"""
Multi-Stage Pipeline Orchestrator CLI.

Runs the complete pipeline or any slice [from_stage .. to_stage]
(1: Download, 2: Analyze, 3: Generate Recap).
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stages.pipeline import run_staged_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Movie Recap pipeline across decoupled stages [from-stage .. to-stage]"
    )
    parser.add_argument(
        "target",
        help="Input URL (for Stage 1) or movie slug (e.g. 诅咒 for Stages 2/3)",
    )
    parser.add_argument(
        "--from-stage",
        "-f",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Starting stage: 1 (Download), 2 (Analyze), 3 (Generate Recap)",
    )
    parser.add_argument(
        "--to-stage",
        "-t",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Ending stage: 1 (Download), 2 (Analyze), 3 (Generate Recap)",
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=None,
        help="Target recap duration in minutes (optional)",
    )
    parser.add_argument(
        "--duration-ratio",
        "-r",
        type=float,
        default=0.20,
        help="Proportional duration ratio (default: 0.20 = 1/5)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-execution of all specified stages",
    )

    args = parser.parse_args()

    stage_names = {
        1: "1. Downloader",
        2: "2. Movie Analyzer",
        3: "3. Recap Generator",
    }

    print("\n" + "=" * 70)
    print("🚀  MOVIE RECAP MULTI-STAGE PIPELINE")
    print("=" * 70)
    print(f"🎯 Target:      {args.target}")
    print(f"🔄 Stage Range: [{stage_names[args.from_stage]}] ➡️  [{stage_names[args.to_stage]}]")
    if args.target_duration:
        print(f"⏱️  Duration:    ~{args.target_duration:.2f} minutes")

    try:
        results = run_staged_pipeline(
            input_target=args.target,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            target_duration=args.target_duration,
            duration_ratio=args.duration_ratio,
            force=args.force,
        )

        print("\n" + "=" * 70)
        print("🎉  PIPELINE EXECUTION COMPLETE")
        print("=" * 70)
        for stage_key, summary in results.items():
            print(f"\n📊 {stage_key.upper()}:")
            for k, v in summary.items():
                if isinstance(v, (str, int, float, bool)):
                    print(f"   • {k}: {v}")
        return 0

    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
