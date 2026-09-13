#!/usr/bin/env python3
"""
CLI tool to profile and display LLM token costs for movie recap projects.

Usage:
    python scripts/profile_cost.py [project_id]
    python scripts/profile_cost.py --latest
    python scripts/profile_cost.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.profiler import TokenCostProfiler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile LLM token consumption and cost for movie recap projects"
    )
    parser.add_argument(
        "project_id",
        nargs="?",
        default=None,
        help="UUID of the project to inspect (or use --latest)",
    )
    parser.add_argument(
        "--projects-dir",
        default="data/projects",
        help="Path to projects directory (default: data/projects)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Profile the most recently updated project",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON profile",
    )

    args = parser.parse_args()
    profiler = TokenCostProfiler(projects_dir=args.projects_dir)
    projects_path = Path(args.projects_dir)

    if not projects_path.exists():
        print(f"Error: Projects directory {projects_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    project_id = args.project_id

    if args.latest or not project_id:
        # Find latest project dir
        dirs = [d for d in projects_path.iterdir() if d.is_dir() and (d / "project.json").exists()]
        if not dirs:
            print("No projects found in", projects_path, file=sys.stderr)
            sys.exit(1)
        latest_dir = max(dirs, key=lambda d: (d / "project.json").stat().st_mtime)
        project_id = latest_dir.name

    try:
        profile = profiler.profile_project(project_id)
    except Exception as e:
        print(f"Error profiling project {project_id}: {e}", file=sys.stderr)
        sys.exit(1)

    # Save to project folder
    project_dir = projects_path / project_id
    profiler.save_profile_report(profile, project_dir)

    if args.json:
        import json
        from dataclasses import asdict
        print(json.dumps(asdict(profile), indent=2, ensure_ascii=False))
    else:
        # Print formatted Markdown report
        report = profiler.format_markdown_report(profile)
        print(report)


if __name__ == "__main__":
    main()
