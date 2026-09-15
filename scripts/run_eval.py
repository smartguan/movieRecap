#!/usr/bin/env python3
"""
CLI tool for Movie Recap Quality Evaluation and Anti-AI-Slop Audits.

Usage:
    python scripts/run_eval.py --benchmark
    python scripts/run_eval.py --latest
    python scripts/run_eval.py <project_id>
    python scripts/run_eval.py --json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.benchmark import BenchmarkSuiteRunner
from src.eval.slop_detector import SlopDetector
from src.orchestrator.project import ProjectManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Movie Recap Quality & Anti-Slop Scorecard")
    parser.add_argument("project_id", nargs="?", help="Project ID to evaluate")
    parser.add_argument("--projects-dir", default="data/projects", help="Path to projects directory")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark suite across all test projects")
    parser.add_argument("--latest", action="store_true", help="Evaluate the most recently modified project")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()
    projects_dir = Path(args.projects_dir)
    manager = ProjectManager(projects_dir)
    detector = SlopDetector()

    if args.benchmark:
        runner = BenchmarkSuiteRunner(projects_dir)
        summary = runner.run_suite()
        if args.json:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            print(runner.format_benchmark_table(summary))
        sys.exit(0 if summary["slop_detected_count"] == 0 else 1)

    project_id = args.project_id
    if args.latest or not project_id:
        projects = manager.list_projects()
        if not projects:
            print(f"Error: No projects found in {projects_dir}", file=sys.stderr)
            sys.exit(1)
        # Sort by updated_at descending
        projects.sort(key=lambda p: p.get("updated_at", ""), reverse=True)
    if Path(project_id).exists() and (Path(project_id) / "script.json").exists():
        project_dir = Path(project_id)
    else:
        project_dir = manager.get_project_dir(project_id)

    if not (project_dir / "script.json").exists():
        print(f"Error: Project {project_id} has no script.json", file=sys.stderr)
        sys.exit(1)

    report = detector.evaluate_project(project_dir)

    if args.json:
        print(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))
    else:
        print(detector.format_markdown_report(report))

    # Exit code: 0 if passed, 1 if slop / hard fail
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
