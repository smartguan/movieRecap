"""
Unified Staged Pipeline Runner.

Allows running end-to-end or jumping in from any stage (1, 2, or 3)
with automatic checkpoint resolution.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.stages.analyze import Stage2Result, run_stage_analyze
from src.stages.download import Stage1Result, run_stage_download
from src.stages.generate import Stage3Result, run_stage_generate

logger = logging.getLogger(__name__)


def run_staged_pipeline(
    input_target: str,
    from_stage: int = 1,
    to_stage: int = 3,
    target_duration: Optional[float] = None,
    duration_ratio: float = 0.20,
    force: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run pipeline from `from_stage` to `to_stage`.

    Args:
        input_target: URL, movie slug, or directory path.
        from_stage: 1 (Download), 2 (Analyze), or 3 (Generate Recap).
        to_stage: 1, 2, or 3.
        target_duration: Optional custom recap duration in minutes.
        duration_ratio: Proportional duration ratio (default: 0.20 = 1/5).
        force: If True, force re-execution of all selected stages.
        config: System configuration dict.

    Returns:
        Dict summarizing results of all executed stages.
    """
    if from_stage < 1 or from_stage > 3 or to_stage < from_stage or to_stage > 3:
        raise ValueError(f"Invalid stage range: from_stage={from_stage}, to_stage={to_stage}")

    results: Dict[str, Any] = {}
    current_target = input_target

    # Stage 1: Download
    if from_stage <= 1 and to_stage >= 1:
        logger.info("=== Executing Stage 1: Downloader ===")
        res1 = run_stage_download(url=current_target, force=force)
        results["stage1"] = res1.to_dict()
        current_target = res1.slug

    # Stage 2: Analyze
    if from_stage <= 2 and to_stage >= 2:
        logger.info("=== Executing Stage 2: Movie Analyzer ===")
        res2 = run_stage_analyze(stage1_input=current_target, config=config, force=force)
        results["stage2"] = res2.to_dict()
        current_target = res2.slug

    # Stage 3: Generate
    if from_stage <= 3 and to_stage >= 3:
        logger.info("=== Executing Stage 3: Recap Generator ===")
        res3 = run_stage_generate(
            stage2_input=current_target,
            target_duration=target_duration,
            duration_ratio=duration_ratio,
            config=config,
            force=force,
        )
        results["stage3"] = res3.to_dict()

    return results
