"""
3-Stage Decoupled & Checkpointed Pipeline Framework.

Stage 1: Downloader (media stream & metadata)
Stage 2: Movie Analyzer (scene detection, keyframes, transcripts, story understanding)
Stage 3: Recap Generator (scripting, TTS, video timeline editing, QA, export)
"""

from __future__ import annotations
from src.stages.download import run_stage_download, Stage1Result
from src.stages.analyze import run_stage_analyze, Stage2Result
from src.stages.generate import run_stage_generate, Stage3Result
from src.stages.pipeline import run_staged_pipeline

__all__ = [
    "run_stage_download",
    "Stage1Result",
    "run_stage_analyze",
    "Stage2Result",
    "run_stage_generate",
    "Stage3Result",
    "run_staged_pipeline",
]
