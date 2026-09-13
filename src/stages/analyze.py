"""
Stage 2: Movie Analyzer (FR-Stage-2).

Performs scene detection, keyframe extraction, audio track isolation,
transcript parsing, scene indexing, and AI story understanding.
Persists all analysis artifacts in data/stages/2_analyzed/<slug>/.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ai.gateway import LLMGateway
from src.ai.story import understand_story
from src.ai.tasks import register_all_tasks
from src.media.audio import extract_audio
from src.media.frames import extract_scene_keyframes
from src.media.probe import extract_media_info
from src.media.scene_detect import detect_scenes
from src.media.subtitles import parse_srt

logger = logging.getLogger(__name__)


@dataclass
class Stage2Result:
    """Artifacts produced by Stage 2 (Movie Analyzer)."""
    slug: str
    stage_dir: Path
    movie_title: str
    duration_seconds: float
    scene_count: int
    keyframe_count: int
    scene_index_path: Path
    story_understanding_path: Path
    scenes_path: Path
    audio_path: Path
    subtitles_path: Optional[Path] = None
    characters: List[str] = field(default_factory=list)
    was_cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slug": self.slug,
            "stage_dir": str(self.stage_dir),
            "movie_title": self.movie_title,
            "duration_seconds": self.duration_seconds,
            "scene_count": self.scene_count,
            "keyframe_count": self.keyframe_count,
            "scene_index_path": str(self.scene_index_path),
            "story_understanding_path": str(self.story_understanding_path),
            "scenes_path": str(self.scenes_path),
            "audio_path": str(self.audio_path),
            "subtitles_path": str(self.subtitles_path) if self.subtitles_path else None,
            "characters": self.characters,
            "was_cached": self.was_cached,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], stage_dir: Path) -> Stage2Result:
        return cls(
            slug=data.get("slug", stage_dir.name),
            stage_dir=stage_dir,
            movie_title=data.get("movie_title", stage_dir.name),
            duration_seconds=data.get("duration_seconds", 0.0),
            scene_count=data.get("scene_count", 0),
            keyframe_count=data.get("keyframe_count", 0),
            scene_index_path=Path(data.get("scene_index_path", stage_dir / "scene_index.json")),
            story_understanding_path=Path(data.get("story_understanding_path", stage_dir / "story_understanding.json")),
            scenes_path=Path(data.get("scenes_path", stage_dir / "scenes.json")),
            audio_path=Path(data.get("audio_path", stage_dir / "audio" / "source_audio.wav")),
            subtitles_path=Path(data["subtitles_path"]) if data.get("subtitles_path") else None,
            characters=data.get("characters", []),
            was_cached=True,
        )


def _resolve_stage1_dir(stage1_input: Path | str) -> Path:
    p = Path(stage1_input)
    if p.exists() and (p / "source.mp4").exists():
        return p

    # Check data/stages/1_download/<slug>
    cand1 = Path("data/stages/1_download") / str(stage1_input)
    if cand1.exists() and (cand1 / "source.mp4").exists():
        return cand1

    # Check data/incoming/<slug>
    cand2 = Path("data/incoming") / str(stage1_input)
    if cand2.exists() and (cand2 / "source.mp4").exists():
        return cand2

    # Check data/queue/pending/<slug> or completed/<slug>
    for q_sub in ("pending", "processing", "completed"):
        cand_q = Path("data/queue") / q_sub / str(stage1_input)
        if cand_q.exists() and (cand_q / "source.mp4").exists():
            return cand_q

    raise FileNotFoundError(
        f"Could not find valid Stage 1 media package for '{stage1_input}' (missing source.mp4)"
    )


def run_stage_analyze(
    stage1_input: Path | str,
    output_dir: Optional[Path | str] = None,
    config: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> Stage2Result:
    """
    Execute Stage 2: Analyze movie media, extract features, and generate story understanding.

    Args:
        stage1_input: Path to Stage 1 directory or movie slug name.
        output_dir: Base directory for Stage 2 artifacts (default: data/stages/2_analyzed).
        config: System configuration dict.
        force: If True, re-run analysis even if checkpoint exists.

    Returns:
        Stage2Result with persistent analysis file paths.
    """
    stage1_dir = _resolve_stage1_dir(stage1_input)
    slug = stage1_dir.name
    base_output = Path(output_dir) if output_dir else Path("data/stages/2_analyzed")
    stage_dir = base_output / slug
    stage_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = stage_dir / "stage2_checkpoint.json"
    story_file = stage_dir / "story_understanding.json"
    scene_index_file = stage_dir / "scene_index.json"

    # Checkpoint check: if already analyzed and not forced, return cached result immediately
    if not force and checkpoint_file.exists() and story_file.exists() and scene_index_file.exists():
        try:
            data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            logger.info("Stage 2 checkpoint HIT for '%s'. Skipping analysis.", slug)
            return Stage2Result.from_dict(data, stage_dir)
        except Exception as e:
            logger.warning("Error reading Stage 2 checkpoint (%s), re-analyzing...", e)

    logger.info("Running Stage 2 Analysis for '%s'...", slug)
    source_path = stage1_dir / "source.mp4"
    metadata_file = stage1_dir / "metadata.json"

    meta_dict = {}
    if metadata_file.exists():
        try:
            meta_dict = json.loads(metadata_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    movie_title = meta_dict.get("title", slug)
    media_info = extract_media_info(source_path)
    duration_seconds = media_info.duration_seconds

    # 1. Extract Audio
    audio_dir = stage_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / "source_audio.wav"
    if not audio_path.exists() or force:
        logger.info("Extracting source audio track to %s...", audio_path)
        extract_audio(source_path, audio_path)

    # 2. Scene Detection (DETERMINISTIC)
    scenes_file = stage_dir / "scenes.json"
    if not scenes_file.exists() or force:
        logger.info("Detecting scene cuts via PySceneDetect...")
        scene_boundaries = detect_scenes(source_path)
        scenes_file.write_text(
            json.dumps([{"start": s, "end": e} for s, e in scene_boundaries], indent=2),
            encoding="utf-8",
        )
    else:
        scenes_data = json.loads(scenes_file.read_text(encoding="utf-8"))
        scene_boundaries = [(s["start"], s["end"]) for s in scenes_data]

    # 3. Keyframe Sampling (DETERMINISTIC)
    keyframes_dir = stage_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    if not any(keyframes_dir.glob("*.jpg")) or force:
        logger.info("Extracting representative keyframes...")
        extract_scene_keyframes(source_path, scene_boundaries, keyframes_dir, frames_per_scene=2)

    keyframe_paths = sorted(keyframes_dir.glob("*.jpg"))

    # 4. Parse Subtitles if present
    subtitles_path = None
    subtitles_file = stage_dir / "subtitles.json"
    srt_candidates = list(stage1_dir.glob("*.srt")) + list(stage_dir.glob("*.srt"))
    subtitle_entries: list[dict] = []
    if srt_candidates:
        subtitles = parse_srt(srt_candidates[0])
        subtitle_entries = [
            {"index": s.index, "start": s.start_seconds, "end": s.end_seconds, "text": s.text}
            for s in subtitles
        ]
        subtitles_file.write_text(
            json.dumps(subtitle_entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        subtitles_path = subtitles_file
    elif subtitles_file.exists():
        subtitle_entries = json.loads(subtitles_file.read_text(encoding="utf-8"))
        subtitles_path = subtitles_file

    # 5. Build Scene Index
    scenes = []
    for i, (start, end) in enumerate(scene_boundaries):
        scene_subs = [s for s in subtitle_entries if s["start"] < end and s["end"] > start]
        transcript = " ".join(s["text"] for s in scene_subs)
        kf_list = sorted(keyframes_dir.glob(f"scene_{i:04d}_*.jpg"))

        scenes.append({
            "scene_id": f"scene-{i:04d}",
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": round(end - start, 3),
            "transcript_text": transcript,
            "keyframe_paths": [str(p) for p in kf_list],
            "characters": [],
        })

    scene_index_dict = {
        "project_id": slug,
        "movie_title": movie_title,
        "scene_count": len(scenes),
        "duration_seconds": duration_seconds,
        "scenes": scenes,
    }
    scene_index_file.write_text(
        json.dumps(scene_index_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 6. Story Understanding (SEMANTIC via LLM Gateway)
    logger.info("Executing Story Understanding for '%s'...", movie_title)
    proj_data = {
        "title": movie_title,
        "project_id": slug,
        "metadata": meta_dict,
    }
    story_dict = understand_story(
        project=proj_data,
        scene_index=scene_index_dict,
        config=config or {},
    )
    story_file.write_text(
        json.dumps(story_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    character_names = [c.get("name", "") for c in story_dict.get("characters", []) if isinstance(c, dict)]

    result = Stage2Result(
        slug=slug,
        stage_dir=stage_dir,
        movie_title=movie_title,
        duration_seconds=duration_seconds,
        scene_count=len(scenes),
        keyframe_count=len(keyframe_paths),
        scene_index_path=scene_index_file,
        story_understanding_path=story_file,
        scenes_path=scenes_file,
        audio_path=audio_path,
        subtitles_path=subtitles_path,
        characters=character_names,
        was_cached=False,
    )

    checkpoint_file.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Stage 2 complete: '%s' analyzed (%d scenes, %d characters)", movie_title, len(scenes), len(character_names))
    return result
