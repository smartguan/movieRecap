"""
Story understanding module (FR-3).

Implements hierarchical processing and fetcher metadata synergy to minimize token usage:
1. Deterministically segment transcript and scene data (done upstream in index/).
2. Leverage structured metadata (synopsis, cast, genre) from Video Acquirer Agent to avoid blind LLM exploration.
3. Deterministically handle silent/dialogue-free scene chunks with 0 LLM tokens.
4. Pre-seed character registry from fetcher cast & synopsis.
5. Create compact bounded semantic summaries for active dialogue sequences.
6. Assemble global story understanding via compact token-dense representations.
"""

from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.ai.gateway import LLMGateway
from src.ai.tasks import register_all_tasks

logger = logging.getLogger(__name__)

# Maximum number of scenes per local summary chunk
CHUNK_SIZE = 15


def understand_story(
    project: dict[str, Any],
    scene_index: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a complete story understanding from the scene index and project metadata.

    Uses hierarchical processing & fetcher metadata synergy:
    1. Extract known metadata (synopsis, stars, genre) from fetcher
    2. Chunk scenes into groups of CHUNK_SIZE
    3. Generate local summaries (skipping silent chunks deterministically)
    4. Extract/seed characters and relationships
    5. Synthesize global story understanding

    Args:
        project: Project data dict.
        scene_index: Scene index with transcript and metadata per scene.
        config: System configuration.

    Returns:
        Story understanding dict with characters, events, relationships, etc.
    """
    gateway = LLMGateway(config)
    register_all_tasks(gateway)

    scenes = scene_index.get("scenes", [])
    title = project.get("title", "Unknown Movie")
    metadata = project.get("metadata", {})

    # Step 1: Create local summaries for scene chunks (SEMANTIC with deterministic gating)
    local_summaries = _create_local_summaries(gateway, scenes, title, metadata)

    # Step 2: Extract / Seed characters using fetcher metadata (DETERMINISTIC seeding + SEMANTIC refinement)
    characters = _extract_characters(gateway, local_summaries, title, metadata)

    # Step 3: Synthesize global story (SEMANTIC with token-optimized context)
    story = _synthesize_story(gateway, local_summaries, characters, title, metadata)

    # Save the gateway telemetry
    story["telemetry"] = gateway.get_telemetry()

    return story


def _create_local_summaries(
    gateway: LLMGateway,
    scenes: list[dict[str, Any]],
    title: str,
    metadata: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    Create bounded local summaries for chunks of scenes.
    Skips dialogue-free chunks deterministically to save 100% LLM tokens on non-dialogue scenes.

    Args:
        gateway: LLM gateway instance.
        scenes: List of scene dicts from the scene index.
        title: Movie title for context.
        metadata: Optional metadata from fetcher agent.

    Returns:
        List of local summary dicts.
    """
    summaries = []
    chunks = [scenes[i : i + CHUNK_SIZE] for i in range(0, len(scenes), CHUNK_SIZE)]
    meta = metadata or {}
    synopsis = meta.get("synopsis", "").strip()
    synopsis_snippet = f" | 背景: {synopsis[:120]}..." if synopsis else ""

    for chunk_idx, chunk in enumerate(chunks):
        # 1. Deterministic Token Optimization: Check if chunk has any dialogue
        non_empty_transcripts = [
            s.get("transcript_text", "").strip()
            for s in chunk
            if s.get("transcript_text", "").strip()
        ]

        time_range_str = (
            f"{chunk[0]['start_seconds']:.1f}s - {chunk[-1]['end_seconds']:.1f}s"
        )

        if not non_empty_transcripts:
            # Chunk is silent / action-only: assemble deterministic summary with 0 LLM tokens
            logger.info(
                "Chunk %d/%d is dialogue-free; generating deterministic summary (0 tokens)",
                chunk_idx + 1,
                len(chunks),
            )
            summaries.append({
                "section_index": chunk_idx,
                "time_range": time_range_str,
                "events": [
                    {
                        "description": f"画面转场与背景镜头 ({time_range_str})",
                        "characters": [],
                        "timestamp_seconds": chunk[0]["start_seconds"],
                        "event_type": "exposition",
                        "confidence": 1.0,
                    }
                ],
                "characters_seen": [],
                "emotional_tone": "过渡/平静",
                "key_dialogue": [],
                "summary": f"该时间段 ({time_range_str}) 主要是无对白背景与场景镜头过渡。",
            })
            continue

        # 2. Build compact context from this chunk only
        context_lines = []
        for scene in chunk:
            time_range = f"{scene['start_seconds']:.1f}s - {scene['end_seconds']:.1f}s"
            transcript = scene.get("transcript_text", "").strip()
            if transcript:
                context_lines.append(f"[{time_range}] {transcript}")

        context = "\n".join(context_lines)

        prompt = f"""Analyze section {chunk_idx + 1}/{len(chunks)} of "{title}"{synopsis_snippet}.

Based on the transcript below, identify:
1. Key events that happen in this section
2. Characters who appear and what they do
3. Important dialogue or revelations
4. The emotional tone of this section

Respond in JSON format:
{{
  "section_index": {chunk_idx},
  "time_range": "{time_range_str}",
  "events": [
    {{
      "description": "what happened",
      "characters": ["character names"],
      "timestamp_seconds": 0.0,
      "event_type": "setup|conflict|turning_point|climax|resolution|exposition",
      "confidence": 0.9
    }}
  ],
  "characters_seen": ["list of character names"],
  "emotional_tone": "mood",
  "key_dialogue": ["important lines"],
  "summary": "2-3 sentence summary of this section"
}}"""

        evidence_refs = [s["scene_id"] for s in chunk]

        response = gateway.invoke(
            task_name="local_summary",
            prompt=prompt,
            context=context,
            evidence_refs=evidence_refs,
            project_id="",
        )

        summary = response.parsed or {
            "section_index": chunk_idx,
            "time_range": time_range_str,
            "summary": response.content,
        }
        summaries.append(summary)

        logger.info(
            "Local summary %d/%d: %d tokens in, %d tokens out",
            chunk_idx + 1,
            len(chunks),
            response.input_tokens,
            response.output_tokens,
        )

    return summaries


def _extract_characters(
    gateway: LLMGateway,
    local_summaries: list[dict[str, Any]],
    title: str,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Extract and consolidate characters using fetcher metadata synergy.
    If fetcher provides cast and synopsis, pre-seeds character registry deterministically.

    Args:
        gateway: LLM gateway instance.
        local_summaries: Local summaries from each chunk.
        title: Movie title.
        metadata: Metadata extracted by fetcher agent.

    Returns:
        Dict with characters and relationships.
    """
    meta = metadata or {}
    stars = meta.get("stars", [])
    synopsis = meta.get("synopsis", "")

    # Collect all character mentions from local summaries (DETERMINISTIC)
    all_characters = set()
    for summary in local_summaries:
        for char in summary.get("characters_seen", []):
            if char:
                all_characters.add(char)

    # 1. Deterministic Token Optimization: If no characters in summaries and stars are known
    if not all_characters and stars:
        logger.info("Using fetcher stars to construct deterministic character registry (0 tokens)")
        char_list = [
            {
                "name": star,
                "aliases": [],
                "description": "主要角色/主演",
                "first_appearance_seconds": 0.0,
            }
            for star in stars
        ]
        return {"characters": char_list, "relationships": []}

    # Prepare compact context — only character-relevant info + fetcher hints
    context_lines = []
    if stars:
        context_lines.append(f"Official Cast/Stars: {', '.join(stars)}")
    if synopsis:
        context_lines.append(f"Official Synopsis: {synopsis}")

    for summary in local_summaries:
        section_summary = summary.get("summary", "")
        chars = summary.get("characters_seen", [])
        if chars or section_summary:
            char_str = f"Characters: {', '.join(chars)}. " if chars else ""
            context_lines.append(
                f"Section {summary.get('section_index', '?')}: {char_str}{section_summary}"
            )

    context = "\n".join(context_lines)

    prompt = f"""Given the section summaries and cast hints for the movie "{title}", create a consolidated character registry.

Respond in JSON format:
{{
  "characters": [
    {{
      "name": "main name",
      "aliases": ["other names used"],
      "description": "role and identity",
      "first_appearance_seconds": 0.0
    }}
  ],
  "relationships": [
    {{
      "character_a": "name",
      "character_b": "name",
      "relationship_type": "type",
      "description": "relationship nature"
    }}
  ]
}}"""

    response = gateway.invoke(
        task_name="character_extraction",
        prompt=prompt,
        context=context,
    )

    parsed = response.parsed or {"characters": [], "relationships": []}

    # If LLM didn't return characters but fetcher stars exist, merge them deterministically
    if not parsed.get("characters") and stars:
        parsed["characters"] = [
            {
                "name": star,
                "aliases": [],
                "description": "主要角色/主演",
                "first_appearance_seconds": 0.0,
            }
            for star in stars
        ]

    return parsed


def _synthesize_story(
    gateway: LLMGateway,
    local_summaries: list[dict[str, Any]],
    characters: Any,
    title: str,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Synthesize a global story understanding using compact token-optimized context.

    Args:
        gateway: LLM gateway instance.
        local_summaries: All local summaries.
        characters: Character registry.
        title: Movie title.
        metadata: Optional fetcher metadata.

    Returns:
        Complete story understanding dict.
    """
    meta = metadata or {}
    synopsis = meta.get("synopsis", "")
    genre = meta.get("genre", "")

    # Build high-density compact context
    context_lines = []
    if genre:
        context_lines.append(f"GENRE: {genre}")
    if synopsis:
        context_lines.append(f"OFFICIAL SYNOPSIS: {synopsis}")

    # Compact character list
    if isinstance(characters, dict) and characters.get("characters"):
        char_strs = [
            f"- {c.get('name')}: {c.get('description', '')}"
            for c in characters.get("characters", [])
        ]
        context_lines.append("CHARACTERS:\n" + "\n".join(char_strs))

    summary_lines = []
    for summary in local_summaries:
        time_range = summary.get("time_range", "?")
        section_summary = summary.get("summary", "")
        summary_lines.append(f"[{time_range}] {section_summary}")

    context_lines.append("SECTION SUMMARIES:\n" + "\n".join(summary_lines))
    context = "\n\n".join(context_lines)

    prompt = f"""Based on the section summaries, character list, and official synopsis for "{title}", synthesize the complete story understanding.

Respond in JSON format:
{{
  "title": "{title}",
  "events": [
    {{
      "event_id": "evt-001",
      "description": "what happened",
      "characters": ["character names"],
      "timestamp_seconds": 0.0,
      "evidence_timestamps": [[0.0, 10.0]],
      "confidence": 0.95,
      "event_type": "conflict|turning_point|climax|resolution|setup|exposition"
    }}
  ],
  "major_conflict": "central conflict",
  "climax": "climax description",
  "resolution": "resolution description",
  "themes": ["themes"],
  "locations": ["locations"],
  "ambiguities": [],
  "one_sentence_summary": "single concise sentence"
}}"""

    response = gateway.invoke(
        task_name="story_synthesis",
        prompt=prompt,
        context=context,
    )

    result = response.parsed or {"title": title, "summary": response.content}

    # Merge character data (DETERMINISTIC assembly)
    if isinstance(characters, dict):
        result["characters"] = characters.get("characters", [])
        result["relationships"] = characters.get("relationships", [])
    else:
        result["characters"] = []
        result["relationships"] = []

    result["local_summaries"] = local_summaries

    return result
