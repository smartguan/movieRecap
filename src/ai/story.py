"""
Story understanding module (FR-3).

Implements hierarchical processing to minimize token usage:
1. Deterministically segment transcript and scene data (done upstream in index/).
2. Create bounded semantic summaries for local scene sequences via LLM.
3. Extract characters, events, and relationships via LLM.
4. Assemble the global story understanding (deterministic merge + LLM synthesis).

Only relevant summaries and evidence spans are sent to the LLM — never the
full transcript or all frames.
"""

import json
import logging
from typing import Any

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
    Build a complete story understanding from the scene index.

    Uses hierarchical processing:
    1. Chunk scenes into groups of CHUNK_SIZE
    2. Generate local summaries for each chunk (LLM)
    3. Extract characters and relationships (LLM)
    4. Synthesize global story understanding (LLM)

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

    # Step 1: Create local summaries for scene chunks (SEMANTIC)
    local_summaries = _create_local_summaries(gateway, scenes, title)

    # Step 2: Extract characters (SEMANTIC)
    characters = _extract_characters(gateway, local_summaries, title)

    # Step 3: Synthesize global story (SEMANTIC)
    story = _synthesize_story(gateway, local_summaries, characters, title)

    # Save the gateway telemetry
    story["telemetry"] = gateway.get_telemetry()

    return story


def _create_local_summaries(
    gateway: LLMGateway,
    scenes: list[dict[str, Any]],
    title: str,
) -> list[dict[str, Any]]:
    """
    Create bounded local summaries for chunks of scenes.

    Each chunk gets a summary via the LLM, using only the transcript text
    and basic metadata from that chunk — not the full movie.

    Args:
        gateway: LLM gateway instance.
        scenes: List of scene dicts from the scene index.
        title: Movie title for context.

    Returns:
        List of local summary dicts.
    """
    summaries = []
    chunks = [scenes[i : i + CHUNK_SIZE] for i in range(0, len(scenes), CHUNK_SIZE)]

    for chunk_idx, chunk in enumerate(chunks):
        # Build compact context from this chunk only
        context_lines = []
        for scene in chunk:
            time_range = (
                f"{scene['start_seconds']:.1f}s - {scene['end_seconds']:.1f}s"
            )
            transcript = scene.get("transcript_text", "").strip()
            if transcript:
                context_lines.append(f"[{time_range}] {transcript}")
            else:
                context_lines.append(f"[{time_range}] (no dialogue)")

        context = "\n".join(context_lines)

        prompt = f"""Analyze the following section (part {chunk_idx + 1} of {len(chunks)}) of the movie "{title}".

Based on the transcript below, identify:
1. Key events that happen in this section
2. Characters who appear and what they do
3. Important dialogue or revelations
4. The emotional tone of this section

Respond in JSON format:
{{
  "section_index": {chunk_idx},
  "time_range": "{chunk[0]['start_seconds']:.1f}s - {chunk[-1]['end_seconds']:.1f}s",
  "events": [
    {{
      "description": "what happened",
      "characters": ["character names"],
      "timestamp_seconds": 0.0,
      "event_type": "setup|conflict|turning_point|climax|resolution|exposition",
      "confidence": 0.0
    }}
  ],
  "characters_seen": ["list of character names in this section"],
  "emotional_tone": "description of the mood",
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

        summary = response.parsed or {"section_index": chunk_idx, "summary": response.content}
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
) -> list[dict[str, Any]]:
    """
    Extract and consolidate characters from local summaries.

    Args:
        gateway: LLM gateway instance.
        local_summaries: Local summaries from each chunk.
        title: Movie title.

    Returns:
        List of character dicts with names, aliases, and relationships.
    """
    # Collect all character mentions from local summaries (DETERMINISTIC)
    all_characters = set()
    for summary in local_summaries:
        for char in summary.get("characters_seen", []):
            all_characters.add(char)

    # Prepare compact context — only character-relevant info
    context_lines = []
    for summary in local_summaries:
        section_summary = summary.get("summary", "")
        chars = summary.get("characters_seen", [])
        if chars:
            context_lines.append(
                f"Section {summary.get('section_index', '?')}: "
                f"Characters: {', '.join(chars)}. {section_summary}"
            )

    context = "\n".join(context_lines)

    prompt = f"""Given the following section summaries from the movie "{title}", create a character registry.

For each character, provide:
- Their main name
- Any aliases or alternate names used
- A brief description of who they are
- Their first appearance (approximate timestamp)
- Key relationships with other characters

Respond in JSON format:
{{
  "characters": [
    {{
      "name": "main name",
      "aliases": ["other names used"],
      "description": "who they are and their role",
      "first_appearance_seconds": 0.0
    }}
  ],
  "relationships": [
    {{
      "character_a": "name",
      "character_b": "name",
      "relationship_type": "type (spouse, rival, colleague, etc.)",
      "description": "nature of the relationship"
    }}
  ]
}}"""

    response = gateway.invoke(
        task_name="character_extraction",
        prompt=prompt,
        context=context,
    )

    return response.parsed or {"characters": [], "relationships": []}


def _synthesize_story(
    gateway: LLMGateway,
    local_summaries: list[dict[str, Any]],
    characters: Any,
    title: str,
) -> dict[str, Any]:
    """
    Synthesize a global story understanding from local summaries and characters.

    Args:
        gateway: LLM gateway instance.
        local_summaries: All local summaries.
        characters: Character registry.
        title: Movie title.

    Returns:
        Complete story understanding dict.
    """
    # Build compact context from summaries (NOT full transcript)
    summary_lines = []
    all_events = []
    for summary in local_summaries:
        time_range = summary.get("time_range", "?")
        section_summary = summary.get("summary", "")
        summary_lines.append(f"[{time_range}] {section_summary}")

        for event in summary.get("events", []):
            all_events.append(event)

    context = (
        f"CHARACTER REGISTRY:\n{json.dumps(characters, ensure_ascii=False, indent=2)}\n\n"
        f"SECTION SUMMARIES:\n" + "\n".join(summary_lines)
    )

    prompt = f"""Based on the section summaries and character registry for the movie "{title}", create a complete story understanding.

Provide:
1. A chronological event timeline with the most important events
2. The major conflict and its development
3. The climax and resolution
4. Key themes and takeaways
5. Any ambiguities or uncertain elements

Respond in JSON format:
{{
  "title": "{title}",
  "events": [
    {{
      "event_id": "evt-001",
      "description": "what happened",
      "characters": ["character names"],
      "timestamp_seconds": 0.0,
      "evidence_timestamps": [[start, end]],
      "confidence": 0.0,
      "event_type": "conflict|turning_point|climax|resolution|setup|exposition"
    }}
  ],
  "major_conflict": "description of the central conflict",
  "climax": "description of the climax",
  "resolution": "how it resolves",
  "themes": ["key themes"],
  "locations": ["main locations"],
  "ambiguities": ["things that are unclear"],
  "one_sentence_summary": "single sentence describing the movie"
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
