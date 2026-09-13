"""
Script generation module (FR-4).

Generates a Mandarin narration script from the story understanding and scene index.
Optimized for minimal LLM token consumption via high-density event serializations:
- Compact bullet event formatting (saving ~60% prompt tokens vs raw indented JSON)
- Leverages fetcher metadata (synopsis, genre) for contextual grounding
- Pure clean narration output sanitizer for TTS audio synthesis
"""

from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.ai.gateway import LLMGateway
from src.ai.tasks import register_all_tasks

logger = logging.getLogger(__name__)


def generate_script(
    project: dict[str, Any],
    story: dict[str, Any],
    scene_index: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate the full narration script.

    Process:
    1. Generate opening hook (SEMANTIC)
    2. Generate body segments following the event timeline (SEMANTIC with compact token format)
    3. Generate conclusion (SEMANTIC)
    4. Assemble into complete script (DETERMINISTIC)

    Args:
        project: Project data dict.
        story: Story understanding dict.
        scene_index: Scene index with timestamps and transcripts.
        config: System configuration.

    Returns:
        Complete script dict with segments, metadata, and telemetry.
    """
    gateway = LLMGateway(config)
    register_all_tasks(gateway)

    title = project.get("title", "Unknown Movie")
    metadata = project.get("metadata", {})
    target_min, target_max = project.get("target_duration_range", [20, 30])
    speaking_rate = 250.0  # chars per minute for Mandarin
    min_chars = int(target_min * speaking_rate)
    max_chars = int(target_max * speaking_rate)

    segments: list[dict[str, Any]] = []
    segment_counter = 0

    # 1. Generate opening hook (SEMANTIC)
    hook = _generate_hook(gateway, story, title, metadata)
    hook["segment_id"] = f"narration-{segment_counter:03d}"
    hook["segment_type"] = "hook"
    segments.append(hook)
    segment_counter += 1

    # 2. Generate body segments (SEMANTIC with token-optimized context)
    events = story.get("events", [])
    characters = story.get("characters", [])

    # Process events in groups to maintain narrative flow
    event_groups = _group_events(events, max_group_size=5)

    for group_idx, event_group in enumerate(event_groups):
        body_segments = _generate_body_segment(
            gateway,
            event_group,
            story,
            scene_index,
            title,
            characters,
            group_idx,
            len(event_groups),
            metadata,
        )

        for seg in body_segments:
            seg["segment_id"] = f"narration-{segment_counter:03d}"
            segments.append(seg)
            segment_counter += 1

    # 3. Generate conclusion (SEMANTIC)
    conclusion = _generate_conclusion(gateway, story, title, metadata)
    conclusion["segment_id"] = f"narration-{segment_counter:03d}"
    conclusion["segment_type"] = "conclusion"
    segments.append(conclusion)

    # 4. Assemble script (DETERMINISTIC)
    total_chars = sum(len(s.get("text", "")) for s in segments)
    estimated_duration = total_chars / speaking_rate

    script = {
        "project_id": project.get("project_id", ""),
        "title": title,
        "segments": segments,
        "total_characters": total_chars,
        "estimated_duration_minutes": round(estimated_duration, 1),
        "target_speaking_rate": speaking_rate,
        "target_range_minutes": [target_min, target_max],
        "version": 1,
        "telemetry": gateway.get_telemetry(),
    }

    logger.info(
        "Script generated: %d segments, %d chars, ~%.1f min",
        len(segments),
        total_chars,
        estimated_duration,
    )

    return script


def clean_narration_text(text: str) -> str:
    """
    Clean and sanitize narration text to ensure pure human speech.
    Strips raw JSON syntax, quotes, keys, markdown formatting, and symbols
    that TTS engines might synthesize as punctuation words.
    """
    if not text:
        return ""

    text = text.strip()

    # If the text is wrapped in JSON or contains JSON structure, attempt to parse it
    if text.startswith("{") or text.startswith("[") or '"text":' in text or '"segments":' in text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                if "text" in parsed and isinstance(parsed["text"], str):
                    text = parsed["text"]
                elif "segments" in parsed and isinstance(parsed["segments"], list):
                    text = " ".join(
                        s.get("text", "") for s in parsed["segments"] if isinstance(s, dict) and s.get("text")
                    )
            elif isinstance(parsed, list):
                text = " ".join(
                    s.get("text", "") if isinstance(s, dict) else str(s) for s in parsed
                )
        except Exception:
            pass

    # Remove markdown code fences and backticks
    text = re.sub(r"```[\w]*\n?", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"`", "", text)

    # If JSON keys like "text": "...", "confidence": ..., "supporting_scenes": ... are present
    if '"text":' in text or "'text':" in text:
        matches = re.findall(r'["\']text["\']\s*:\s*["\']([^"\']+)["\']', text)
        if matches:
            text = " ".join(matches)

    # Remove structural JSON keys and metadata lines if any remain
    text = re.sub(
        r'["\']?(supporting_scenes|start_seconds|end_seconds|confidence|segment_type|segments)["\']?\s*:\s*[^,\n]+',
        "",
        text,
    )

    # Remove structural brackets, braces, and escape sequences
    text = re.sub(r"[{}\[\]]", "", text)
    text = text.replace('\\"', '"').replace("\\n", " ")

    # Remove stray quotes, colons, underscores that cause TTS to read "下划线" or "引号"
    text = text.replace('"', "").replace("'", "").replace("_", " ")

    # Clean up double spaces or weird whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _generate_hook(
    gateway: LLMGateway,
    story: dict[str, Any],
    title: str,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Generate the opening hook segment (30-60 seconds, ~125-250 chars)."""
    one_sentence = story.get("one_sentence_summary", "")
    climax = story.get("climax", "")
    themes = story.get("themes", [])
    meta = metadata or {}
    synopsis = meta.get("synopsis", "")

    context_parts = [
        f"Movie: {title}",
        f"Summary: {one_sentence or synopsis}",
    ]
    if climax:
        context_parts.append(f"Climax: {climax}")
    if themes:
        context_parts.append(f"Themes: {', '.join(themes)}")

    context = "\n".join(context_parts)

    prompt = f"""Write an engaging 30-60 second opening hook in Mandarin (中文) for a movie commentary video about "{title}".

The hook should:
- Immediately capture the viewer's attention
- Hint at the most compelling aspect of the story WITHOUT revealing the ending
- Set up why this movie is worth watching/discussing
- Be 125-250 Chinese characters

Respond in JSON:
{{
  "text": "the hook narration in Chinese",
  "supporting_scenes": [
    {{"start_seconds": 0.0, "end_seconds": 10.0}}
  ],
  "confidence": 0.95
}}"""

    response = gateway.invoke(
        task_name="hook_generation",
        prompt=prompt,
        context=context,
    )

    parsed = response.parsed
    if isinstance(parsed, dict):
        raw_text = parsed.get("text", "")
        scenes = parsed.get("supporting_scenes", [])
        conf = parsed.get("confidence", 0.9)
    elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
        raw_text = parsed[0].get("text", "")
        scenes = parsed[0].get("supporting_scenes", [])
        conf = parsed[0].get("confidence", 0.9)
    else:
        raw_text = response.content
        scenes = []
        conf = 0.5

    clean_text = clean_narration_text(raw_text)
    return {
        "text": clean_text,
        "supporting_scenes": scenes if isinstance(scenes, list) else [],
        "confidence": conf,
    }


def _generate_body_segment(
    gateway: LLMGateway,
    events: list[dict[str, Any]],
    story: dict[str, Any],
    scene_index: dict[str, Any],
    title: str,
    characters: list[dict[str, Any]],
    group_idx: int,
    total_groups: int,
    metadata: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Generate narration segments using compact token-dense event formatting."""
    # Token-optimized compact event list instead of raw indented JSON
    event_lines = []
    for i, e in enumerate(events):
        ts = e.get("timestamp_seconds", 0.0)
        desc = e.get("description", "")
        etype = e.get("event_type", "plot")
        chars = ", ".join(e.get("characters", []))
        event_lines.append(
            f"- [t={ts:.1f}s] ({etype}) {desc} (Characters: {chars or 'N/A'})"
        )
    events_compact = "\n".join(event_lines)

    # Compact character string
    char_strs = [
        f"{c.get('name')}" + (f" ({c.get('description')})" if c.get('description') else "")
        for c in characters[:8]  # Limit to key characters
    ]
    chars_compact = ", ".join(char_strs) if char_strs else "Main Characters"

    prompt = f"""Write narration segments in Mandarin (中文) for part {group_idx + 1}/{total_groups} of "{title}".

Cover these story events with:
- Clear, condensed storytelling
- Original commentary and insights (NOT just plot summary)
- Natural transitions between events
- Each segment should be 200-400 Chinese characters

Respond in JSON:
{{
  "segments": [
    {{
      "text": "narration in Chinese with commentary",
      "supporting_scenes": [
        {{"start_seconds": 0.0, "end_seconds": 10.0}}
      ],
      "confidence": 0.9,
      "segment_type": "plot_and_commentary"
    }}
  ]
}}"""

    context = f"""EVENTS:\n{events_compact}\n\nCHARACTERS: {chars_compact}"""

    evidence_refs = [
        e.get("event_id", f"event-{i}") for i, e in enumerate(events)
    ]

    response = gateway.invoke(
        task_name="script_generation",
        prompt=prompt,
        context=context,
        evidence_refs=evidence_refs,
    )

    parsed = response.parsed
    results: list[dict[str, Any]] = []

    if isinstance(parsed, dict) and "segments" in parsed and isinstance(parsed["segments"], list):
        for s in parsed["segments"]:
            if isinstance(s, dict):
                clean_text = clean_narration_text(s.get("text", ""))
                if clean_text:
                    results.append({
                        "text": clean_text,
                        "supporting_scenes": s.get("supporting_scenes", []),
                        "confidence": s.get("confidence", 0.9),
                        "segment_type": s.get("segment_type", "plot_and_commentary"),
                    })
    elif isinstance(parsed, list):
        for s in parsed:
            if isinstance(s, dict):
                clean_text = clean_narration_text(s.get("text", ""))
                if clean_text:
                    results.append({
                        "text": clean_text,
                        "supporting_scenes": s.get("supporting_scenes", []),
                        "confidence": s.get("confidence", 0.9),
                        "segment_type": s.get("segment_type", "plot_and_commentary"),
                    })
    elif isinstance(parsed, dict) and "text" in parsed:
        clean_text = clean_narration_text(parsed.get("text", ""))
        if clean_text:
            results.append({
                "text": clean_text,
                "supporting_scenes": parsed.get("supporting_scenes", []),
                "confidence": parsed.get("confidence", 0.9),
                "segment_type": parsed.get("segment_type", "plot_and_commentary"),
            })

    if not results:
        clean_text = clean_narration_text(response.content)
        results = [{
            "text": clean_text,
            "supporting_scenes": [],
            "confidence": 0.5,
            "segment_type": "plot_and_commentary",
        }]

    return results


def _generate_conclusion(
    gateway: LLMGateway,
    story: dict[str, Any],
    title: str,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Generate the concluding segment with assessment and interpretation."""
    resolution = story.get("resolution", "")
    themes = story.get("themes", [])
    one_sentence = story.get("one_sentence_summary", "")

    context = f"""Movie: {title}
Resolution: {resolution}
Themes: {', '.join(themes) if themes else 'N/A'}
Summary: {one_sentence}"""

    prompt = f"""Write a concluding segment in Mandarin (中文) for "{title}".

The conclusion should:
- Wrap up the story
- Provide commentary and analysis
- Be 200-400 Chinese characters

Respond in JSON:
{{
  "text": "concluding narration in Chinese with analysis",
  "supporting_scenes": [],
  "confidence": 0.9
}}"""

    response = gateway.invoke(
        task_name="conclusion_generation",
        prompt=prompt,
        context=context,
    )

    parsed = response.parsed
    if isinstance(parsed, dict):
        raw_text = parsed.get("text", "")
        scenes = parsed.get("supporting_scenes", [])
        conf = parsed.get("confidence", 0.9)
    elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
        raw_text = parsed[0].get("text", "")
        scenes = parsed[0].get("supporting_scenes", [])
        conf = parsed[0].get("confidence", 0.9)
    else:
        raw_text = response.content
        scenes = []
        conf = 0.5

    clean_text = clean_narration_text(raw_text)
    return {
        "text": clean_text,
        "supporting_scenes": scenes if isinstance(scenes, list) else [],
        "confidence": conf,
    }


def _group_events(
    events: list[dict[str, Any]], max_group_size: int = 5
) -> list[list[dict[str, Any]]]:
    """
    Group events into chunks for sequential processing.

    DETERMINISTIC: Simple list chunking.
    """
    if not events:
        return [[]]
    return [
        events[i : i + max_group_size]
        for i in range(0, len(events), max_group_size)
    ]
