"""
Script generation module (FR-4).

Generates a Mandarin narration script from the story understanding and scene index.
The script contains:
- A 30-60 second opening hook
- Clear, condensed story progression
- Original transitions and editorial voice
- Interpretation, reaction, cultural context, criticism, humor, or analysis
- A concluding assessment

Target: ~6,000-8,000 Chinese characters for 20-30 minutes at ~250 chars/min.

Each segment references supporting source scenes for evidence grounding.
"""

from __future__ import annotations
import json
import logging
from typing import Any

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
    2. Generate body segments following the event timeline (SEMANTIC)
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
    target_min, target_max = project.get("target_duration_range", [20, 30])
    speaking_rate = 250.0  # chars per minute for Mandarin
    min_chars = int(target_min * speaking_rate)
    max_chars = int(target_max * speaking_rate)

    segments: list[dict[str, Any]] = []
    segment_counter = 0

    # 1. Generate opening hook (SEMANTIC)
    hook = _generate_hook(gateway, story, title)
    hook["segment_id"] = f"narration-{segment_counter:03d}"
    hook["segment_type"] = "hook"
    segments.append(hook)
    segment_counter += 1

    # 2. Generate body segments (SEMANTIC)
    events = story.get("events", [])
    characters_info = json.dumps(
        story.get("characters", []), ensure_ascii=False
    )

    # Process events in groups to maintain narrative flow
    event_groups = _group_events(events, max_group_size=5)

    for group_idx, event_group in enumerate(event_groups):
        body_segments = _generate_body_segment(
            gateway,
            event_group,
            story,
            scene_index,
            title,
            characters_info,
            group_idx,
            len(event_groups),
        )

        for seg in body_segments:
            seg["segment_id"] = f"narration-{segment_counter:03d}"
            segments.append(seg)
            segment_counter += 1

    # 3. Generate conclusion (SEMANTIC)
    conclusion = _generate_conclusion(gateway, story, title)
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

    # Check if duration is in range (DETERMINISTIC)
    if estimated_duration < target_min:
        logger.warning(
            "Script is too short: %.1f min < %d min target",
            estimated_duration,
            target_min,
        )
    elif estimated_duration > target_max:
        logger.warning(
            "Script is too long: %.1f min > %d min target",
            estimated_duration,
            target_max,
        )

    return script


def clean_narration_text(text: str) -> str:
    """
    Clean and sanitize narration text to ensure pure human speech.
    Strips raw JSON syntax, quotes, keys, markdown formatting, and symbols
    that TTS engines might synthesize as punctuation words.
    """
    import re

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
) -> dict[str, Any]:
    """Generate the opening hook segment (30-60 seconds, ~125-250 chars)."""
    one_sentence = story.get("one_sentence_summary", "")
    climax = story.get("climax", "")
    themes = story.get("themes", [])

    context = f"""Movie: {title}
Summary: {one_sentence}
Climax: {climax}
Themes: {', '.join(themes) if themes else 'N/A'}"""

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
    {{"start_seconds": 0.0, "end_seconds": 0.0}}
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
    characters_info: str,
    group_idx: int,
    total_groups: int,
) -> list[dict[str, Any]]:
    """Generate narration segments for a group of story events."""
    events_context = json.dumps(events, ensure_ascii=False, indent=2)

    prompt = f"""Write narration segments in Mandarin (中文) for part {group_idx + 1} of {total_groups} of the movie commentary for "{title}".

Cover these story events with:
- Clear, condensed storytelling
- Original commentary and insights (NOT just plot summary)
- Natural transitions between events
- Cultural context or analysis where appropriate
- Each segment should be 200-400 Chinese characters

For each narration segment, include the source timestamps that support it.

IMPORTANT: This is a commentary video, not a transcript. Add your own perspective, reactions, and analysis.

Respond in JSON:
{{
  "segments": [
    {{
      "text": "narration in Chinese with commentary",
      "supporting_scenes": [
        {{"start_seconds": 0.0, "end_seconds": 0.0}}
      ],
      "confidence": 0.9,
      "segment_type": "plot_and_commentary"
    }}
  ]
}}"""

    context = f"""EVENTS:\n{events_context}\n\nCHARACTERS:\n{characters_info}"""

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
) -> dict[str, Any]:
    """Generate the concluding segment with assessment and interpretation."""
    resolution = story.get("resolution", "")
    themes = story.get("themes", [])
    one_sentence = story.get("one_sentence_summary", "")

    context = f"""Movie: {title}
Resolution: {resolution}
Themes: {', '.join(themes) if themes else 'N/A'}
Summary: {one_sentence}"""

    prompt = f"""Write a concluding segment in Mandarin (中文) for the movie commentary video about "{title}".

The conclusion should:
- Wrap up the story
- Provide your assessment and interpretation of the film
- Discuss what makes this movie notable or worth watching
- Include a viewing recommendation
- Be 300-500 Chinese characters

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
