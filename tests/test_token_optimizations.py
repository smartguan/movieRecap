"""
Unit tests for LLM Token Usage Optimizations and Video Fetcher Metadata Synergy.
"""

from __future__ import annotations
import json
from unittest.mock import MagicMock, patch

import pytest
from src.acquirer.models import VideoMetadata
from src.ai.gateway import LLMGateway, GatewayResponse
from src.ai.script import _generate_body_segment, _generate_hook, clean_narration_text
from src.ai.story import _create_local_summaries, _extract_characters, _synthesize_story


def test_video_metadata_token_optimized_summary() -> None:
    meta = VideoMetadata(
        title="微风襟袖同卿心",
        source_url="https://www.yfsp.tv/play/zqBWB3mQYiB",
        genre="爱情,古装",
        year="2026",
        stars=["范梦", "张植绿"],
        directors=["张亚海"],
        synopsis="本剧讲述李度与杨策的故事。",
    )
    summary = meta.to_token_optimized_summary()
    assert "Title: 微风襟袖同卿心" in summary
    assert "Genre: 爱情,古装" in summary
    assert "Cast: 范梦, 张植绿" in summary
    assert "Director: 张亚海" in summary
    assert "Synopsis: 本剧讲述李度与杨策的故事。" in summary


def test_silent_scene_chunk_skips_llm_call() -> None:
    """Ensure dialogue-free scene chunks produce deterministic summaries with 0 LLM calls."""
    mock_gateway = MagicMock(spec=LLMGateway)

    # 15 scenes with NO transcript
    silent_scenes = [
        {"scene_id": f"scene-{i:03d}", "start_seconds": float(i * 2), "end_seconds": float(i * 2 + 2), "transcript_text": ""}
        for i in range(15)
    ]

    summaries = _create_local_summaries(
        gateway=mock_gateway,
        scenes=silent_scenes,
        title="Test Movie",
        metadata={"synopsis": "A movie"},
    )

    # Gateway invoke should NOT be called at all
    mock_gateway.invoke.assert_not_called()
    assert len(summaries) == 1
    assert "无对白" in summaries[0]["summary"] or "过渡" in summaries[0]["summary"]
    assert summaries[0]["section_index"] == 0


def test_character_seeding_from_fetcher_metadata() -> None:
    """Ensure character registry is seeded deterministically from fetcher stars without LLM call when no dialogue characters exist."""
    mock_gateway = MagicMock(spec=LLMGateway)

    local_summaries = [
        {"section_index": 0, "summary": "过渡镜头", "characters_seen": []}
    ]

    metadata = {
        "stars": ["范梦", "张植绿"],
        "synopsis": "李度与杨策相伴成长",
    }

    result = _extract_characters(
        gateway=mock_gateway,
        local_summaries=local_summaries,
        title="微风襟袖同卿心",
        metadata=metadata,
    )

    # Gateway should not be called because all characters come directly from fetcher stars
    mock_gateway.invoke.assert_not_called()
    assert len(result["characters"]) == 2
    names = [c["name"] for c in result["characters"]]
    assert "范梦" in names
    assert "张植绿" in names


def test_compact_body_segment_prompt() -> None:
    """Verify body segment prompt uses compact event listing."""
    mock_gateway = MagicMock(spec=LLMGateway)
    mock_gateway.invoke.return_value = GatewayResponse(
        content='{"segments": [{"text": "李度在杨策的鼓励下奋发图强。", "supporting_scenes": [], "confidence": 0.95}]}',
        parsed={"segments": [{"text": "李度在杨策的鼓励下奋发图强。", "supporting_scenes": [], "confidence": 0.95}]},
        input_tokens=150,
        output_tokens=60,
    )

    events = [
        {
            "event_id": "evt-1",
            "timestamp_seconds": 10.5,
            "event_type": "conflict",
            "description": "夫妻发生争执后和解",
            "characters": ["李度", "杨策"],
        }
    ]

    segments = _generate_body_segment(
        gateway=mock_gateway,
        events=events,
        story={"characters": [{"name": "李度"}, {"name": "杨策"}]},
        scene_index={"scenes": []},
        title="微风襟袖同卿心",
        characters=[{"name": "李度"}, {"name": "杨策"}],
        group_idx=0,
        total_groups=1,
        metadata={"synopsis": "夫妻成长"},
    )

    mock_gateway.invoke.assert_called_once()
    call_args = mock_gateway.invoke.call_args[1]
    # Check that context uses compact bullet format, NOT bulky indented JSON
    context = call_args["context"]
    assert "[t=10.5s] (conflict) 夫妻发生争执后和解" in context
    assert len(segments) == 1
    assert segments[0]["text"] == "李度在杨策的鼓励下奋发图强。"
