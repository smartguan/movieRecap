"""Tests for deterministic Scene Semantic Matcher (ADR 0005)."""
from __future__ import annotations

import pytest
from src.media.scene_matcher import (
    CONCEPT_MAP,
    SemanticSceneIndex,
    build_semantic_scene_index,
    find_best_scenes,
    score_scene_for_narration,
    tokenize_text,
)


def test_tokenize_text_cjk_and_alphanumeric():
    text = "2024年，雪乃在东京理发店使用SNS发帖！"
    tokens = tokenize_text(text)

    # Check alphanumeric
    assert "2024" in tokens
    assert "sns" in tokens
    assert "SNS" in tokens

    # Check unigrams & bigrams
    assert "雪" in tokens
    assert "乃" in tokens
    assert "雪乃" in tokens
    assert "理发" in tokens
    assert "发店" in tokens
    assert "东京" in tokens


def test_concept_map_coverage():
    assert "理发" in CONCEPT_MAP
    assert "カット" in CONCEPT_MAP["理发"]
    assert "台湾" in CONCEPT_MAP
    assert "浴缸" in CONCEPT_MAP
    assert "風呂" in CONCEPT_MAP["浴缸"]
    assert "神庙" in CONCEPT_MAP
    assert "纸人" in CONCEPT_MAP


def test_semantic_scene_index_context_expansion():
    scene_index = {
        "duration_seconds": 300.0,
        "scenes": [
            {"scene_id": "sc-001", "start_seconds": 0.0, "end_seconds": 10.0, "transcript_text": "いらっしゃいませ"},
            {"scene_id": "sc-002", "start_seconds": 10.0, "end_seconds": 20.0, "transcript_text": "カットとカラーで"},
            {"scene_id": "sc-003", "start_seconds": 20.0, "end_seconds": 30.0, "transcript_text": "かしこまりました"},
        ],
    }
    story_understanding = {
        "local_summaries": [
            {"section_index": 0, "summary": "顾客来到东京理发店进行理发剪发。", "events": []}
        ]
    }

    index = build_semantic_scene_index(scene_index, story_understanding)
    assert len(index.scenes) == 3

    # Reaction scene sc-003 should have "カット" in its expanded context window
    sc3 = index.scenes[2]
    assert any("カット" in t for t in sc3.context_tokens)


def test_score_scene_for_narration_concept_and_transcript():
    scene_index = {
        "duration_seconds": 3000.0,
        "scenes": [
            {"scene_id": "sc-salon", "start_seconds": 200.0, "end_seconds": 230.0, "transcript_text": "カットとカラーでお願いします"},
            {"scene_id": "sc-random", "start_seconds": 1500.0, "end_seconds": 1530.0, "transcript_text": "何でもないです"},
        ],
    }
    index = build_semantic_scene_index(scene_index)

    narration_haircut = "故事从东京一家静谧的理发店拉开帷幕，女主为顾客修剪发丝。"
    
    score_salon = score_scene_for_narration(
        scene=index.scenes[0],
        narration_text=narration_haircut,
        target_timestamp=200.0,
        total_duration=3000.0,
    )
    score_random = score_scene_for_narration(
        scene=index.scenes[1],
        narration_text=narration_haircut,
        target_timestamp=1500.0,
        total_duration=3000.0,
    )

    # Hair salon scene should receive a substantially higher affinity score
    assert score_salon > 0.40
    assert score_salon > score_random + 0.20


def test_find_best_scenes_ranking_and_spacing():
    scene_index = {
        "duration_seconds": 5000.0,
        "scenes": [
            {"scene_id": f"s-{i}", "start_seconds": float(i * 100), "end_seconds": float(i * 100 + 50), "transcript_text": ""}
            for i in range(50)
        ],
    }
    # Add a scene with specific dialogue in Taiwan
    scene_index["scenes"][26]["transcript_text"] = "私も台湾に行く"

    index = build_semantic_scene_index(scene_index)

    narration_taiwan = "为了追寻真相，女主背起行囊登上了飞往台北的航班，决定前往台湾。"
    candidates = find_best_scenes(
        semantic_index=index,
        narration_text=narration_taiwan,
        target_timestamp=2500.0,
        needed_duration=10.0,
        used_start_timestamps=[],
        top_k=3,
    )

    assert len(candidates) > 0
    top_scene, top_score = candidates[0]
    # Should identify scene at 2600s with Taiwan dialogue
    assert top_scene.scene_id == "s-26"
    assert top_score > 0.35

    # If s-26 is in used_start_timestamps (within 8s), it should pick an alternative
    candidates_spaced = find_best_scenes(
        semantic_index=index,
        narration_text=narration_taiwan,
        target_timestamp=2500.0,
        needed_duration=10.0,
        used_start_timestamps=[2600.0],
        min_spacing=8.0,
        top_k=1,
    )
    assert candidates_spaced[0][0].scene_id != "s-26"
