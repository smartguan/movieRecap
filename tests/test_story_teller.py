"""Tests for StoryTeller zero-repetition narrative synthesis engine."""
import pytest
from src.ai.story_teller import StoryTeller
from src.eval.cliches import calculate_lexical_diversity, find_repeated_phrases, scan_for_cliches
from src.eval.slop_detector import SlopDetector


def test_story_teller_zero_repetition_and_diversity():
    teller = StoryTeller()
    title = "诅咒"
    synopsis = "女子发现早已离世的台湾友人社交账号还在不断更新。只要留言、点开诡异视频就会沾染上纸人形诅咒，死亡顺着网络四处扩散。为追查真相，女主跨海奔赴台湾，红衣怨灵如影随形，在民俗诅咒与网络流言裹挟下，揭开这场夺命诅咒的残酷根源。"
    cast = ["海津雪乃", "杨宇腾", "邵奕玫", "诗步"]
    genre = "剧情"

    segments = teller.generate_full_recap(
        title=title,
        synopsis=synopsis,
        cast=cast,
        genre=genre,
        total_duration_sec=5673.7,
        target_duration_min=18.9,
    )

    assert len(segments) >= 15
    full_text = " ".join(s["text"] for s in segments)

    # 1. Lexical diversity check (natural Mandarin benchmarks)
    metrics = calculate_lexical_diversity(full_text)
    assert metrics["ttr"] >= 0.30
    assert metrics["distinct_2"] >= 0.80
    assert metrics["distinct_3"] >= 0.90

    # 2. No repeated boilerplate 6-grams >= 3 times
    repeated_6grams = find_repeated_phrases(full_text, ngram_size=6, min_count=3)
    assert len(repeated_6grams) == 0, f"Found repetitive 6-grams: {repeated_6grams}"

    # 3. No formulaic template phrases
    forbidden_templates = [
        "随着剧情推进至",
        "画面中展现出关键情节",
        "这一刻不仅交代了人物的真实动机",
        "剧情的张力在这一刻被彻底拉满",
    ]
    for seg in segments:
        for t in forbidden_templates:
            assert t not in seg["text"], f"Forbidden template '{t}' found in segment: {seg['text']}"

    # 4. Monotonic chronological progression
    start_times = [s["supporting_scenes"][0]["start_seconds"] for s in segments if s.get("supporting_scenes")]
    for i in range(1, len(start_times)):
        assert start_times[i] >= start_times[i - 1], f"Non-monotonic timestamps: {start_times[i-1]} -> {start_times[i]}"


def test_story_teller_multi_movie_generalization():
    teller = StoryTeller()
    test_movies = [
        {
            "title": "异星觉醒",
            "synopsis": "科学家在火星样本中发现休眠的单细胞地外生命，随着实验唤醒，该生物以惊人速度进化并展现出残暴的攻击性，空间站宇航员面临灭顶之灾。",
            "cast": ["杰克·吉伦哈尔", "丽贝卡·弗格森", "瑞安·雷诺兹"],
            "genre": "科幻/惊悚",
            "duration": 6200.0,
            "target_recap": 12.0,
        },
        {
            "title": "消失的爱人",
            "synopsis": "结婚五周年纪念日当天，妻子离奇失踪，家中留下激烈的搏斗痕迹，随着警方调查深入，所有线索与日记竟全部指向丈夫是冷酷的谋杀犯。",
            "cast": ["本·阿弗莱克", "罗莎蒙德·派克", "尼尔·帕特里克·哈里斯"],
            "genre": "悬疑/剧情",
            "duration": 8900.0,
            "target_recap": 15.0,
        }
    ]

    for m in test_movies:
        segments = teller.generate_full_recap(
            title=m["title"],
            synopsis=m["synopsis"],
            cast=m["cast"],
            genre=m["genre"],
            total_duration_sec=m["duration"],
            target_duration_min=m["target_recap"],
        )
        assert len(segments) >= 10
        full_text = " ".join(s["text"] for s in segments)
        assert m["title"] in full_text
        for lead in m["cast"][:1]:
            assert lead in full_text or "主角" in full_text or "女主" in full_text
