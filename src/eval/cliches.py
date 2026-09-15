"""
Deterministic Cliché Matcher and Lexical Diversity Analysis (FR-Eval).

Detects notorious "AI Slop" patterns, formulaic templates, repetitive phrases,
and calculates lexical diversity metrics (TTR, Distinct-N) on Mandarin text.
"""

from __future__ import annotations
import re
from collections import Counter
from typing import Any

# Known AI Slop Clichés & Stereotypical Formula Patterns in Chinese Commentary
AI_SLOP_CLICHES = [
    # Generic opening/hook fillers
    r"在这个充满.+的世界(?:里|中)",
    r"故事的开头",
    r"今天(?:我们)?要给大家(?:带来|讲述|解构)的",
    r"接下来发生的事情",
    r"究竟会发生什么(?:呢)?[，,。]让我们拭目以待",
    
    # Generic transition fillers
    r"不得不说",
    r"事情并没有那么简单",
    r"然而好景不长",
    r"镜头一转",
    r"紧接着",
    r"随后",
    r"值得一提的是",
    r"众所周知",
    r"显而易见",
    r"让人意想不到的是",
    
    # Generic passive plot descriptions
    r"男主接下来做出了一个惊人的决定",
    r"所有人都惊呆了",
    r"事情的发展超出了所有人的预料",
    
    # Generic moralizing conclusions
    r"总而言之",
    r"总的来说",
    r"这部电影告诉我们(?:一个道理)?",
    r"看完让人深思",
    r"引发了人们对.+的深思",
]

# Ungrounded Viewer Comments & Film-School Meta-Jargon
UNGROUNDED_META_PATTERNS = [
    # Audience & Fourth-Wall Directives
    r"你会选择.+还是.+",
    r"如果你遇到.+你会.+",
    r"今天深度解说",
    r"今天给大家讲述",
    r"留给观众的",
    r"让观众(?:感到|体会|沉浸)",
    r"把观众拉入",
    r"屏幕前的你",
    r"各位观众",

    # Film-making & Director Meta-Commentary
    r"导演巧妙地",
    r"导演以.+镜头",
    r"快节奏的蒙太奇",
    r"蒙太奇(?:手法)?",
    r"视听交互",
    r"逼仄的景别",
    r"生活流镜头",
    r"镜头缓缓拉开",
    r"镜头在此处克制",
    r"摄影机",
    r"服化道",
    r"演员的表现",

    # Rhetorical Theatrical Fluff & Moralizing Jargon
    r"折射出.+微妙距离感",
    r"谁也没有料到.+暴风雨来临前最后的安宁",
    r"尖锐地刺破了.+的阴暗角落",
    r"提升到了对.+的深度探讨",
    r"迎来命运的终极对抗",
]


def scan_for_cliches(text: str) -> list[str]:
    """
    Scan Mandarin narration text for known AI Slop cliché formulas.

    Returns:
        List of matched cliché strings or descriptions.
    """
    if not text:
        return []

    matched = []
    for pattern in AI_SLOP_CLICHES:
        found = re.findall(pattern, text)
        if found:
            for item in found:
                matched.append(item if isinstance(item, str) else str(item))
    return matched


def scan_for_meta_commentary(text: str) -> list[str]:
    """
    Scan Mandarin narration text for ungrounded viewer meta-commentary,
    fourth-wall breaks, film-school jargon, and rhetorical theatrical fluff.

    Returns:
        List of matched meta-commentary strings or descriptions.
    """
    if not text:
        return []

    matched = []
    for pattern in UNGROUNDED_META_PATTERNS:
        found = re.findall(pattern, text)
        if found:
            for item in found:
                matched.append(item if isinstance(item, str) else str(item))
    return matched



def calculate_lexical_diversity(text: str) -> dict[str, float]:
    """
    Calculate lexical diversity metrics on Mandarin text.

    Metrics:
    - TTR (Type-Token Ratio): Unique characters / Total characters
    - Distinct-1: Unique 1-grams / Total 1-grams
    - Distinct-2: Unique 2-grams / Total 2-grams
    - Distinct-3: Unique 3-grams / Total 3-grams

    Returns:
        Dict with 'ttr', 'distinct_1', 'distinct_2', 'distinct_3'.
    """
    # Clean text to Chinese characters and punctuation
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return {"ttr": 1.0, "distinct_1": 1.0, "distinct_2": 1.0, "distinct_3": 1.0}

    total_chars = len(chars)
    unique_chars = len(set(chars))
    ttr = unique_chars / total_chars

    # 2-grams
    if total_chars >= 2:
        bigrams = [text[i : i + 2] for i in range(total_chars - 1)]
        distinct_2 = len(set(bigrams)) / len(bigrams)
    else:
        distinct_2 = 1.0

    # 3-grams
    if total_chars >= 3:
        trigrams = [text[i : i + 3] for i in range(total_chars - 2)]
        distinct_3 = len(set(trigrams)) / len(trigrams)
    else:
        distinct_3 = 1.0

    return {
        "ttr": round(ttr, 4),
        "distinct_1": round(ttr, 4),
        "distinct_2": round(distinct_2, 4),
        "distinct_3": round(distinct_3, 4),
    }


def find_repeated_phrases(
    text: str,
    ngram_size: int = 5,
    min_count: int = 3,
) -> list[tuple[str, int]]:
    """
    Find repeated phrase patterns of length `ngram_size` occurring >= `min_count` times.
    """
    clean = re.sub(r"[，。！？、“”《》\s]", "", text)
    if len(clean) < ngram_size:
        return []

    ngrams = [clean[i : i + ngram_size] for i in range(len(clean) - ngram_size + 1)]
    counts = Counter(ngrams)
    repeated = [(phrase, count) for phrase, count in counts.items() if count >= min_count]
    return sorted(repeated, key=lambda x: x[1], reverse=True)
