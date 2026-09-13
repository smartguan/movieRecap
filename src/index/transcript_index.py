"""
Transcript indexing and segmentation.

DETERMINISTIC: All operations are text processing — no LLM calls.
Segments transcript data into searchable chunks with timestamps.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TranscriptChunk:
    """A chunk of transcript with timestamp range."""

    chunk_id: str
    start_seconds: float
    end_seconds: float
    text: str
    word_count: int
    source_entries: int  # Number of original subtitle entries merged


def segment_transcript(
    subtitle_entries: list[dict[str, Any]],
    max_duration_seconds: float = 60.0,
    max_words: int = 200,
) -> list[TranscriptChunk]:
    """
    Segment subtitle entries into coherent transcript chunks.

    Groups consecutive entries into chunks based on time gaps and word count limits.
    This is purely deterministic text processing.

    Args:
        subtitle_entries: List of dicts with 'start', 'end', 'text' fields.
        max_duration_seconds: Maximum duration of each chunk.
        max_words: Maximum word count per chunk.

    Returns:
        List of TranscriptChunk objects.
    """
    if not subtitle_entries:
        return []

    chunks: list[TranscriptChunk] = []
    current_texts: list[str] = []
    current_start = subtitle_entries[0].get("start", 0.0)
    current_end = subtitle_entries[0].get("end", 0.0)
    current_word_count = 0
    entry_count = 0
    chunk_counter = 0

    for entry in subtitle_entries:
        text = entry.get("text", "").strip()
        if not text:
            continue

        start = entry.get("start", 0.0)
        end = entry.get("end", 0.0)

        # Calculate words (for CJK, count characters as "words")
        words = len(text)

        # Check if we should start a new chunk
        duration = end - current_start
        should_split = (
            duration > max_duration_seconds
            or current_word_count + words > max_words
        )

        if should_split and current_texts:
            # Save current chunk
            chunks.append(
                TranscriptChunk(
                    chunk_id=f"chunk-{chunk_counter:04d}",
                    start_seconds=current_start,
                    end_seconds=current_end,
                    text=" ".join(current_texts),
                    word_count=current_word_count,
                    source_entries=entry_count,
                )
            )
            chunk_counter += 1
            current_texts = []
            current_start = start
            current_word_count = 0
            entry_count = 0

        current_texts.append(text)
        current_end = end
        current_word_count += words
        entry_count += 1

    # Save final chunk
    if current_texts:
        chunks.append(
            TranscriptChunk(
                chunk_id=f"chunk-{chunk_counter:04d}",
                start_seconds=current_start,
                end_seconds=current_end,
                text=" ".join(current_texts),
                word_count=current_word_count,
                source_entries=entry_count,
            )
        )

    return chunks


def search_transcript(
    chunks: list[TranscriptChunk],
    query: str,
) -> list[TranscriptChunk]:
    """
    Search transcript chunks for text matching a query.

    DETERMINISTIC: Simple substring search.

    Args:
        chunks: List of TranscriptChunk objects.
        query: Search query string.

    Returns:
        List of matching chunks.
    """
    query_lower = query.lower()
    return [c for c in chunks if query_lower in c.text.lower()]


def get_chunks_in_range(
    chunks: list[TranscriptChunk],
    start_seconds: float,
    end_seconds: float,
) -> list[TranscriptChunk]:
    """
    Get transcript chunks that overlap with a time range.

    DETERMINISTIC: Simple range comparison.
    """
    return [
        c
        for c in chunks
        if c.start_seconds < end_seconds and c.end_seconds > start_seconds
    ]


def save_transcript_index(chunks: list[TranscriptChunk], path: Path) -> None:
    """Save transcript index to JSON."""
    data = [
        {
            "chunk_id": c.chunk_id,
            "start_seconds": c.start_seconds,
            "end_seconds": c.end_seconds,
            "text": c.text,
            "word_count": c.word_count,
            "source_entries": c.source_entries,
        }
        for c in chunks
    ]
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_transcript_index(path: Path) -> list[TranscriptChunk]:
    """Load transcript index from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        TranscriptChunk(
            chunk_id=d["chunk_id"],
            start_seconds=d["start_seconds"],
            end_seconds=d["end_seconds"],
            text=d["text"],
            word_count=d["word_count"],
            source_entries=d["source_entries"],
        )
        for d in data
    ]
