"""Tests for transcript indexing."""
from __future__ import annotations

from src.index.transcript_index import (
    TranscriptChunk,
    segment_transcript,
    search_transcript,
    get_chunks_in_range,
    save_transcript_index,
    load_transcript_index,
)


def _make_entries():
    """Create test subtitle entries as dicts."""
    return [
        {"start": 0.0, "end": 5.0, "text": "Welcome to the story."},
        {"start": 5.5, "end": 10.0, "text": "Something happens here."},
        {"start": 10.5, "end": 15.0, "text": "The plot thickens."},
        {"start": 60.0, "end": 65.0, "text": "A new chapter begins."},
        {"start": 65.5, "end": 70.0, "text": "Find the answer."},
    ]


def test_segment_transcript():
    """Test segmenting transcript into chunks."""
    entries = _make_entries()
    chunks = segment_transcript(entries, max_duration_seconds=20.0, max_words=100)
    assert len(chunks) >= 1
    assert all(isinstance(c, TranscriptChunk) for c in chunks)
    # All entries should be covered
    total_entries = sum(c.source_entries for c in chunks)
    assert total_entries == len(entries)


def test_search_transcript():
    """Test searching transcript chunks."""
    entries = _make_entries()
    chunks = segment_transcript(entries, max_duration_seconds=120.0, max_words=500)
    results = search_transcript(chunks, "find")
    assert len(results) >= 1
    assert any("Find" in c.text or "find" in c.text for c in results)


def test_search_no_results():
    """Test searching with no matches."""
    entries = _make_entries()
    chunks = segment_transcript(entries, max_duration_seconds=120.0, max_words=500)
    results = search_transcript(chunks, "zzzznonexistent")
    assert len(results) == 0


def test_get_chunks_in_range():
    """Test getting chunks overlapping with a time range."""
    entries = _make_entries()
    chunks = segment_transcript(entries, max_duration_seconds=20.0, max_words=100)
    results = get_chunks_in_range(chunks, 4.0, 12.0)
    assert len(results) >= 1


def test_save_load_roundtrip(tmp_path):
    """Test saving and loading the transcript index."""
    entries = _make_entries()
    chunks = segment_transcript(entries, max_duration_seconds=120.0, max_words=500)

    path = tmp_path / "transcript_index.json"
    save_transcript_index(chunks, path)

    loaded = load_transcript_index(path)
    assert len(loaded) == len(chunks)
    assert loaded[0].chunk_id == chunks[0].chunk_id
    assert loaded[0].text == chunks[0].text
