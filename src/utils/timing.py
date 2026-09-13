from __future__ import annotations
def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def seconds_to_vtt_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def format_duration(seconds: float) -> str:
    """Human readable duration like '1h 23m 45s'"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def time_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Compute overlap in seconds between two time ranges"""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))
