"""Subtitle parsing and ASR transcription utilities (FR-2 / FR-3)."""
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

@dataclass
class SubtitleEntry:
    """A single subtitle entry."""
    index: int
    start_seconds: float
    end_seconds: float
    text: str

def subtitle_time_to_seconds(time_str: str) -> float:
    """Parse SRT/VTT timestamp to seconds."""
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(time_str)

def parse_srt(path: Path) -> List[SubtitleEntry]:
    """Parse SRT file into list of SubtitleEntry."""
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    blocks = content.split('\n\n')
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            try:
                index = int(lines[0].strip())
                time_line = lines[1]
                text = '\n'.join(lines[2:]).strip()
                
                if ' -->' in time_line:
                    start_str, end_str = time_line.split(' --> ')
                    start = subtitle_time_to_seconds(start_str.strip())
                    end = subtitle_time_to_seconds(end_str.strip())
                    entries.append(SubtitleEntry(index, start, end, text))
            except ValueError:
                continue
    return entries

def parse_vtt(path: Path) -> List[SubtitleEntry]:
    """Parse VTT file into list of SubtitleEntry."""
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    index = 1
    current_start = None
    current_end = None
    current_text = []
    
    for line in lines:
        line = line.strip()
        if line == 'WEBVTT' or line.startswith('NOTE') or not line:
            if current_start is not None:
                entries.append(SubtitleEntry(index, current_start, current_end, '\n'.join(current_text)))
                index += 1
                current_start = None
                current_end = None
                current_text = []
            continue
            
        if '-->' in line:
            parts = line.split('-->')
            current_start = subtitle_time_to_seconds(parts[0].strip())
            current_end = subtitle_time_to_seconds(parts[1].strip().split(' ')[0])
        elif current_start is not None:
            current_text.append(line)
            
    if current_start is not None:
        entries.append(SubtitleEntry(index, current_start, current_end, '\n'.join(current_text)))
        
    return entries

def search_subtitles(entries: List[SubtitleEntry], query: str) -> List[SubtitleEntry]:
    """Search for query in subtitle entries."""
    query = query.lower()
    return [entry for entry in entries if query in entry.text.lower()]

def get_subtitles_in_range(entries: List[SubtitleEntry], start: float, end: float) -> List[SubtitleEntry]:
    """Get subtitles within a time range."""
    return [entry for entry in entries if entry.start_seconds >= start and entry.start_seconds <= end]


def transcribe_audio_to_srt(
    audio_path: Path,
    output_srt: Path,
    model_size: str = "base",
    language: Optional[str] = None,
) -> List[SubtitleEntry]:
    """
    Transcribe audio track using faster-whisper to generate timestamped subtitles.

    Args:
        audio_path: Path to source audio file (WAV or MP3).
        output_srt: Output path for generated .srt file.
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium').
        language: Optional language code (e.g. 'zh', 'ja', 'en').

    Returns:
        List of SubtitleEntry objects.
    """
    from src.utils.timing import seconds_to_srt_time

    output_srt.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Transcribing audio %s with faster-whisper (model=%s)...", audio_path, model_size)

    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments_gen, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        entries: list[SubtitleEntry] = []
        srt_lines: list[str] = []

        for idx, seg in enumerate(segments_gen, start=1):
            text = seg.text.strip()
            if not text:
                continue
            entry = SubtitleEntry(
                index=idx,
                start_seconds=float(seg.start),
                end_seconds=float(seg.end),
                text=text,
            )
            entries.append(entry)

            start_str = seconds_to_srt_time(entry.start_seconds)
            end_str = seconds_to_srt_time(entry.end_seconds)
            srt_lines.append(str(idx))
            srt_lines.append(f"{start_str} --> {end_str}")
            srt_lines.append(text)
            srt_lines.append("")

        output_srt.write_text("\n".join(srt_lines), encoding="utf-8")
        logger.info(
            "Transcription complete: %d subtitle entries extracted across %s (detected lang=%s, prob=%.2f)",
            len(entries),
            audio_path,
            info.language,
            info.language_probability,
        )
        return entries

    except Exception as e:
        logger.warning("faster-whisper transcription failed (%s). Generating fallback empty subtitles.", e)
        output_srt.write_text("", encoding="utf-8")
        return []

