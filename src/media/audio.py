"""Audio extraction utilities."""
from __future__ import annotations
import subprocess
from pathlib import Path

def extract_audio(source_path: Path, output_path: Path, format: str = 'wav', sample_rate: int = 16000) -> Path:
    """Extract audio track using ffmpeg."""
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
        
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(source_path),
        "-vn",
        "-acodec", "pcm_s16le" if format == 'wav' else format,
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio extraction failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

def extract_audio_segment(source_path: Path, output_path: Path, start_seconds: float, end_seconds: float) -> Path:
    """Extract a specific segment of audio."""
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
        
    duration = end_seconds - start_seconds
    if duration <= 0:
        raise ValueError("End time must be greater than start time.")
        
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_seconds),
        "-i", str(source_path),
        "-t", str(duration),
        "-vn",
        "-acodec", "copy",
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio segment extraction failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found.")
