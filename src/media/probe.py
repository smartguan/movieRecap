"""Media probe utilities using ffprobe."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

from src.models.project import MediaInfo

def probe_media(path: Path) -> Dict[str, Any]:
    """Run ffprobe on a media file and return the parsed JSON output."""
    if not path.exists():
        raise FileNotFoundError(f"Media file not found: {path}")
    
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except FileNotFoundError:
        raise RuntimeError("ffprobe executable not found. Please install ffmpeg.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed to process {path}. Error: {e.stderr}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse ffprobe output for {path}: {e}")

def extract_media_info(path: Path) -> MediaInfo:
    """Extract structured MediaInfo from a media file."""
    probe_data = probe_media(path)
    
    format_info = probe_data.get("format", {})
    streams = probe_data.get("streams", [])
    
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    subtitle_streams = [s for s in streams if s.get("codec_type") == "subtitle"]
    
    duration = float(format_info.get("duration", 0.0))
    
    if video_stream:
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        resolution = (width, height)
        frame_rate_str = video_stream.get("r_frame_rate", "0/1")
        try:
            num, den = map(int, frame_rate_str.split('/'))
            frame_rate = num / den if den != 0 else 0.0
        except ValueError:
            frame_rate = 0.0
        video_codec = video_stream.get("codec_name", "unknown")
    else:
        resolution = (0, 0)
        frame_rate = 0.0
        video_codec = "unknown"
        
    audio_codec = audio_streams[0].get("codec_name", "unknown") if audio_streams else "none"
    audio_tracks = len(audio_streams)
    has_subtitles = len(subtitle_streams) > 0
    
    return MediaInfo(
        duration_seconds=duration,
        resolution=resolution,
        frame_rate=frame_rate,
        video_codec=video_codec,
        audio_codec=audio_codec,
        audio_tracks=audio_tracks,
        has_subtitles=has_subtitles
    )
