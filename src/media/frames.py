"""Keyframe extraction utilities."""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict

def extract_keyframes(video_path: Path, timestamps: List[float], output_dir: Path) -> List[Path]:
    """Extract frames at specific timestamps using ffmpeg."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_paths = []
    
    for i, ts in enumerate(timestamps):
        output_file = output_dir / f"frame_{i:04d}_{ts:.2f}.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(ts),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(output_file)
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            if output_file.exists():
                extracted_paths.append(output_file)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract frame at {ts}: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")
            
    return extracted_paths

def extract_scene_keyframes(video_path: Path, scenes: List[Tuple[float, float]], output_dir: Path, frames_per_scene: int = 3) -> Dict[int, List[Path]]:
    """Extract keyframes for each scene."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    
    for i, (start, end) in enumerate(scenes):
        duration = end - start
        if frames_per_scene == 1:
            timestamps = [start + duration / 2]
        else:
            step = duration / (frames_per_scene - 1) if frames_per_scene > 1 else 0
            timestamps = [start + j * step for j in range(frames_per_scene)]
            
        extracted = []
        for j, ts in enumerate(timestamps):
            output_file = output_dir / f"scene_{i:04d}_frame_{j:02d}_{ts:.2f}.jpg"
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(ts),
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                str(output_file)
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                if output_file.exists():
                    extracted.append(output_file)
            except Exception:
                continue
        result[i] = extracted
        
    return result
