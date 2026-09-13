"""Source ingestion modules."""
from __future__ import annotations
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

from src.models.project import Project, MediaInfo
from src.media.probe import extract_media_info

SUPPORTED_MEDIA_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".ts", ".m4v"}

def compute_file_hash(path: Path, algorithm: str = 'sha256') -> str:
    """Compute hash of a file in chunks."""
    hash_func = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def validate_incoming(incoming_dir: Path) -> Tuple[bool, List[str]]:
    """Validate incoming folder has source media."""
    errors = []
    if not incoming_dir.exists():
        errors.append(f"Directory {incoming_dir} does not exist.")
        return False, errors
        
    found_media = any(f.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS for f in incoming_dir.iterdir() if f.is_file())
    
    if not found_media:
        errors.append(f"No valid media file found in {incoming_dir}")
        
    return len(errors) == 0, errors

def ingest_movie(incoming_dir: Path, projects_dir: Path) -> Dict[str, Any]:
    """Ingest a movie into a new project."""
    valid, errors = validate_incoming(incoming_dir)
    if not valid:
        raise ValueError(f"Invalid incoming directory: {', '.join(errors)}")
        
    project_id = str(uuid.uuid4())
    project_dir = projects_dir / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (project_dir / "media").mkdir(exist_ok=True)
    (project_dir / "audio").mkdir(exist_ok=True)
    (project_dir / "keyframes").mkdir(exist_ok=True)
    (project_dir / "assets").mkdir(exist_ok=True)
    (project_dir / "renders").mkdir(exist_ok=True)
    
    # Find source video
    source_file = next(f for f in incoming_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS)
    
    dest_file = project_dir / "media" / source_file.name
    shutil.copy2(source_file, dest_file)
    
    file_hash = compute_file_hash(dest_file)
    media_info_model = extract_media_info(dest_file)
    media_info_dict = media_info_model.model_dump()
    
    # Check for metadata and subtitles
    title = source_file.stem.replace("_", " ")
    metadata = {}
    
    metadata_file = incoming_dir / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                title = metadata.get("title", title)
        except Exception:
            pass
            
    srt_file = incoming_dir / "subtitles.srt"
    if srt_file.exists():
        shutil.copy2(srt_file, project_dir / "media" / "subtitles.srt")
    
    from src.orchestrator.project import ProjectManager
    pm = ProjectManager(projects_dir)
    project = pm.create_project(
        source_path=dest_file,
        source_hash=file_hash,
        title=title,
        media_info=media_info_dict,
        metadata=metadata
    )
    return project
