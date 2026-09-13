"""Scene detection using PySceneDetect."""
from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from scenedetect import detect, ContentDetector, AdaptiveDetector
except ImportError:
    logger.warning("scenedetect is not installed. Scene detection features will be limited.")
    detect = None
    ContentDetector = None
    AdaptiveDetector = None

def detect_scenes(video_path: Path, threshold: float = 27.0) -> List[Tuple[float, float]]:
    """Detect scenes using ContentDetector."""
    if detect is None or ContentDetector is None:
        raise RuntimeError("scenedetect is not installed.")
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    scene_list = detect(str(video_path), ContentDetector(threshold=threshold))
    
    result = []
    for scene in scene_list:
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        result.append((start_time, end_time))
        
    return result

def detect_scenes_adaptive(video_path: Path) -> List[Tuple[float, float]]:
    """Detect scenes using AdaptiveDetector."""
    if detect is None or AdaptiveDetector is None:
        raise RuntimeError("scenedetect is not installed.")
        
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    scene_list = detect(str(video_path), AdaptiveDetector())
    
    result = []
    for scene in scene_list:
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        result.append((start_time, end_time))
        
    return result
