import hashlib
import json
from pathlib import Path


def hash_file(path: Path, algorithm: str = 'sha256') -> str:
    """Hash file in 64KB chunks."""
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def hash_string(s: str, algorithm: str = 'sha256') -> str:
    """Hash string."""
    h = hashlib.new(algorithm)
    h.update(s.encode('utf-8'))
    return h.hexdigest()


def hash_dict(d: dict) -> str:
    """Deterministic JSON serialization then hash."""
    serialized = json.dumps(d, sort_keys=True, separators=(',', ':'))
    return hash_string(serialized)
