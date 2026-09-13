from __future__ import annotations
from pathlib import Path
import os


def validate_project_dir(path: Path) -> tuple[bool, list[str]]:
    """Check project directory structure."""
    errors = []
    if not path.exists():
        errors.append(f"Directory {path} does not exist.")
        return False, errors
    if not path.is_dir():
        errors.append(f"{path} is not a directory.")
        return False, errors
    return True, errors


def validate_media_file(path: Path) -> tuple[bool, list[str]]:
    """Check if file exists, is readable, has reasonable size."""
    errors = []
    if not path.exists():
        errors.append(f"File {path} does not exist.")
        return False, errors
    if not path.is_file():
        errors.append(f"{path} is not a file.")
        return False, errors
    if not os.access(path, os.R_OK):
        errors.append(f"File {path} is not readable.")
    if path.stat().st_size == 0:
        errors.append(f"File {path} is empty.")
    return len(errors) == 0, errors


def validate_json_schema(data: dict, schema: dict) -> tuple[bool, list[str]]:
    """Basic JSON schema validation."""
    # Simplified validation, ideally use jsonschema library
    errors = []
    if not isinstance(data, dict):
        return False, ["Data must be a dictionary."]
    return True, errors
