"""
Utility functions for JSON quality analysis.
"""

import json
import re
from pathlib import Path
from typing import Any, Optional


def find_json_files(directory: str | Path, pattern: str = "*.json") -> list[Path]:
    """
    Find JSON files in a directory.

    Args:
        directory: Directory to search
        pattern: File pattern (default: *.json)

    Returns:
        List of matching file paths
    """
    directory = Path(directory)

    if not directory.exists():
        return []

    if directory.is_file():
        return [directory] if directory.suffix.lower() == ".json" else []

    return list(directory.glob(pattern))


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def format_duration(milliseconds: float) -> str:
    """
    Format duration in human-readable format.

    Args:
        milliseconds: Duration in milliseconds

    Returns:
        Formatted duration string
    """
    if milliseconds < 1000:
        return f"{milliseconds:.0f} ms"
    elif milliseconds < 60 * 1000:
        return f"{milliseconds / 1000:.1f} s"
    else:
        minutes = int(milliseconds / (60 * 1000))
        seconds = (milliseconds % (60 * 1000)) / 1000
        return f"{minutes}m {seconds:.1f}s"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys

    Returns:
        Flattened dictionary
    """
    items = []

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)
