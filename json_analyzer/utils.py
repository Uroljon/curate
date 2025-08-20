"""
Utility functions for JSON quality analysis.
"""

import json
import re
from pathlib import Path
from typing import Any, Optional


def load_json_file(file_path: str | Path) -> dict[str, Any]:
    """
    Load and parse a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data

    Raises:
        ValueError: If file cannot be loaded or parsed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        msg = f"File not found: {file_path}"
        raise ValueError(msg)

    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in {file_path}: {e}"
        raise ValueError(msg) from e
    except Exception as e:
        msg = f"Failed to load {file_path}: {e}"
        raise ValueError(msg) from e


def save_json_file(
    data: dict[str, Any], file_path: str | Path, indent: int = 2
) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation level
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, separators=(",", ": "))


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


def validate_json_structure(
    data: dict[str, Any], required_fields: list[str]
) -> list[str]:
    """
    Validate that JSON data has required fields.

    Args:
        data: JSON data to validate
        required_fields: List of required field names

    Returns:
        List of missing fields
    """
    missing_fields = []

    for field in required_fields:
        if field not in data:
            missing_fields.append(field)
        elif not data[field]:  # Empty list, None, etc.
            missing_fields.append(f"{field} (empty)")

    return missing_fields


def clean_filename(filename: str) -> str:
    """
    Clean filename for safe filesystem usage.

    Args:
        filename: Original filename

    Returns:
        Cleaned filename
    """
    # Remove or replace problematic characters
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", filename)

    # Remove extra underscores and spaces
    cleaned = re.sub(r"[_\s]+", "_", cleaned)

    # Trim and ensure reasonable length
    cleaned = cleaned.strip("_")[:200]

    return cleaned


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


def extract_entity_name(entity: dict[str, Any]) -> str:
    """
    Extract the name/title from an entity.

    Args:
        entity: Entity dictionary

    Returns:
        Entity name or empty string
    """
    content = entity.get("content", {})

    # Try common name fields
    for field in ["title", "name", "action_field"]:
        name = content.get(field, "")
        if isinstance(name, str) and name.strip():
            return name.strip()

    # Fallback to entity ID
    entity_id = entity.get("id", "")
    if entity_id:
        return entity_id

    return ""


def group_by_type(
    items: list[dict[str, Any]], type_key: str = "type"
) -> dict[str, list[dict[str, Any]]]:
    """
    Group items by a type field.

    Args:
        items: List of items to group
        type_key: Field name containing the type

    Returns:
        Dictionary mapping types to item lists
    """
    groups = {}

    for item in items:
        item_type = item.get(type_key, "unknown")
        if item_type not in groups:
            groups[item_type] = []
        groups[item_type].append(item)

    return groups


def calculate_percentage(part: float, total: float) -> float:
    """
    Calculate percentage with safe division.

    Args:
        part: Part value
        total: Total value

    Returns:
        Percentage (0-100)
    """
    if total == 0:
        return 0.0
    return (part / total) * 100


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Perform safe division with default for zero denominator.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero

    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


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


def create_summary_table(data: dict[str, Any], title: str = "Summary") -> str:
    """
    Create a simple text table for summary data.

    Args:
        data: Data to display
        title: Table title

    Returns:
        Formatted table string
    """
    if not data:
        return f"{title}: No data"

    lines = [title, "=" * len(title)]

    max_key_length = max(len(str(k)) for k in data.keys())

    for key, value in data.items():
        if isinstance(value, float):
            value_str = f"{value:.2f}"
        elif isinstance(value, dict):
            value_str = f"{len(value)} items"
        elif isinstance(value, list):
            value_str = f"{len(value)} items"
        else:
            value_str = str(value)

        lines.append(f"{str(key).ljust(max_key_length)}: {value_str}")

    return "\n".join(lines)
