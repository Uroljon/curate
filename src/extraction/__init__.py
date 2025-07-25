"""Extraction logic for CURATE."""

from .prompts import (
    STAGE1_SYSTEM_MESSAGE,
    get_stage1_prompt,
    get_stage2_prompt,
    get_stage2_system_message,
    get_stage3_prompt,
    get_stage3_system_message,
)
from .structure_extractor import (
    extract_action_fields_only,
    extract_project_details,
    extract_projects_for_field,
    extract_structures_with_retry,
    extract_with_accumulation,
    merge_similar_action_fields,
)

__all__ = [
    # Prompts
    "STAGE1_SYSTEM_MESSAGE",
    # Structure extractor
    "extract_action_fields_only",
    "extract_project_details",
    "extract_projects_for_field",
    "extract_structures_with_retry",
    "extract_with_accumulation",
    "get_stage1_prompt",
    "get_stage2_prompt",
    "get_stage2_system_message",
    "get_stage3_prompt",
    "get_stage3_system_message",
    "merge_similar_action_fields",
]
