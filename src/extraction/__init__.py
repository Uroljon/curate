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
    build_structure_prompt,
    extract_structures_with_retry,
    extract_with_accumulation,
    prepare_llm_chunks,
    extract_action_fields_only,
    extract_projects_for_field,
    extract_project_details,
    merge_similar_action_fields,
)

__all__ = [
    # Prompts
    "STAGE1_SYSTEM_MESSAGE",
    "get_stage1_prompt",
    "get_stage2_prompt",
    "get_stage2_system_message",
    "get_stage3_prompt",
    "get_stage3_system_message",
    # Structure extractor
    "build_structure_prompt",
    "extract_structures_with_retry",
    "extract_with_accumulation",
    "prepare_llm_chunks",
    "extract_action_fields_only",
    "extract_projects_for_field",
    "extract_project_details",
    "merge_similar_action_fields",
]