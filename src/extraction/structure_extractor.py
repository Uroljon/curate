import json
from typing import Any

from src.core import (
    CHUNK_WARNING_THRESHOLD,
    EXTRACTION_MAX_RETRIES,
    MODEL_TEMPERATURE,
    ActionFieldList,
    ExtractionResult,
    ProjectDetails,
    ProjectList,
    query_ollama_structured,
)

from src.prompts import get_prompt

# prepare_llm_chunks is imported from semantic_llm_chunker


def extract_action_fields_only(
    chunks: list[str],
    log_file_path: str | None = None,
    log_context_prefix: str | None = None,
) -> list[str]:
    """
    Stage 1: Extract just action field names from all chunks.

    This is a lightweight extraction that only identifies the main categories
    (Handlungsfelder) without extracting projects or details.
    """
    all_action_fields = set()  # Use set to automatically deduplicate

    # Use enhanced prompt for mixed-topic robustness
    system_message = get_prompt("legacy.system_messages.stage1_action_fields")

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        print(
            f"🔍 Stage 1: Scanning chunk {i+1}/{len(chunks)} for action fields ({len(chunk)} chars)"
        )

        prompt = get_prompt("legacy.templates.stage1_chunk", chunk=chunk)

        result = query_ollama_structured(
            prompt=prompt,
            response_model=ActionFieldList,
            system_message=system_message,
            temperature=MODEL_TEMPERATURE,
            log_file_path=log_file_path,
            log_context=(
                f"{log_context_prefix} - Stage 1: Action Field Discovery, Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)"
                if log_context_prefix
                else f"Stage 1: Action Field Discovery, Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)"
            ),
        )

        if result and result.action_fields:
            found_fields = set(result.action_fields)
            print(
                f"   ✓ Found {len(found_fields)} action fields: {', '.join(sorted(found_fields))}"
            )
            all_action_fields.update(found_fields)
        else:
            print(f"   ✗ No action fields found in chunk {i+1}")

    # Convert to sorted list and merge similar fields
    merged_fields = merge_similar_action_fields(list(all_action_fields))

    print(f"\n📊 Stage 1 Complete: Found {len(merged_fields)} unique action fields")
    for field in merged_fields:
        print(f"   • {field}")

    return merged_fields


def merge_similar_action_fields(fields: list[str]) -> list[str]:
    """
    Merge similar action field names to avoid duplication.

    Examples:
    - "Klimaschutz" and "Klimaschutz und Energie" → "Klimaschutz und Energie"
    - "Mobilität" and "Mobilität und Verkehr" → "Mobilität und Verkehr"

    This function now uses the advanced EntityResolver for more sophisticated merging.
    """
    if not fields:
        return []

    # Convert fields to structure format for entity resolver
    temp_structures = [{"action_field": field, "projects": []} for field in fields]

    # Use entity resolver for sophisticated merging
    from src.core.config import ENTITY_RESOLUTION_ENABLED
    from src.processing.entity_resolver import resolve_extraction_entities

    if ENTITY_RESOLUTION_ENABLED:
        try:
            resolved_structures = resolve_extraction_entities(
                temp_structures, resolve_action_fields=True, resolve_projects=False
            )
            # Extract the resolved field names
            merged_fields = [struct["action_field"] for struct in resolved_structures]
            return sorted(merged_fields)  # Return alphabetically sorted
        except Exception as e:
            print(f"⚠️ Entity resolution failed, falling back to simple merging: {e}")

    # Fallback to original simple merging logic
    # Sort by length (longer names often contain more context)
    sorted_fields = sorted(fields, key=len, reverse=True)
    merged: list[str] = []

    for field in sorted_fields:
        field_lower = field.lower()
        is_subset = False

        # Check if this field is a subset of any already merged field
        for merged_field in merged:
            merged_lower = merged_field.lower()
            # Check if one contains the other
            if field_lower in merged_lower or merged_lower in field_lower:
                is_subset = True
                break

        if not is_subset:
            merged.append(field)

    return sorted(merged)  # Return alphabetically sorted


def extract_projects_for_field(chunks: list[str], action_field: str) -> list[str]:
    """
    Stage 2: Extract project names for a specific action field.

    Given an action field (e.g., "Klimaschutz"), find all projects
    that belong to this category across all chunks.
    """
    all_projects = set()

    # Use enhanced prompt for mixed-topic robustness
    system_message = get_prompt("legacy.system_messages.stage2_projects", action_field=action_field)

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        # Remove quick check - with mixed topics, action field might not be explicitly mentioned

        print(
            f"🔎 Stage 2: Searching chunk {i+1}/{len(chunks)} for {action_field} projects"
        )

        prompt = get_prompt("legacy.templates.stage2_chunk", chunk=chunk, action_field=action_field)

        result = query_ollama_structured(
            prompt=prompt,
            response_model=ProjectList,
            system_message=system_message,
            temperature=MODEL_TEMPERATURE,
        )

        if result and result.projects:
            found_projects = set(result.projects)
            print(f"   ✓ Found {len(found_projects)} projects")
            all_projects.update(found_projects)

    # Remove duplicates and sort
    unique_projects = sorted(all_projects)

    print(f"   📋 Total {len(unique_projects)} projects for {action_field}")

    return unique_projects


def extract_project_details(
    chunks: list[str], action_field: str, project_title: str
) -> ProjectDetails:
    """
    Stage 3: Extract measures and indicators for a specific project.

    This is the most focused extraction, looking for specific details
    about a single project within an action field.
    """
    all_measures = set()
    all_indicators = set()

    # Use enhanced prompt for mixed-topic robustness
    system_message = get_prompt("legacy.system_messages.stage3_measures_indicators", 
                                action_field=action_field, project_title=project_title)

    # Process ALL chunks - indicators might be separated from project mentions
    # in mixed-topic chunks
    print(f"🔬 Stage 3: Analyzing ALL {len(chunks)} chunks for {project_title} details")

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        prompt = get_prompt("legacy.templates.stage3_chunk", chunk=chunk, project_title=project_title)

        result = query_ollama_structured(
            prompt=prompt,
            response_model=ProjectDetails,
            system_message=system_message,
            temperature=MODEL_TEMPERATURE,
        )

        if result:
            if result.measures:
                all_measures.update(result.measures)
                print(f"   ✓ Chunk {i+1}: Found {len(result.measures)} measures")
            if result.indicators:
                all_indicators.update(result.indicators)
                print(f"   ✓ Chunk {i+1}: Found {len(result.indicators)} indicators")

    # Create final result
    details = ProjectDetails(
        measures=sorted(all_measures), indicators=sorted(all_indicators)
    )

    print(
        f"   📊 Total: {len(details.measures)} measures, {len(details.indicators)} indicators"
    )

    return details


def extract_structures_with_retry(
    chunk_text: str,
    max_retries: int = EXTRACTION_MAX_RETRIES,
    log_file_path: str | None = None,
    log_context: str | None = None,
) -> list[dict[str, Any]]:
    """
    Extract structures from text using Ollama structured output.
    """
    system_message = get_prompt("extraction.system_messages.retry_extraction")

    prompt = get_prompt("extraction.templates.retry_chunk", chunk_text=chunk_text.strip())

    # Validate chunk size
    if len(chunk_text) > CHUNK_WARNING_THRESHOLD:
        print(
            f"⚠️ WARNING: Chunk size ({len(chunk_text)} chars) exceeds recommended limit "
            f"of {CHUNK_WARNING_THRESHOLD} chars!"
        )
        print("   This may cause JSON parsing issues or incomplete responses.")

    for attempt in range(max_retries):
        print(
            f"📝 Extraction attempt {attempt + 1}/{max_retries} for chunk ({len(chunk_text)} chars)"
        )

        # Use structured output with Pydantic model
        result = query_ollama_structured(
            prompt=prompt,
            response_model=ExtractionResult,
            system_message=system_message,
            temperature=MODEL_TEMPERATURE,  # Zero temperature for deterministic extraction
            log_file_path=log_file_path,
            log_context=log_context,
        )

        if result is not None:
            # Convert Pydantic model to dict format expected by rest of pipeline
            extracted_data = []
            for af in result.action_fields:
                action_field_dict: dict[str, Any] = {
                    "action_field": af.action_field,
                    "projects": [],
                }
                for project in af.projects:
                    project_dict: dict[str, Any] = {"title": project.title}
                    if project.measures:
                        project_dict["measures"] = project.measures
                    if project.indicators:
                        project_dict["indicators"] = project.indicators
                    action_field_dict["projects"].append(project_dict)
                extracted_data.append(action_field_dict)

            print(
                f"✅ Successfully extracted {len(extracted_data)} action fields on attempt {attempt + 1}"
            )

            return extracted_data
        else:
            print(f"❌ Attempt {attempt + 1} failed - structured output returned None")
            if attempt < max_retries - 1:
                print("🔄 Retrying...")

    print(f"⚠️ All {max_retries} attempts failed for chunk")
    return []


def extract_with_accumulation(
    accumulated_data: dict[str, Any],
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
) -> dict[str, Any]:
    """
    Extract structures from text while enhancing previously accumulated results.

    Args:
        accumulated_data: The current accumulated extraction results
        chunk_text: New text chunk to process
        chunk_index: Current chunk number (0-based)
        total_chunks: Total number of chunks

    Returns:
        Enhanced complete structure with new and updated data
    """
    # Special handling for first chunk
    if chunk_index == 0:
        # First chunk uses regular extraction
        print(
            f"📝 Initial extraction for chunk 1/{total_chunks} ({len(chunk_text)} chars)"
        )
        result = extract_structures_with_retry(chunk_text)
        return {"action_fields": result}

    print(
        f"🔄 Progressive extraction for chunk {chunk_index + 1}/{total_chunks} ({len(chunk_text)} chars)"
    )

    system_message = get_prompt("extraction.system_messages.accumulated_enhancement")

    prompt = get_prompt("extraction.templates.accumulated_enhance",
                        count=len(accumulated_data.get('action_fields', [])),
                        accumulated_json=json.dumps(accumulated_data, indent=2, ensure_ascii=False),
                        chunk_text=chunk_text.strip())

    # Use structured output for consistency
    enhanced_result = query_ollama_structured(
        prompt=prompt,
        response_model=ExtractionResult,
        system_message=system_message,
        temperature=MODEL_TEMPERATURE,
    )

    if enhanced_result:
        # Count what was added/enhanced
        old_count = len(accumulated_data.get("action_fields", []))
        new_count = len(enhanced_result.action_fields)

        old_projects = sum(
            len(af.get("projects", []))
            for af in accumulated_data.get("action_fields", [])
        )
        new_projects = sum(len(af.projects) for af in enhanced_result.action_fields)

        print(
            f"✅ Enhanced: {old_count}→{new_count} action fields, {old_projects}→{new_projects} projects"
        )

        # Convert to dict format
        result_dict: dict[str, Any] = {"action_fields": []}
        for af in enhanced_result.action_fields:
            af_dict: dict[str, Any] = {"action_field": af.action_field, "projects": []}
            for project in af.projects:
                proj_dict: dict[str, Any] = {"title": project.title}
                if project.measures:
                    proj_dict["measures"] = project.measures
                if project.indicators:
                    proj_dict["indicators"] = project.indicators
                af_dict["projects"].append(proj_dict)
            result_dict["action_fields"].append(af_dict)

        return result_dict
    else:
        print("⚠️ Enhancement failed, keeping previous state")
        return accumulated_data
