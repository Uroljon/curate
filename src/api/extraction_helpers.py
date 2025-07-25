"""
Helper functions for extraction endpoints.

This module contains refactored helper functions to break down
the large extraction functions in routes.py.
"""

from typing import Any

from src.extraction.structure_extractor import (
    extract_action_fields_only,
    extract_project_details,
    extract_projects_for_field,
    extract_structures_with_retry,
)
from src.processing.chunker import chunk_for_llm
from src.processing.embedder import get_all_chunks_for_document


def prepare_chunks_for_extraction(
    source_id: str, max_chars: int, min_chars: int
) -> list[str]:
    """
    Prepare optimized chunks for LLM extraction.

    Args:
        source_id: Document identifier
        max_chars: Maximum characters per chunk
        min_chars: Minimum characters per chunk

    Returns:
        List of optimized text chunks
    """
    chunks = get_all_chunks_for_document(source_id)
    raw_texts = [c["text"] for c in chunks]
    return chunk_for_llm(raw_texts, max_chars=max_chars, min_chars=min_chars)


def extract_all_action_fields(chunks: list[str]) -> list[str]:
    """
    Stage 1: Extract action fields from chunks.

    Args:
        chunks: List of text chunks

    Returns:
        List of action field names
    """
    print("=" * 60)
    print("STAGE 1: DISCOVERING ACTION FIELDS")
    print("=" * 60)

    action_fields = extract_action_fields_only(chunks)

    if not action_fields:
        print("⚠️ No action fields found in Stage 1")

    return action_fields


def extract_projects_and_details(
    chunks: list[str], action_fields: list[str]
) -> list[dict[str, Any]]:
    """
    Stage 2 & 3: Extract projects and their details for each action field.

    Args:
        chunks: List of text chunks
        action_fields: List of action field names

    Returns:
        List of action field data with projects and details
    """
    all_extracted_data = []

    for field_idx, action_field in enumerate(action_fields):
        print(f"\n{'=' * 60}")
        print(
            f"STAGE 2: EXTRACTING PROJECTS FOR '{action_field}' ({field_idx + 1}/{len(action_fields)})"
        )
        print("=" * 60)

        projects = extract_projects_for_field(chunks, action_field)

        if not projects:
            print(f"   ⚠️ No projects found for {action_field}")
            continue

        # Stage 3: Extract details for each project
        action_field_data = extract_project_details_for_field(
            chunks, action_field, projects
        )
        all_extracted_data.append(action_field_data)

    return all_extracted_data


def extract_project_details_for_field(
    chunks: list[str], action_field: str, projects: list[str]
) -> dict[str, Any]:
    """
    Extract details for all projects in an action field.

    Args:
        chunks: List of text chunks
        action_field: Action field name
        projects: List of project titles

    Returns:
        Action field data with projects and details
    """
    action_field_data: dict[str, Any] = {
        "action_field": action_field,
        "projects": [],
    }

    print(f"\n{'=' * 60}")
    print(f"STAGE 3: EXTRACTING DETAILS FOR {len(projects)} PROJECTS")
    print("=" * 60)

    for proj_idx, project_title in enumerate(projects):
        print(f"\n📁 Project {proj_idx + 1}/{len(projects)}: {project_title}")

        details = extract_project_details(chunks, action_field, project_title)

        project_data: dict[str, Any] = {"title": project_title}
        if details.measures:
            project_data["measures"] = details.measures
        if details.indicators:
            project_data["indicators"] = details.indicators

        action_field_data["projects"].append(project_data)

    return action_field_data


def deduplicate_extraction_results(
    extracted_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Deduplicate action fields and merge their projects.

    Args:
        extracted_data: List of extracted action field data

    Returns:
        Deduplicated list of action field data
    """
    deduplicated_data: dict[str, Any] = {}

    for item in extracted_data:
        field_name: str = str(item["action_field"])

        if field_name in deduplicated_data:
            # Merge projects from duplicate action fields
            existing_projects = deduplicated_data[field_name]["projects"]
            new_projects = item["projects"]

            # Deduplicate projects by title
            existing_titles = {p["title"] for p in existing_projects}
            for project in new_projects:
                if project["title"] not in existing_titles:
                    existing_projects.append(project)
                else:
                    # Merge measures and indicators for duplicate projects
                    merge_project_details(existing_projects, project)
        else:
            deduplicated_data[field_name] = item

    return list(deduplicated_data.values())


def merge_project_details(existing_projects: list[dict], new_project: dict) -> None:
    """
    Merge measures and indicators from a duplicate project.

    Args:
        existing_projects: List of existing projects
        new_project: New project data to merge
    """
    for existing in existing_projects:
        if existing["title"] == new_project["title"]:
            # Merge measures
            if "measures" in new_project:
                if "measures" not in existing:
                    existing["measures"] = []
                for measure in new_project["measures"]:
                    if measure not in existing["measures"]:
                        existing["measures"].append(measure)

            # Merge indicators
            if "indicators" in new_project:
                if "indicators" not in existing:
                    existing["indicators"] = []
                for indicator in new_project["indicators"]:
                    if indicator not in existing["indicators"]:
                        existing["indicators"].append(indicator)
            break


def print_extraction_summary(
    all_extracted_data: list[dict[str, Any]], chunk_count: int
) -> None:
    """
    Print summary of extraction results.

    Args:
        all_extracted_data: List of extracted action field data
        chunk_count: Number of chunks processed
    """
    total_projects = sum(len(af["projects"]) for af in all_extracted_data)
    projects_with_indicators = sum(
        1 for af in all_extracted_data for p in af["projects"] if p.get("indicators")
    )

    print(f"\n{'=' * 60}")
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"📊 Action fields: {len(all_extracted_data)}")
    print(f"📁 Total projects: {total_projects}")
    print(f"📍 Projects with indicators: {projects_with_indicators}")
    print(f"📄 Chunks processed: {chunk_count}")
    print("=" * 60)


def process_chunks_for_fast_extraction(
    chunks: list[str], max_chunks: int | None = None
) -> tuple[list[dict[str, Any]], int]:
    """
    Process chunks for fast extraction with optional limiting.

    Args:
        chunks: List of text chunks
        max_chunks: Optional limit on number of chunks to process

    Returns:
        Tuple of (all_extracted_data, actual_chunks_processed)
    """
    all_extracted_data = []

    # Apply chunk limit if configured
    chunks_to_process = chunks
    if max_chunks and max_chunks > 0:
        chunks_to_process = chunks[:max_chunks]
        print(
            f"⚡ Fast extraction: Processing first {len(chunks_to_process)} of {len(chunks)} chunks"
        )

    # Process chunks
    for i, chunk in enumerate(chunks_to_process):
        print(f"\n📄 Processing chunk {i + 1}/{len(chunks_to_process)}...")
        chunk_data = extract_structures_with_retry(chunk)

        if chunk_data:
            all_extracted_data.extend(chunk_data)
            print(f"   ✓ Extracted {len(chunk_data)} action fields from chunk {i + 1}")
        else:
            print(f"   ✗ No structures extracted from chunk {i + 1}")

    return all_extracted_data, len(chunks_to_process)


def merge_extraction_results(
    all_extracted_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Merge extraction results by action field name.

    Args:
        all_extracted_data: List of extraction results from all chunks

    Returns:
        Dictionary mapping action field names to merged data
    """
    merged_structures: dict[str, Any] = {}

    for item in all_extracted_data:
        if not isinstance(item, dict) or "action_field" not in item:
            continue

        field_name = item["action_field"]

        if field_name in merged_structures:
            # Merge projects
            existing_projects = merged_structures[field_name].get("projects", [])
            new_projects = item.get("projects", [])

            # Deduplicate by title
            existing_titles = {
                p.get("title")
                for p in existing_projects
                if isinstance(p, dict) and p.get("title")
            }

            for project in new_projects:
                if (
                    isinstance(project, dict)
                    and project.get("title")
                    and project["title"] not in existing_titles
                ):
                    existing_projects.append(project)
                    existing_titles.add(project["title"])
                elif (
                    isinstance(project, dict)
                    and project.get("title")
                    and project["title"] in existing_titles
                ):
                    # Merge details for duplicate project
                    for existing_proj in existing_projects:
                        if existing_proj.get("title") == project["title"]:
                            merge_project_details([existing_proj], project)
                            break
        else:
            merged_structures[field_name] = item

    return merged_structures
