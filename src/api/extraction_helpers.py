"""
Helper functions for extraction endpoints.

This module contains refactored helper functions to break down
the large extraction functions in routes.py.
"""

import json
import re
import time
from functools import lru_cache
from typing import Any

from src.core.config import (
    AGGREGATION_CHUNK_SIZE,
    CONFIDENCE_THRESHOLD,
    MIN_QUOTE_LENGTH,
    QUOTE_MATCH_THRESHOLD,
    USE_CHAIN_OF_THOUGHT,
)
from src.extraction.structure_extractor import (
    extract_action_fields_only,
    extract_project_details,
    extract_project_details_cot,
    extract_projects_for_field,
    extract_structures_with_retry,
)


def extract_all_action_fields(
    chunks: list[str],
    log_file_path: str | None = None,
    log_context_prefix: str | None = None,
) -> list[str]:
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

    action_fields = extract_action_fields_only(
        chunks, log_file_path, log_context_prefix
    )

    if not action_fields:
        print("⚠️ No action fields found in Stage 1")

    return action_fields


def extract_projects_and_details(
    chunks: list[str],
    action_fields: list[str],
    log_file_path: str | None = None,
    log_context_prefix: str | None = None,
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

        # Use Chain-of-Thought extraction if enabled
        if USE_CHAIN_OF_THOUGHT:
            print(
                f"   🧠 Using Chain-of-Thought classification (confidence ≥ {CONFIDENCE_THRESHOLD})"
            )
            details = extract_project_details_cot(
                chunks, action_field, project_title, CONFIDENCE_THRESHOLD
            )
        else:
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

    # Log the input for debugging
    print(f"\n📊 Deduplicating {len(extracted_data)} action fields...")
    field_names = [item["action_field"] for item in extracted_data]
    unique_names = set(field_names)
    print(f"   Found {len(unique_names)} unique action field names")
    if len(unique_names) < len(extracted_data):
        print(
            f"   ⚠️ Duplicate names found: {len(extracted_data) - len(unique_names)} duplicates will be merged"
        )

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

    result = list(deduplicated_data.values())
    print(
        f"   ✅ Deduplication complete: {len(extracted_data)} → {len(result)} action fields"
    )

    return result


def validate_german_only_content(
    action_fields: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Final validation to remove any English contamination from action fields.

    This serves as a safety net to catch any English content that might have
    slipped through the LLM aggregation process.
    """
    # More comprehensive list with word boundaries
    english_terms = [
        "current",
        "future",
        "state",
        "vision",
        "enhanced",
        "new",
        "findings",
        "data",
        "urban",
        "mobility",
        "plan",
        "strategy",
        "analysis",
        "report",
        "overview",
        "summary",
        "background",
        "energy",
        "development",
        "framework",
        "concept",
        "program",
        "initiative",
        "approach",
        "implementation",
        "assessment",
        "review",
    ]

    # Create pattern with word boundaries
    english_pattern = re.compile(
        r"\b(" + "|".join(re.escape(term) for term in english_terms) + r")\b",
        re.IGNORECASE,
    )

    validated_fields = []

    for action_field in action_fields:
        field_name = action_field.get("action_field", "")

        # Check if action field contains English terms (with word boundaries)
        if english_pattern.search(field_name):
            print(f"🚫 FINAL FILTER: Removing English action field '{field_name}'")
            continue

        # Also validate project titles
        validated_projects = []
        for project in action_field.get("projects", []):
            project_title = project.get("title", "")

            if english_pattern.search(project_title):
                print(f"🚫 FINAL FILTER: Removing English project '{project_title}'")
                continue

            validated_projects.append(project)

        # Only include action field if it has valid projects
        if validated_projects:
            validated_field = action_field.copy()
            validated_field["projects"] = validated_projects
            validated_fields.append(validated_field)
        else:
            print(
                f"🚫 FINAL FILTER: Removing action field '{field_name}' - no valid projects"
            )

    return validated_fields


def chunked_aggregation(
    all_chunk_results: list[dict[str, Any]],
    chunk_size: int = AGGREGATION_CHUNK_SIZE,  # Keep parameter for compatibility
    recursion_depth: int = 0,
    max_recursion: int = 3,  # Increased for binary split depth
    log_file_path: str | None = None,
    log_context_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """
    Handle large datasets by recursively processing them using binary splitting.
    Always processes ALL input data by splitting in half for balanced aggregation.
    """
    num_fields = len(all_chunk_results)
    print(
        f"🔄 Processing {num_fields} action fields (recursion depth: {recursion_depth})"
    )

    # Base cases
    if recursion_depth >= max_recursion:
        print(
            f"⚠️ Maximum recursion depth ({max_recursion}) reached - falling back to simple deduplication"
        )
        return simple_deduplication_fallback(all_chunk_results)

    if num_fields <= 20:
        print(f"✅ Already at target size ({num_fields} ≤ 20), returning as is")
        return all_chunk_results

    # Binary split for balanced processing
    if num_fields > 20:
        mid_point = num_fields // 2
        first_half = all_chunk_results[:mid_point]
        second_half = all_chunk_results[mid_point:]

        print(
            f"📊 Splitting into two halves: {len(first_half)} and {len(second_half)} action fields"
        )

        intermediate_results = []

        # Process first half
        print(f"📋 Processing first half with {len(first_half)} action fields")
        chunk_data = json.dumps(first_half, indent=2, ensure_ascii=False)
        log_context = (
            f"{log_context_prefix} - Aggregation Pass {recursion_depth + 1}, First Half ({len(first_half)} fields)"
            if log_context_prefix
            else None
        )
        result = perform_single_aggregation(chunk_data, log_file_path, log_context)

        if result:
            intermediate_results.extend(result)
            print(f"   ✅ First half aggregated to {len(result)} action fields")
        else:
            print("   ❌ First half aggregation failed, retaining original data")
            intermediate_results.extend(first_half)

        # Process second half
        print(f"📋 Processing second half with {len(second_half)} action fields")
        chunk_data = json.dumps(second_half, indent=2, ensure_ascii=False)
        log_context = (
            f"{log_context_prefix} - Aggregation Pass {recursion_depth + 1}, Second Half ({len(second_half)} fields)"
            if log_context_prefix
            else None
        )
        result = perform_single_aggregation(chunk_data, log_file_path, log_context)

        if result:
            intermediate_results.extend(result)
            print(f"   ✅ Second half aggregated to {len(result)} action fields")
        else:
            print("   ❌ Second half aggregation failed, retaining original data")
            intermediate_results.extend(second_half)

        print(
            f"✅ Pass {recursion_depth + 1} completed: {len(intermediate_results)} intermediate results"
        )

        # Recursively aggregate if still too many
        if len(intermediate_results) > 20 and len(intermediate_results) < num_fields:
            return chunked_aggregation(
                intermediate_results,
                chunk_size,
                recursion_depth + 1,
                max_recursion,
                log_file_path,
                log_context_prefix,
            )
        elif len(intermediate_results) > 20:
            print("⚠️ Aggregation stalled, falling back to simple deduplication")
            return simple_deduplication_fallback(intermediate_results)

        return intermediate_results

    return all_chunk_results


def perform_single_aggregation(
    chunk_data: str,
    log_file_path: str | None = None,
    log_context: str | None = None,
    override_output_tokens: int | None = None,
) -> list[dict[str, Any]] | None:
    """
    Perform a single aggregation pass on JSON data.
    """
    from src.core.config import MODEL_TEMPERATURE
    from src.core.llm_providers import get_llm_provider
    from src.core.schemas import ExtractionResult

    system_message = """Sie sind ein Experte für die KONSERVATIVE Deduplizierung von Handlungsfeldern aus deutschen kommunalen Strategiedokumenten.
Ihre Hauptaufgabe ist es, DUPLIKATE zu entfernen und die GRANULARITÄT zu erhalten.
Konsolidieren Sie NUR offensichtliche Duplikate oder fast identische Felder.
Behalten Sie die meisten Handlungsfelder bei - Reduzierung um maximal 30-40%.
Antworten Sie AUSSCHLIESSLICH mit einem JSON-Objekt, das der vorgegebenen Struktur entspricht.
KEIN zusätzlicher Text, KEINE Erklärungen, NUR JSON."""

    # Count the actual action fields
    action_field_count = chunk_data.count('"action_field"')

    # Calculate target range (keep 70-80% of fields)
    min_target = int(action_field_count * 0.7)
    max_target = int(action_field_count * 0.8)

    prompt = f"""Sie erhalten {action_field_count} Handlungsfelder zur Konsolidierung.

KRITISCH: Sie MÜSSEN mindestens {min_target} bis {max_target} Handlungsfelder in Ihrer Antwort zurückgeben!

Ihre Aufgabe:
1. Entfernen Sie NUR exakte Duplikate (identischer Name)
2. Konsolidieren Sie NUR fast identische Felder (>90% Überlappung)
3. ALLE anderen Felder müssen SEPARAT bleiben

STRENGE VORGABE:
- Input: {action_field_count} Felder
- Output: MINDESTENS {min_target} Felder (besser {max_target})
- Wenn Sie weniger als {min_target} Felder zurückgeben, ist die Aufgabe NICHT erfüllt!

BEISPIELE was NICHT konsolidiert werden darf:
❌ "Klimaschutz" und "Energie" → MÜSSEN getrennt bleiben
❌ "Mobilität" und "Verkehr" → MÜSSEN getrennt bleiben
❌ "Umwelt" und "Nachhaltigkeit" → MÜSSEN getrennt bleiben
❌ "Stadtentwicklung" und "Wohnen" → MÜSSEN getrennt bleiben

NUR diese Fälle dürfen konsolidiert werden:
✅ "Mobilität" + "Mobilität" → "Mobilität" (identisch)
✅ "Klimaschutz und Energie" + "Energie und Klimaschutz" → "Klimaschutz und Energie" (gleicher Inhalt)

Beachten Sie folgende KONSERVATIVE Konsolidierungsregeln:

NUR DIESE FÄLLE konsolidieren:
✅ "Mobilität" + "Mobilität" → "Mobilität" (exaktes Duplikat)
✅ "Verkehr und Mobilität" + "Mobilität und Verkehr" → "Mobilität und Verkehr" (fast identisch)
✅ "Klimaschutz" + "Klimaschutz und Energie" → "Klimaschutz und Energie" (eines ist Teilmenge)

DIESE FÄLLE NICHT konsolidieren:
❌ "Klimaschutz" und "Energie" → Bleiben getrennt (verschiedene Schwerpunkte)
❌ "Umwelt" und "Nachhaltigkeit" → Bleiben getrennt (unterschiedliche Aspekte)
❌ "Mobilität" und "ÖPNV" → Bleiben getrennt (allgemein vs. spezifisch)
❌ "Verkehr" und "Radverkehr" → Bleiben getrennt (allgemein vs. spezifisch)
❌ "Wirtschaft" und "Innovation" → Bleiben getrennt (verschiedene Bereiche)
❌ "Kultur" und "Bildung" → Bleiben getrennt (verschiedene Ressorts)
❌ "Wohnen" und "Stadtentwicklung" → Bleiben getrennt (verschiedene Planungsebenen)
❌ "Soziales" und "Gesundheit" → Bleiben getrennt (verschiedene Zuständigkeiten)
❌ "Integration" und "Teilhabe" → Bleiben getrennt (verschiedene Konzepte)

Hier sind die Handlungsfelder zur Konsolidierung:
{chunk_data}

Antworten Sie AUSSCHLIESSLICH mit einem JSON-Objekt, das die konsolidierten Handlungsfelder enthält. KEIN zusätzlicher Text, KEINE Erklärungen, NUR JSON.
"""

    try:
        # Check input size
        data_size = len(chunk_data)
        print(f"   🔍 Attempting aggregation with {data_size} characters of JSON")

        # Calculate dynamic num_predict based on input size
        # Improved token estimation: 1 token ≈ 3.5 characters for German text
        estimated_input_tokens = int(data_size / 3.5)
        # Use override if provided, otherwise estimate 50-80% of input size
        if override_output_tokens:
            dynamic_num_predict = override_output_tokens
        else:
            estimated_output_tokens = int(estimated_input_tokens * 0.8)
            # Cap at 30720 (75% of 40K context) to leave room for prompt
            dynamic_num_predict = max(8192, min(estimated_output_tokens, 30720))
        print(
            f"   📊 Input: ~{estimated_input_tokens} tokens, Output: ~{dynamic_num_predict} tokens"
        )

        # Update log context to include token info
        enhanced_log_context = (
            f"{log_context} - Dynamic tokens: {dynamic_num_predict}"
            if log_context
            else f"Dynamic tokens: {dynamic_num_predict}"
        )

        llm_provider = get_llm_provider()
        result = llm_provider.query_structured(
            prompt=prompt,
            response_model=ExtractionResult,
            system_message=system_message,
            log_file_path=log_file_path,
            log_context=enhanced_log_context,
            override_num_predict=dynamic_num_predict,
        )

        if result:
            # Convert to the expected format
            aggregated_data = []
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
                aggregated_data.append(action_field_dict)

            # Calculate actual output token length
            output_json = json.dumps(aggregated_data, ensure_ascii=False)
            actual_output_tokens = int(len(output_json) / 3.5)
            print(
                f"   📊 Output: ~{actual_output_tokens} tokens (predicted: ~{dynamic_num_predict})"
            )

            # Check if aggregation was too aggressive
            input_count = chunk_data.count('"action_field"')
            output_count = len(aggregated_data)
            reduction_percent = ((input_count - output_count) / input_count) * 100

            if reduction_percent > 40:
                print(
                    f"   ⚠️ WARNING: Aggregation too aggressive! {input_count} → {output_count} fields ({reduction_percent:.1f}% reduction)"
                )
                print(
                    f"   ⚠️ Target was {int(input_count * 0.7)}-{int(input_count * 0.8)} fields"
                )
            else:
                print(
                    f"   ✅ LLM aggregation successful: {output_count} action fields ({reduction_percent:.1f}% reduction)"
                )

            return aggregated_data
        else:
            print("   ❌ LLM returned None - structured output failed")
            return None

    except Exception as e:
        print(f"   ❌ Single aggregation error: {type(e).__name__}: {e}")
        return None


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


class ExtractionChangeTracker:
    """Track changes in extraction results across chunks."""

    def __init__(self):
        self.history = []

    def track_changes(
        self, old_data: dict[str, Any], new_data: dict[str, Any], chunk_index: int
    ) -> dict[str, Any]:
        """
        Track changes between old and new extraction data.

        Returns a dictionary of changes for logging.
        """
        changes: dict[str, Any] = {
            "chunk": chunk_index + 1,
            "action_fields": {"added": [], "total": 0},
            "projects": {"added": 0, "enhanced": 0, "total": 0},
            "measures": {"added": 0, "total": 0},
            "indicators": {"added": 0, "total": 0},
        }

        old_afs = {af["action_field"]: af for af in old_data.get("action_fields", [])}
        new_afs = {af["action_field"]: af for af in new_data.get("action_fields", [])}

        # Track new action fields
        for af_name in new_afs:
            if af_name not in old_afs:
                changes["action_fields"]["added"].append(af_name)

        changes["action_fields"]["total"] = len(new_afs)

        # Track project changes
        for af_name, af_data in new_afs.items():
            old_af = old_afs.get(af_name, {"projects": []})
            old_projects = {p["title"]: p for p in old_af.get("projects", [])}
            new_projects = {p["title"]: p for p in af_data.get("projects", [])}

            for proj_name, proj_data in new_projects.items():
                if proj_name not in old_projects:
                    changes["projects"]["added"] += 1
                else:
                    # Check if enhanced with new measures/indicators
                    old_proj = old_projects[proj_name]
                    if len(proj_data.get("measures", [])) > len(
                        old_proj.get("measures", [])
                    ) or len(proj_data.get("indicators", [])) > len(
                        old_proj.get("indicators", [])
                    ):
                        changes["projects"]["enhanced"] += 1

                changes["measures"]["total"] += len(proj_data.get("measures", []))
                changes["indicators"]["total"] += len(proj_data.get("indicators", []))

            changes["projects"]["total"] += len(new_projects)

        # Calculate new additions
        if chunk_index > 0:  # Not first chunk
            for af_data in new_afs.values():
                for proj in af_data.get("projects", []):
                    old_af = old_afs.get(af_data["action_field"], {"projects": []})
                    old_proj = next(
                        (p for p in old_af["projects"] if p["title"] == proj["title"]),
                        None,
                    )

                    if old_proj:
                        old_measures = set(old_proj.get("measures", []))
                        new_measures = set(proj.get("measures", []))
                        changes["measures"]["added"] += len(new_measures - old_measures)

                        old_indicators = set(old_proj.get("indicators", []))
                        new_indicators = set(proj.get("indicators", []))
                        changes["indicators"]["added"] += len(
                            new_indicators - old_indicators
                        )
                    else:
                        changes["measures"]["added"] += len(proj.get("measures", []))
                        changes["indicators"]["added"] += len(
                            proj.get("indicators", [])
                        )

        self.history.append(changes)
        return changes

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all changes across chunks."""
        if not self.history:
            return {}

        return {
            "total_chunks": len(self.history),
            "final_counts": self.history[-1] if self.history else {},
            "progression": [
                {
                    "chunk": h["chunk"],
                    "action_fields": h["action_fields"]["total"],
                    "projects": h["projects"]["total"],
                    "indicators": h["indicators"]["total"],
                }
                for h in self.history
            ],
        }


def aggregate_extraction_results(
    all_chunk_results: list[dict[str, Any]],
    log_file_path: str | None = None,
    log_context_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """
    Aggregate extraction results using intelligent context-aware processing.

    Uses single-pass aggregation when data fits in context window (40K tokens),
    falling back to chunked approach only for genuinely large datasets.
    """
    if not all_chunk_results:
        return []

    print(f"🔄 Aggregating {len(all_chunk_results)} extracted action fields...")

    # Check if data fits in context window for single-pass processing
    chunk_data = json.dumps(all_chunk_results, indent=2, ensure_ascii=False)
    data_size = len(chunk_data)
    estimated_tokens = int(data_size / 3.5)  # Conservative token estimation

    # qwen3:14b-AWQ: 32K total context
    # Reserve: 2K prompt + 15K output = 17K overhead
    # Safe input limit: 15K tokens (32K - 17K overhead)
    context_safety_limit = 15000

    print(
        f"📊 Aggregation data analysis: {data_size} characters ≈ {estimated_tokens} tokens"
    )
    print(
        f"📊 Context safety limit: {context_safety_limit} tokens (leaves 17K for prompt+output)"
    )
    print(f"📊 Total continuous input length: {len(all_chunk_results)} action fields")

    if estimated_tokens <= context_safety_limit:
        # Calculate remaining tokens for output (32K context)
        remaining_tokens = 32768 - estimated_tokens - 2000  # Reserve 2K for prompt
        output_tokens = min(
            remaining_tokens - 1000, 15000
        )  # Cap output, leave 1K buffer
        print(f"📊 Output allocation: {output_tokens} tokens available")
        print("✅ Data fits in context window - using single-pass aggregation")

        result = perform_single_aggregation(
            chunk_data,
            log_file_path,
            (
                f"{log_context_prefix} - Single Pass ({estimated_tokens}→{output_tokens} tokens)"
                if log_context_prefix
                else f"Single Pass ({estimated_tokens}→{output_tokens} tokens)"
            ),
            override_output_tokens=output_tokens,
        )
    else:
        print(
            f"⚠️ Data too large ({estimated_tokens} tokens) - falling back to chunked aggregation"
        )
        result = chunked_aggregation(
            all_chunk_results,
            log_file_path=log_file_path,
            log_context_prefix=log_context_prefix,
        )

    if result:
        # Final validation to remove any English contamination
        validated_data = validate_german_only_content(result)
        print(f"✅ Aggregated to {len(validated_data)} clean German action fields")

        # Apply entity resolution to fix node fragmentation
        resolved_data = apply_entity_resolution_with_monitoring(validated_data)

        # Apply consistency validation to fix uneven edge distribution
        consistent_data = apply_consistency_validation(resolved_data)
        return consistent_data
    else:
        print("⚠️ Single aggregation failed, using simple deduplication fallback")
        fallback_data = simple_deduplication_fallback(all_chunk_results)

        # Apply entity resolution to fallback data as well
        resolved_data = apply_entity_resolution_with_monitoring(fallback_data)

        # Apply consistency validation to fallback data as well
        consistent_data = apply_consistency_validation(resolved_data)
        return consistent_data


def apply_entity_resolution_with_monitoring(
    structures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Apply entity resolution with quality monitoring and reporting.

    Args:
        structures: List of extraction structures

    Returns:
        Structures with resolved entities
    """
    from src.core.config import (
        ENTITY_RESOLUTION_ENABLED,
        ENTITY_RESOLUTION_RESOLVE_ACTION_FIELDS,
        ENTITY_RESOLUTION_RESOLVE_PROJECTS,
    )
    from src.processing.entity_resolver import resolve_extraction_entities
    from src.utils.graph_quality import (
        GraphQualityAnalyzer,
        analyze_graph_quality,
        print_improvement_report,
        print_quality_report,
    )

    if not ENTITY_RESOLUTION_ENABLED:
        return structures

    if not structures:
        return structures

    try:
        print("\n🔗 Applying entity resolution with quality monitoring...")

        # Measure quality before resolution
        before_metrics = analyze_graph_quality(structures, before_resolution=True)
        print_quality_report(before_metrics)

        # Apply entity resolution
        resolved_structures = resolve_extraction_entities(
            structures,
            resolve_action_fields=ENTITY_RESOLUTION_RESOLVE_ACTION_FIELDS,
            resolve_projects=ENTITY_RESOLUTION_RESOLVE_PROJECTS,
        )

        # Measure quality after resolution
        after_metrics = analyze_graph_quality(
            resolved_structures, before_resolution=False
        )
        print_quality_report(after_metrics)

        # Calculate and report improvements
        analyzer = GraphQualityAnalyzer()
        improvement_metrics = analyzer.compare_before_after(
            before_metrics, after_metrics
        )
        print_improvement_report(improvement_metrics)

        return resolved_structures

    except Exception as e:
        print(f"⚠️ Entity resolution failed: {e}")
        print("   Falling back to original structures")
        return structures


def apply_entity_resolution(structures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Apply entity resolution to fix node fragmentation issues (without monitoring).

    Args:
        structures: List of extraction structures

    Returns:
        Structures with resolved entities
    """
    from src.core.config import (
        ENTITY_RESOLUTION_ENABLED,
        ENTITY_RESOLUTION_RESOLVE_ACTION_FIELDS,
        ENTITY_RESOLUTION_RESOLVE_PROJECTS,
    )
    from src.processing.entity_resolver import resolve_extraction_entities

    if not ENTITY_RESOLUTION_ENABLED:
        return structures

    if not structures:
        return structures

    try:
        print(f"\n🔗 Applying entity resolution to {len(structures)} structures...")

        resolved_structures = resolve_extraction_entities(
            structures,
            resolve_action_fields=ENTITY_RESOLUTION_RESOLVE_ACTION_FIELDS,
            resolve_projects=ENTITY_RESOLUTION_RESOLVE_PROJECTS,
        )

        return resolved_structures

    except Exception as e:
        print(f"⚠️ Entity resolution failed: {e}")
        print("   Falling back to original structures")
        return structures


def rebuild_enhanced_structure_from_resolved(
    resolved_structures: list[dict[str, Any]], original_enhanced: Any
) -> Any:
    """
    Rebuild enhanced structure from resolved intermediate structures with unique ID validation.

    Args:
        resolved_structures: Entity-resolved structures in intermediate format
        original_enhanced: Original enhanced structure to use as template

    Returns:
        New enhanced structure with resolved entities and guaranteed unique IDs
    """
    from src.core.schemas import (
        ConnectionWithConfidence,
        EnhancedActionField,
        EnhancedIndicator,
        EnhancedMeasure,
        EnhancedProject,
        EnrichedReviewJSON,
    )

    # Track used IDs to prevent duplicates
    used_ids = set()
    id_counters = {"af": 0, "proj": 0, "msr": 0, "ind": 0}

    def get_unique_id(prefix: str, title: str) -> str:
        """Generate guaranteed unique ID"""
        # Try sequential numbering
        for i in range(1, 10000):  # Reasonable upper limit
            candidate_id = f"{prefix}_{i}"
            if candidate_id not in used_ids:
                used_ids.add(candidate_id)
                id_counters[prefix] = max(id_counters[prefix], i)
                return candidate_id

        # Fallback with title hash if sequential fails
        import hashlib

        hash_suffix = hashlib.md5(title.encode()).hexdigest()[:4]
        candidate_id = f"{prefix}_{hash_suffix}"

        # Ensure even hash-based IDs are unique
        counter = 1
        while candidate_id in used_ids:
            candidate_id = f"{prefix}_{hash_suffix}_{counter}"
            counter += 1

        used_ids.add(candidate_id)
        return candidate_id

    # Build new enhanced structure
    new_action_fields = []
    new_projects = []
    new_measures = []
    new_indicators = []

    # Maps for connection building
    entity_id_map = {}  # old_title -> new_id

    print(
        f"🔄 Rebuilding enhanced structure from {len(resolved_structures)} resolved entities..."
    )

    # First pass: Create all entities with unique IDs
    for structure in resolved_structures:
        action_field_title = structure.get("action_field", "")
        if not action_field_title:
            continue

        # Create or reuse action field
        af_id = entity_id_map.get(f"af:{action_field_title}")
        if not af_id:
            af_id = get_unique_id("af", action_field_title)
            entity_id_map[f"af:{action_field_title}"] = af_id

            new_action_fields.append(
                EnhancedActionField(
                    id=af_id,
                    content={"title": action_field_title},
                    connections=[],  # Will be populated in second pass
                )
            )

        # Process projects
        for project in structure.get("projects", []):
            project_title = project.get("title", "")
            if not project_title:
                continue

            proj_id = entity_id_map.get(f"proj:{project_title}")
            if not proj_id:
                proj_id = get_unique_id("proj", project_title)
                entity_id_map[f"proj:{project_title}"] = proj_id

                new_projects.append(
                    EnhancedProject(
                        id=proj_id,
                        content={"title": project_title},
                        connections=[],  # Will be populated in second pass
                    )
                )

            # Process measures
            for measure_title in project.get("measures", []):
                if (
                    not measure_title
                    or measure_title == "Information im Quelldokument nicht verfügbar"
                ):
                    continue  # Skip null values

                msr_id = entity_id_map.get(f"msr:{measure_title}")
                if not msr_id:
                    msr_id = get_unique_id("msr", measure_title)
                    entity_id_map[f"msr:{measure_title}"] = msr_id

                    new_measures.append(
                        EnhancedMeasure(
                            id=msr_id,
                            content={"title": measure_title},
                            connections=[],  # Will be populated in second pass
                        )
                    )

            # Process indicators
            for indicator_title in project.get("indicators", []):
                if (
                    not indicator_title
                    or indicator_title == "Information im Quelldokument nicht verfügbar"
                ):
                    continue  # Skip null values

                ind_id = entity_id_map.get(f"ind:{indicator_title}")
                if not ind_id:
                    ind_id = get_unique_id("ind", indicator_title)
                    entity_id_map[f"ind:{indicator_title}"] = ind_id

                    new_indicators.append(
                        EnhancedIndicator(
                            id=ind_id,
                            content={"title": indicator_title},
                            connections=[],  # Will be populated in second pass
                        )
                    )

    # Second pass: Build connections with deduplication
    af_lookup = {af.content["title"]: af for af in new_action_fields}
    proj_lookup = {p.content["title"]: p for p in new_projects}
    msr_lookup = {m.content["title"]: m for m in new_measures}
    ind_lookup = {i.content["title"]: i for i in new_indicators}

    for structure in resolved_structures:
        action_field_title = structure.get("action_field", "")
        if action_field_title not in af_lookup:
            continue

        action_field = af_lookup[action_field_title]

        for project in structure.get("projects", []):
            project_title = project.get("title", "")
            if not project_title or project_title not in proj_lookup:
                continue

            project_node = proj_lookup[project_title]

            # Add AF -> Project connection (avoid duplicates)
            proj_connection = ConnectionWithConfidence(
                target_id=project_node.id, confidence_score=0.9
            )
            if not any(
                c.target_id == project_node.id for c in action_field.connections
            ):
                action_field.connections.append(proj_connection)

            # Add Project -> Measure connections
            for measure_title in project.get("measures", []):
                if measure_title and measure_title in msr_lookup:
                    measure_node = msr_lookup[measure_title]
                    msr_connection = ConnectionWithConfidence(
                        target_id=measure_node.id, confidence_score=0.8
                    )
                    if not any(
                        c.target_id == measure_node.id for c in project_node.connections
                    ):
                        project_node.connections.append(msr_connection)

            # Add Project -> Indicator connections
            for indicator_title in project.get("indicators", []):
                if indicator_title and indicator_title in ind_lookup:
                    indicator_node = ind_lookup[indicator_title]
                    ind_connection = ConnectionWithConfidence(
                        target_id=indicator_node.id, confidence_score=0.8
                    )
                    if not any(
                        c.target_id == indicator_node.id
                        for c in project_node.connections
                    ):
                        project_node.connections.append(ind_connection)

                # Add Measure -> Indicator connections (measures within this project may connect to indicators)
                for measure_title in project.get("measures", []):
                    if measure_title and measure_title in msr_lookup:
                        measure_node = msr_lookup[measure_title]
                        for indicator_title in project.get("indicators", []):
                            if indicator_title and indicator_title in ind_lookup:
                                indicator_node = ind_lookup[indicator_title]
                                msr_ind_connection = ConnectionWithConfidence(
                                    target_id=indicator_node.id, confidence_score=0.75
                                )
                                if not any(
                                    c.target_id == indicator_node.id
                                    for c in measure_node.connections
                                ):
                                    measure_node.connections.append(msr_ind_connection)

    # Validate all IDs are unique
    all_ids = (
        [af.id for af in new_action_fields]
        + [p.id for p in new_projects]
        + [m.id for m in new_measures]
        + [i.id for i in new_indicators]
    )

    if len(all_ids) != len(set(all_ids)):
        msg = "❌ CRITICAL: Duplicate IDs found after rebuild - this should never happen!"
        raise ValueError(
            msg
        )

    print(
        f"✅ Rebuilt with unique IDs: {len(new_action_fields)} AF, {len(new_projects)} P, {len(new_measures)} M, {len(new_indicators)} I"
    )
    print(f"🆔 ID validation: {len(all_ids)} total IDs, all unique")

    return EnrichedReviewJSON(
        action_fields=new_action_fields,
        projects=new_projects,
        measures=new_measures,
        indicators=new_indicators,
    )


def apply_consistency_validation(
    structures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Apply extraction consistency validation to improve edge distribution.

    Args:
        structures: List of extraction structures

    Returns:
        Structures with improved consistency
    """
    from src.processing.extraction_consistency import validate_extraction_consistency

    if not structures:
        return structures

    try:
        consistent_structures = validate_extraction_consistency(structures)
        return consistent_structures

    except Exception as e:
        print(f"⚠️ Consistency validation failed: {e}")
        print("   Falling back to original structures")
        return structures


@lru_cache(maxsize=1)
def _get_english_pattern():
    """Get compiled English pattern (cached for performance)."""
    english_terms = [
        "current",
        "future",
        "enhanced",
        "new findings",
        "overview",
        "summary",
        "background",
        "framework",
        "implementation",
        "assessment",
        "review",
    ]
    pattern = r"\b(" + "|".join(re.escape(term) for term in english_terms) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def simple_deduplication_fallback(
    all_chunk_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fallback deduplication if LLM aggregation fails."""
    deduplicated_data: dict[str, Any] = {}

    # Use cached compiled pattern
    english_pattern = _get_english_pattern()

    for item in all_chunk_results:
        field_name = item.get("action_field", "")

        # Skip English action fields using whole-word matching
        if english_pattern.search(field_name):
            print(f"🚫 FALLBACK FILTER: Removing English action field '{field_name}'")
            continue

        if field_name in deduplicated_data:
            # Merge projects
            existing_projects = deduplicated_data[field_name]["projects"]
            new_projects = item.get("projects", [])

            existing_titles = {p["title"] for p in existing_projects}
            for project in new_projects:
                if project["title"] not in existing_titles:
                    existing_projects.append(project)
        else:
            deduplicated_data[field_name] = item

    return list(deduplicated_data.values())


def add_source_attributions(
    final_results: list[dict[str, Any]],
    page_aware_chunks: list[dict[str, Any]],
    original_page_text: list[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """
    Add source attribution with page numbers to extracted results.

    Args:
        final_results: Final extraction results after deduplication
        page_aware_chunks: LLM chunks with page metadata (text, pages, chunk_id)
        original_page_text: Optional list of (text, page_num) tuples for precise page attribution

    Returns:
        Enhanced results with source attribution
    """
    import re
    from difflib import SequenceMatcher

    from src.core.schemas import SourceAttribution

    print(
        f"🔍 Adding source attribution for {len(final_results)} action fields using {len(page_aware_chunks)} LLM chunks"
    )
    if original_page_text:
        print(
            f"   📄 Using {len(original_page_text)} original pages for precise attribution"
        )

    def find_quote_page(
        quote: str, original_pages: list[tuple[str, int]], fallback_pages: list[int]
    ) -> int:
        """
        Find the specific page number where a quote appears.

        Args:
            quote: The quote to search for
            original_pages: List of (text, page_num) tuples
            fallback_pages: Pages from the LLM chunk to use if quote not found

        Returns:
            The page number where the quote was found, or fallback to first page
        """
        if not original_pages or not quote:
            return fallback_pages[len(fallback_pages) // 2] if fallback_pages else 1

        # Clean the quote for better matching
        clean_quote = quote.strip()
        if len(clean_quote) < 15:  # Too short for reliable matching
            return fallback_pages[len(fallback_pages) // 2] if fallback_pages else 1

        # Try to find a significant portion of the quote (at least 50 chars)
        search_text = clean_quote[: min(100, len(clean_quote))]

        best_match_page = None
        best_match_score = 0.0

        for page_text, page_num in original_pages:
            # Skip if page is not in the LLM chunk's page range
            if fallback_pages and page_num not in fallback_pages:
                continue

            # Use SequenceMatcher for fuzzy matching
            matcher = SequenceMatcher(None, search_text.lower(), page_text.lower())

            # Find the best matching block
            match = matcher.find_longest_match(0, len(search_text), 0, len(page_text))

            # Calculate match score (ratio of matched length to search length)
            if match.size > 0:
                score = match.size / len(search_text)

                # If we find a very good match (>80%), use it immediately
                if score > 0.8:
                    return page_num

                # Track the best match
                if score > best_match_score:
                    best_match_score = score
                    best_match_page = page_num

        # Use best match if reasonably good (above threshold), otherwise fallback
        if best_match_page and best_match_score > QUOTE_MATCH_THRESHOLD:
            return best_match_page
        else:
            # Fallback to middle page of the LLM chunk
            return fallback_pages[len(fallback_pages) // 2] if fallback_pages else 1

    enhanced_results = []

    for action_field in final_results:
        enhanced_field = action_field.copy()
        enhanced_projects = []

        for project in action_field.get("projects", []):
            enhanced_project = project.copy()
            project_title = project.get("title", "")

            # Find source chunks for this project
            project_sources = []

            # Search for project title in chunks
            for chunk in page_aware_chunks:
                chunk_text = chunk.get("text", "")
                chunk_pages = chunk.get("pages", [])
                chunk_id = chunk.get("chunk_id")

                # Check if project title appears in chunk (case-insensitive)
                if project_title.lower() in chunk_text.lower():
                    # Find the best matching excerpt around the project title
                    quote = extract_relevant_quote(chunk_text, project_title)

                    if quote and chunk_pages:
                        # Find the specific page where this quote appears
                        if original_page_text:
                            page_number = find_quote_page(
                                quote, original_page_text, chunk_pages
                            )
                        else:
                            # Fallback to middle page of the LLM chunk
                            page_number = chunk_pages[len(chunk_pages) // 2]

                        project_sources.append(
                            SourceAttribution(
                                page_number=page_number, quote=quote, chunk_id=chunk_id
                            )
                        )

            # Also search for measures and indicators
            for measure in project.get("measures", []):
                if len(project_sources) >= 3:  # Limit to avoid too many sources
                    break

                for chunk in page_aware_chunks:
                    chunk_text = chunk.get("text", "")
                    chunk_pages = chunk.get("pages", [])
                    chunk_id = chunk.get("chunk_id")

                    # Check for measure content (partial matching)
                    if measure_appears_in_chunk(measure, chunk_text):
                        quote = extract_relevant_quote(chunk_text, measure)

                        if quote and chunk_pages:
                            # Find the specific page where this quote appears
                            if original_page_text:
                                page_number = find_quote_page(
                                    quote, original_page_text, chunk_pages
                                )
                            else:
                                # Fallback to middle page of the LLM chunk
                                page_number = chunk_pages[len(chunk_pages) // 2]

                            # Avoid duplicate pages
                            existing_pages = {
                                src.page_number for src in project_sources
                            }
                            if page_number not in existing_pages:
                                project_sources.append(
                                    SourceAttribution(
                                        page_number=page_number,
                                        quote=quote,
                                        chunk_id=chunk_id,
                                    )
                                )
                                break

            # Add sources to project if found
            if project_sources:
                # Sort by page number and limit to top 3 sources
                project_sources.sort(key=lambda x: x.page_number)
                enhanced_project["sources"] = [
                    source.model_dump() for source in project_sources[:3]
                ]
                print(
                    f"   ✅ Found {len(enhanced_project['sources'])} sources for '{project_title}'"
                )
            else:
                print(f"   ⚠️ No sources found for '{project_title}'")

            enhanced_projects.append(enhanced_project)

        enhanced_field["projects"] = enhanced_projects
        enhanced_results.append(enhanced_field)

    attribution_stats = {
        "total_projects": sum(len(af["projects"]) for af in enhanced_results),
        "projects_with_sources": sum(
            1
            for af in enhanced_results
            for project in af["projects"]
            if project.get("sources")
        ),
    }

    print(
        f"📊 Attribution complete: {attribution_stats['projects_with_sources']}/{attribution_stats['total_projects']} projects have sources"
    )

    return enhanced_results


def measure_appears_in_chunk(measure: str, chunk_text: str) -> bool:
    """
    Check if a measure appears in chunk text using fuzzy matching.

    Args:
        measure: The measure text to search for
        chunk_text: The chunk text to search in

    Returns:
        True if measure likely appears in chunk
    """
    # Simple keyword-based matching for German text
    measure_words = measure.lower().split()
    chunk_lower = chunk_text.lower()

    # Check if most measure words appear in chunk
    found_words = sum(
        1 for word in measure_words if len(word) > 3 and word in chunk_lower
    )
    return found_words >= len(measure_words) * 0.6  # 60% of words must match


def extract_relevant_quote(text: str, search_term: str, max_length: int = 300) -> str:
    """
    Extract the most relevant quote from text around the search term.

    Args:
        text: The full text to search in
        search_term: The term to search for
        max_length: Maximum length of quote

    Returns:
        Most relevant quote or empty string if not found
    """
    text_lower = text.lower()
    search_lower = search_term.lower()

    # Find all occurrences
    occurrences = []
    pattern = re.escape(search_lower)
    for match in re.finditer(pattern, text_lower):
        occurrences.append(match.start())

    if not occurrences:
        # Fallback to fuzzy matching (existing code)
        lines = text.split("\n")
        for line in lines:
            if any(
                word in line.lower() for word in search_lower.split() if len(word) > 3
            ):
                clean_line = line.strip()
                if len(clean_line) > max_length:
                    clean_line = clean_line[:max_length] + "..."
                return clean_line
        return ""

    # Find best occurrence by checking for sentence boundaries
    best_quote = ""
    best_score = 0.0

    for pos in occurrences:
        # Extract context
        start = max(0, pos - max_length // 2)
        end = min(len(text), pos + len(search_term) + max_length // 2)

        # Try to find sentence boundaries
        quote_start = start
        quote_end = end

        # Look for sentence start
        for i in range(start, max(0, start - 100), -1):
            if i == 0 or text[i - 1] in ".!?\n":
                quote_start = i
                break

        # Look for sentence end
        for i in range(end, min(len(text), end + 100)):
            if i == len(text) - 1 or text[i] in ".!?\n":
                quote_end = i + 1
                break

        quote = text[quote_start:quote_end].strip()

        # Score based on completeness and relevance
        score = 1.0
        if quote_start > 0:
            score *= 0.9  # Penalty for truncated start
        if quote_end < len(text):
            score *= 0.9  # Penalty for truncated end

        # Bonus for complete sentences
        if quote.endswith((".", "!", "?")):
            score *= 1.1

        # Bonus for quotes that contain the search term in the middle (not at edges)
        term_pos_in_quote = quote.lower().find(search_lower)
        if (
            term_pos_in_quote > 10
            and term_pos_in_quote < len(quote) - len(search_term) - 10
        ):
            score *= 1.2

        if score > best_score:
            best_score = score
            best_quote = quote

    # Clean up the quote
    if len(best_quote) > max_length:
        # Truncate at word boundary
        truncated = best_quote[:max_length].rsplit(" ", 1)[0]
        best_quote = truncated + "..."

    # Add ellipsis if needed
    if best_quote and not best_quote.startswith(text[:10]):
        best_quote = "..." + best_quote
    if best_quote and not best_quote.endswith(text[-10:]):
        best_quote = best_quote + "..."

    # Remove excessive whitespace
    best_quote = " ".join(best_quote.split())

    return best_quote


def transform_to_enhanced_structure(
    intermediate_data: dict,
    log_file_path: str | None = None,
    log_context: str | None = None,
):
    """
    Transform intermediate extraction results into enhanced 4-bucket relational structure.

    This is the "Transformer LLM" from the two-layer pipeline strategy that takes
    the flawed nested JSON and converts it to clean relational structure.

    Args:
        intermediate_data: The intermediate extraction JSON data
        log_file_path: Optional path to log LLM dialog
        log_context: Optional context for logging

    Returns:
        EnrichedReviewJSON structure or None if transformation fails
    """
    import json

    from src.core.config import LLM_BACKEND, MODEL_TEMPERATURE
    from src.core.llm_providers import get_llm_provider
    from src.core.schemas import EnrichedReviewJSON

    # Extract the structures from intermediate data
    structures = intermediate_data.get("structures", [])
    if not structures:
        print("⚠️ No structures found in intermediate data")
        return None

    print(f"🔄 Transforming {len(structures)} action fields to enhanced structure...")

    # Use enhanced prompts for external models with better reasoning capabilities
    if LLM_BACKEND in ["openai", "gemini"]:
        system_message = """Sie sind ein Experte für die Transformation deutscher kommunaler Strategiedokumente in detaillierte Datenbank-Strukturen.

Ihre Mission: Erstellen Sie eine umfassende, relationale 4-Bucket-Struktur mit maximaler Informationsdichte und Erklärungen.

TRANSFORMATION ZIELE:
1. action_fields - Handlungsfelder mit strategischem Kontext und Nachhaltigkeitstypen
2. projects - Projekte mit vollständigen Implementierungsdetails (Budget, Zeitpläne, Verantwortlichkeiten)
3. measures - Konkrete Maßnahmen mit detaillierten Beschreibungen und Umsetzungswegen
4. indicators - Quantitative Metriken mit Berechnungsformeln, Zielwerten und Datenquellen

ERWEITERTE ANFORDERUNGEN:
1. VOLLSTÄNDIGE BESCHREIBUNGEN: Generieren Sie umfassende, erklärende Beschreibungen für alle Entitäten
2. REICHE METADATEN: Extrahieren und inferieren Budget, Status, Verantwortlichkeiten, Zeitpläne, SDG-Bezüge
3. QUANTITATIVE DETAILS: Erfassen Sie Zielwerte, Berechnungsmethoden, Einheiten, Granularität
4. STRATEGISCHE VERBINDUNGEN: Identifizieren Sie Synergien und hierarchische Beziehungen
5. NACHHALTIGKEITSKONTEXT: Ordnen Sie Handlungsfelder Nachhaltigkeitstypen (Umwelt/Sozial/Wirtschaft) zu

DATENQUALITÄT:
- Inferenz erlaubt: Ergänzen Sie sinnvolle Details basierend auf dem Kontext
- Quellenvalidierung: Entfernen Sie Metadaten, Header, Einzelwörter
- Explizite Verbindungen: Alle Connections mit präzisen Konfidenz-Scores (0.0-1.0)
- ID-Format: af_1, proj_1, msr_1, ind_1

Antworten Sie AUSSCHLIESSLICH mit einem JSON-Objekt, das der EnrichedReviewJSON-Struktur entspricht."""
    else:
        # Simpler prompt for local models
        system_message = """Sie sind ein Experte für die Transformation von deutschen kommunalen Strategiedokumenten in eine relationale Datenstruktur.

Ihre Aufgabe ist es, die verschachtelte JSON-Struktur in vier separate, miteinander verknüpfte Listen zu konvertieren:
1. action_fields - Handlungsfelder mit eindeutigen IDs
2. projects - Projekte mit eindeutigen IDs
3. measures - Maßnahmen mit eindeutigen IDs
4. indicators - Indikatoren mit eindeutigen IDs

KRITISCHE Anforderungen:
1. Konservative Deduplizierung: NUR offensichtliche Duplikate zusammenführen
2. Quellenvalidierung: Entfernen Sie Dokumentmetadaten, Header, Einzelwörter
3. Explizite Verbindungen: Jede Verbindung braucht Konfidenz-Score (0.0-1.0)
4. ID-Format: af_1, proj_1, msr_1, ind_1

Antworten Sie AUSSCHLIESSLICH mit einem JSON-Objekt, das der EnrichedReviewJSON-Struktur entspricht."""

    # Create the transformation prompt
    structures_json = json.dumps(structures, indent=2, ensure_ascii=False)

    # Enhanced prompt for external models with detailed field requirements
    if LLM_BACKEND in ["openai", "gemini"]:
        prompt = f"""Transformieren Sie diese verschachtelte Struktur in das neue relationale 4-Bucket-Format mit vollständigen Appwrite-kompatiblen Feldern:

{structures_json}

DETAILLIERTE FELD-ANFORDERUNGEN:

ACTION FIELDS (Handlungsfelder):
- name* (str): Handlungsfeldname
- description* (str): Umfassende Erklärung des Handlungsfelds (2-3 Sätze)
- sustainability_type (str): "Environmental", "Social", "Economic" - inferieren Sie basierend auf Kontext
- strategic_goals (list[str]): Strategische Ziele, die dieses Feld adressiert
- sdgs (list[str]): UN-Nachhaltigkeitsziele (z.B. ["SDG 11", "SDG 13"])
- parent_dimension_id (str): Hierarchische Beziehungen, falls vorhanden
- icon_ref (str): Thematisch passende Icon-Bezeichnung

PROJECTS (Projekte/Maßnahmen):
- title* (str): Projekttitel
- description (str): Kurze Projektbeschreibung (1-2 Sätze)
- full_description (str): Detaillierte Projektbeschreibung (3-5 Sätze)
- type* (str): "Infrastructure", "Policy", "Program", "Study" etc.
- status (str): "In Planung", "Aktiv", "Abgeschlossen", "Pausiert"
- measure_start (str): Startdatum (YYYY-MM-DD Format, inferieren wenn möglich)
- measure_end (str): Enddatum (YYYY-MM-DD Format, inferieren wenn möglich)
- budget (float): Projektbudget in Euro (inferieren/schätzen wenn Hinweise vorhanden)
- department (str): Zuständige Abteilung/Amt
- responsible_person (list[str]): Verantwortliche Personen
- operative_goal (str): Spezifisches operatives Ziel
- sdgs (list[str]): Relevante SDGs
- priority (str): "Hoch", "Mittel", "Niedrig"

INDICATORS (Indikatoren):
- title* (str): Indikator-Titel
- description* (str): Detaillierte Beschreibung was gemessen wird
- unit (str): Messeinheit (z.B. "Tonnen CO2/Jahr", "Anzahl", "Prozent")
- granularity* (str): "annual", "monthly", "quarterly", "continuous"
- target_values (str): Zielwerte mit Zeitbezug
- actual_values (str): Aktuelle/historische Werte
- should_increase (bool): true für positive Indikatoren, false für zu reduzierende
- calculation (str): Berechnungsformel oder -methode
- values_source (str): Datenquelle für die Messwerte
- operational_goal (str): Was konkret verfolgt wird
- sdgs (list[str]): Relevante SDGs

TRANSFORMATION REGELN:

1. INTELLIGENTE KONSOLIDIERUNG (3-Schritt-Prozess):
   SCHRITT 1: Identifizieren Sie semantisch identische Gruppen:
   - "Klimaschutz", "Klimaanpassung", "Klimaschutz und Klimaanpassung" → EINE Gruppe
   - "Siedlungsentwicklung", "Quartiersentwicklung" → EINE Gruppe
   - "Freizeit- und Erholungsachse", "Freizeit- und Kulturachse" → EINE Gruppe

   SCHRITT 2: Erstellen Sie kanonische Namen:
   - Wählen Sie den umfassendsten Namen: "Klimaschutz und Klimaanpassung"
   - Oder kombinieren Sie: "Siedlungs- und Quartiersentwicklung"

   SCHRITT 3: Verbindungen neu verknüpfen:
   - ALLE Projekte/Measures/Indikatoren der alten Fragmente
   - Verknüpfen Sie mit dem NEUEN kanonischen Knoten
   - Löschen Sie die alten fragmentierten Knoten

   Beispiele NICHT konsolidieren:
   ❌ "Klimaschutz" und "Energie" → Bleiben getrennt
   ❌ "Mobilität" und "ÖPNV" → Bleiben getrennt
   ❌ "Wohnen" und "Stadtentwicklung" → Bleiben getrennt

2. QUELLENVALIDIERUNG:
   - Entfernen Sie: "Gestaltung: Ibañez Design, Regensburg"
   - Entfernen Sie: "ABSCHNITT: Handlungsfelder der Stadtentwicklung"
   - Entfernen Sie: Einzelwörter wie "Mobilitätsstruktur."
   - Behalten Sie: Vollständige Sätze mit relevantem Inhalt

3. VERBINDUNGEN mit KONFIDENZ:
   - 0.9-1.0: Explizit im Text genannt oder gruppiert
   - 0.7-0.8: Starke thematische Verbindung
   - 0.5-0.6: Schwache/inferierte Verbindung
   - <0.5: Unsicher, vermeiden

4. ID-GENERIERUNG:
   - Action Fields: af_1, af_2, af_3...
   - Projects: proj_1, proj_2, proj_3...
   - Measures: msr_1, msr_2, msr_3...
   - Indicators: ind_1, ind_2, ind_3...

BEISPIEL Verbindung:
"connections": [
  {{
    "target_id": "proj_1",
    "confidence_score": 0.95
  }}
]

Erstellen Sie die vier separaten Listen mit allen Verbindungen und Konfidenz-Scores."""
    else:
        # Simpler prompt for local models
        prompt = f"""Transformieren Sie diese verschachtelte Struktur in das neue relationale 4-Bucket-Format:

{structures_json}

TRANSFORMATION REGELN:

1. INTELLIGENTE KONSOLIDIERUNG (3-Schritt-Prozess):
   SCHRITT 1: Identifizieren Sie semantisch identische Gruppen:
   - "Klimaschutz", "Klimaanpassung", "Klimaschutz und Klimaanpassung" → EINE Gruppe
   - "Siedlungsentwicklung", "Quartiersentwicklung" → EINE Gruppe
   - "Freizeit- und Erholungsachse", "Freizeit- und Kulturachse" → EINE Gruppe

   SCHRITT 2: Erstellen Sie kanonische Namen:
   - Wählen Sie den umfassendsten Namen: "Klimaschutz und Klimaanpassung"
   - Oder kombinieren Sie: "Siedlungs- und Quartiersentwicklung"

   SCHRITT 3: Verbindungen neu verknüpfen:
   - ALLE Projekte/Measures/Indikatoren der alten Fragmente
   - Verknüpfen Sie mit dem NEUEN kanonischen Knoten
   - Löschen Sie die alten fragmentierten Knoten

   Beispiele NICHT konsolidieren:
   ❌ "Klimaschutz" und "Energie" → Bleiben getrennt
   ❌ "Mobilität" und "ÖPNV" → Bleiben getrennt
   ❌ "Wohnen" und "Stadtentwicklung" → Bleiben getrennt

2. QUELLENVALIDIERUNG:
   - Entfernen Sie: "Gestaltung: Ibañez Design, Regensburg"
   - Entfernen Sie: "ABSCHNITT: Handlungsfelder der Stadtentwicklung"
   - Entfernen Sie: Einzelwörter wie "Mobilitätsstruktur."
   - Behalten Sie: Vollständige Sätze mit relevantem Inhalt

3. VERBINDUNGEN mit KONFIDENZ:
   - 0.9-1.0: Explizit im Text genannt oder gruppiert
   - 0.7-0.8: Starke thematische Verbindung
   - 0.5-0.6: Schwache/inferierte Verbindung
   - <0.5: Unsicher, vermeiden

4. ID-GENERIERUNG:
   - Action Fields: af_1, af_2, af_3...
   - Projects: proj_1, proj_2, proj_3...
   - Measures: msr_1, msr_2, msr_3...
   - Indicators: ind_1, ind_2, ind_3...

BEISPIEL Verbindung:
"connections": [
  {{
    "target_id": "proj_1",
    "confidence_score": 0.95
  }}
]

Erstellen Sie die vier separaten Listen mit allen Verbindungen und Konfidenz-Scores."""

    try:
        # Calculate context size for dynamic token allocation
        data_size = len(structures_json)
        estimated_input_tokens = int(data_size / 3.5)  # Conservative estimate

        # For qwen3:14b-AWQ with 32K context, leave room for prompt and output
        context_safety_limit = 20000  # Conservative limit

        if estimated_input_tokens > context_safety_limit:
            print(
                f"⚠️ Data too large ({estimated_input_tokens} tokens) - using chunked processing"
            )
            return transform_large_dataset_chunked(
                structures, log_file_path, log_context
            )

        print(f"📊 Transformation: ~{estimated_input_tokens} input tokens")

        # Get LLM provider and call for transformation
        llm_provider = get_llm_provider()
        print(f"🔧 Using {llm_provider.__class__.__name__} for transformation")

        result = llm_provider.query_structured(
            prompt=prompt,
            response_model=EnrichedReviewJSON,
            system_message=system_message,
            log_file_path=log_file_path,
            log_context=log_context,
        )

        if result:
            # Validate the result
            total_entities = (
                len(result.action_fields)
                + len(result.projects)
                + len(result.measures)
                + len(result.indicators)
            )
            print(f"✅ Enhanced structure created: {total_entities} total entities")
            print(f"   - Action Fields: {len(result.action_fields)}")
            print(f"   - Projects: {len(result.projects)}")
            print(f"   - Measures: {len(result.measures)}")
            print(f"   - Indicators: {len(result.indicators)}")

            return result
        else:
            print("❌ LLM transformation failed - returned None")
            return None

    except Exception as e:
        print(f"❌ Transformation error: {type(e).__name__}: {e}")
        return None


def transform_large_dataset_chunked(
    structures: list,
    log_file_path: str | None = None,
    log_context: str | None = None,
):
    """
    Transform large extracted structures in chunks using cumulative context.

    This implementation processes chunks progressively, maintaining global context
    of the evolving enhanced structure to enable proper entity relationships
    across chunk boundaries.

    Args:
        structures: List of extraction structures to transform
        log_file_path: Optional path for detailed logging
        log_context: Context info for logging

    Returns:
        EnrichedReviewJSON: The enhanced structure, or None if transformation fails
    """
    from ..core.llm_providers import get_llm_provider
    from ..core.schemas import EnrichedReviewJSON

    # Initialize progressive enhanced structure
    enhanced_structure = EnrichedReviewJSON(
        action_fields=[], projects=[], measures=[], indicators=[]
    )

    # Global ID counters for consistent numbering
    global_counters = {
        "action_fields": 0,
        "projects": 0,
        "measures": 0,
        "indicators": 0,
    }

    success_count = 0
    timing_summary = {}
    chunk_timings = []

    # Process each chunk with cumulative context
    for chunk_idx, chunk_data in enumerate(structures, 1):
        chunk_start_time = time.time()
        print(f"  📦 Processing chunk {chunk_idx}/{len(structures)}")

        # Build context from existing entities
        entity_context = build_entity_context_summary(enhanced_structure)
        available_targets = get_available_connection_targets(enhanced_structure)
        next_ids = get_next_available_ids(global_counters)

        # Create context-aware prompt
        context_aware_prompt = f"""
You are transforming extracted data into an enhanced relational structure. You have access to the EXISTING enhanced structure and must build upon it.

EXISTING ENTITIES IN THE ENHANCED STRUCTURE:
{entity_context}

AVAILABLE CONNECTION TARGETS:
{available_targets}

NEXT AVAILABLE IDS:
{next_ids}

DATA TO TRANSFORM (Chunk {chunk_idx}/{len(structures)}):
{json.dumps(chunk_data, ensure_ascii=False, indent=2)}

INSTRUCTIONS:
1. Transform the new data into the 4-bucket structure (action_fields, projects, measures, indicators)
2. Use the NEXT AVAILABLE IDS for new entities
3. Create connections to EXISTING entities where relationships exist
4. Ensure all IDs are unique and don't conflict with existing entities
5. When connecting to existing entities, use their exact IDs from the AVAILABLE CONNECTION TARGETS
6. Assign confidence scores based on relationship clarity
7. Only create NEW entities - do not duplicate existing ones

Respond with the enhanced structure containing ONLY the new entities from this chunk.
"""

        # Calculate token lengths for logging
        input_token_length = int(len(context_aware_prompt) / 3.5)
        print(f"    📊 Input: ~{input_token_length} tokens")

        # Query LLM with cumulative context (with retry)
        result = None
        llm_provider = get_llm_provider()
        for attempt in range(2):
            result = llm_provider.query_structured(
                prompt=context_aware_prompt,
                response_model=EnrichedReviewJSON,
                system_message=None,  # System message already included in prompt
                log_file_path=log_file_path,
                log_context=(
                    f"{log_context}_chunk_{chunk_idx}"
                    if log_context
                    else f"chunk_{chunk_idx}"
                ),
            )
            if result:
                break
            print(f"    ⚠️ Attempt {attempt+1} failed, retrying...")

        chunk_processing_time = time.time() - chunk_start_time
        chunk_timings.append(chunk_processing_time)

        if result:
            # Calculate output token length
            result_json = json.dumps(result.model_dump(), ensure_ascii=False)
            output_token_length = int(len(result_json) / 3.5)
            print(f"    📊 Output: ~{output_token_length} tokens")

            # Merge new entities into the enhanced structure
            merge_result = merge_chunk_result(
                enhanced_structure, result, global_counters
            )
            if merge_result:
                success_count += 1
                print(
                    f"    ✅ Chunk {chunk_idx} processed in {chunk_processing_time:.2f}s"
                )
            else:
                print(
                    f"    ⚠️ Chunk {chunk_idx} merge failed in {chunk_processing_time:.2f}s"
                )
        else:
            print(
                f"    ❌ Chunk {chunk_idx} transformation failed in {chunk_processing_time:.2f}s"
            )

    # Calculate timing summary
    if chunk_timings:
        timing_summary = {
            "total_chunks": len(structures),
            "successful_chunks": success_count,
            "avg_chunk_time": sum(chunk_timings) / len(chunk_timings),
            "total_processing_time": sum(chunk_timings),
            "success_rate": success_count / len(structures) if structures else 0,
        }

        print("\n📊 Processing Summary:")
        print(
            f"    Chunks: {success_count}/{len(structures)} ({timing_summary['success_rate']:.1%})"
        )
        print(f"    Avg time: {timing_summary['avg_chunk_time']:.2f}s/chunk")
        print(f"    Total time: {timing_summary['total_processing_time']:.2f}s")
        print(
            f"    Final entities: {len(enhanced_structure.action_fields)} AF, "
            f"{len(enhanced_structure.projects)} P, {len(enhanced_structure.measures)} M, "
            f"{len(enhanced_structure.indicators)} I"
        )

    return enhanced_structure


def build_entity_context_summary(enhanced_structure) -> str:
    """
    Build a concise summary of existing entities for context-aware prompting.

    Args:
        enhanced_structure: Current EnrichedReviewJSON structure

    Returns:
        str: Formatted summary of existing entities
    """
    summary_parts = []

    # Action Fields
    if enhanced_structure.action_fields:
        af_list = [
            f"  {af.id}: {af.content.get('name', 'Unnamed')}"
            for af in enhanced_structure.action_fields
        ]
        summary_parts.append(
            f"Action Fields ({len(enhanced_structure.action_fields)}):"
        )
        summary_parts.extend(af_list)

    # Projects
    if enhanced_structure.projects:
        proj_list = [
            f"  {p.id}: {p.content.get('title', 'Untitled')}"
            for p in enhanced_structure.projects
        ]
        summary_parts.append(f"\nProjects ({len(enhanced_structure.projects)}):")
        summary_parts.extend(proj_list)

    # Measures
    if enhanced_structure.measures:
        msr_list = [
            f"  {m.id}: {m.content.get('title', 'Unnamed')}"
            for m in enhanced_structure.measures
        ]
        summary_parts.append(f"\nMeasures ({len(enhanced_structure.measures)}):")
        summary_parts.extend(msr_list)

    # Indicators
    if enhanced_structure.indicators:
        ind_list = [
            f"  {i.id}: {i.content.get('name', 'Unnamed')}"
            for i in enhanced_structure.indicators
        ]
        summary_parts.append(f"\nIndicators ({len(enhanced_structure.indicators)}):")
        summary_parts.extend(ind_list)

    if not summary_parts:
        return "No existing entities."

    return "\n".join(summary_parts)


def get_available_connection_targets(enhanced_structure) -> str:
    """
    Get formatted list of available connection targets with their IDs.

    Args:
        enhanced_structure: Current EnrichedReviewJSON structure

    Returns:
        str: Formatted list of connection targets
    """
    targets = []

    # Add all entity IDs as potential connection targets
    for af in enhanced_structure.action_fields:
        targets.append(f"af: {af.id} ({af.content.get('name', 'Unnamed')})")

    for p in enhanced_structure.projects:
        targets.append(f"project: {p.id} ({p.content.get('title', 'Untitled')})")

    for m in enhanced_structure.measures:
        targets.append(f"measure: {m.id} ({m.content.get('title', 'Unnamed')})")

    for i in enhanced_structure.indicators:
        targets.append(f"indicator: {i.id} ({i.content.get('name', 'Unnamed')})")

    if not targets:
        return "No existing connection targets."

    return "\n".join(targets)


def get_next_available_ids(global_counters: dict) -> str:
    """
    Get the next available IDs for each entity type.

    Args:
        global_counters: Dictionary tracking global ID counters

    Returns:
        str: Formatted list of next available IDs
    """
    return f"""
Next Action Field ID: af_{global_counters['action_fields'] + 1}
Next Project ID: proj_{global_counters['projects'] + 1}
Next Measure ID: msr_{global_counters['measures'] + 1}
Next Indicator ID: ind_{global_counters['indicators'] + 1}"""


def merge_chunk_result(enhanced_structure, chunk_result, global_counters: dict) -> bool:
    """
    Merge chunk processing result into the enhanced structure.

    Args:
        enhanced_structure: Main enhanced structure to update
        chunk_result: Result from processing a single chunk
        global_counters: Global ID counters to update

    Returns:
        bool: True if merge was successful, False otherwise
    """
    try:
        # Add new action fields
        for af in chunk_result.action_fields:
            enhanced_structure.action_fields.append(af)
            # Update counter based on ID
            if af.id.startswith("af_"):
                try:
                    id_num = int(af.id.split("_")[1])
                    global_counters["action_fields"] = max(
                        global_counters["action_fields"], id_num
                    )
                except (ValueError, IndexError):
                    pass

        # Add new projects
        for proj in chunk_result.projects:
            enhanced_structure.projects.append(proj)
            # Update counter based on ID
            if proj.id.startswith("proj_"):
                try:
                    id_num = int(proj.id.split("_")[1])
                    global_counters["projects"] = max(
                        global_counters["projects"], id_num
                    )
                except (ValueError, IndexError):
                    pass

        # Add new measures
        for msr in chunk_result.measures:
            enhanced_structure.measures.append(msr)
            # Update counter based on ID
            if msr.id.startswith("msr_"):
                try:
                    id_num = int(msr.id.split("_")[1])
                    global_counters["measures"] = max(
                        global_counters["measures"], id_num
                    )
                except (ValueError, IndexError):
                    pass

        # Add new indicators
        for ind in chunk_result.indicators:
            enhanced_structure.indicators.append(ind)
            # Update counter based on ID
            if ind.id.startswith("ind_"):
                try:
                    id_num = int(ind.id.split("_")[1])
                    global_counters["indicators"] = max(
                        global_counters["indicators"], id_num
                    )
                except (ValueError, IndexError):
                    pass

        return True

    except Exception as e:
        print(f"❌ Failed to merge chunk result: {e}")
        return False


def extract_direct_to_enhanced(
    page_aware_text: list[tuple[str, int]],
    source_id: str,
    log_file_path: str | None = None,
) -> dict[str, Any] | None:
    """
    Extract directly from PDF text to enhanced 4-bucket structure in a single pass.

    This function implements the consolidated extraction approach:
    1. Chunks text with smaller windows and minimal overlap
    2. Uses simplified prompts focused on descriptions
    3. Extracts directly to 4-bucket EnrichedReviewJSON structure
    4. Applies conservative entity resolution

    Args:
        page_aware_text: List of (text, page_number) tuples from parser
        source_id: Source identifier for logging
        log_file_path: Optional path to log LLM dialog

    Returns:
        Enhanced JSON structure or None if extraction fails
    """
    import os
    import time
    from pathlib import Path

    from src.core.config import (
        ENHANCED_CHUNK_MAX_CHARS,
        ENHANCED_CHUNK_MIN_CHARS,
        ENHANCED_CHUNK_OVERLAP,
        FAST_EXTRACTION_MAX_CHUNKS,
        LLM_BACKEND,
        PROJECT_ROOT,
    )
    from src.core.llm_providers import get_llm_provider
    from src.core.schemas import EnrichedReviewJSON
    from src.processing.chunker import chunk_for_llm_with_pages

    print(f"🔄 Starting direct enhanced extraction for {source_id}")
    start_time = time.time()

    if not page_aware_text:
        print("⚠️ No page-aware text provided")
        return None

    # Step 1: Create smaller chunks optimized for focused extraction
    print(f"📝 Chunking with enhanced settings: {ENHANCED_CHUNK_MIN_CHARS}-{ENHANCED_CHUNK_MAX_CHARS} chars, {ENHANCED_CHUNK_OVERLAP*100}% overlap")

    chunks_with_pages = chunk_for_llm_with_pages(
        page_aware_text=page_aware_text,
        max_chars=ENHANCED_CHUNK_MAX_CHARS,
        min_chars=ENHANCED_CHUNK_MIN_CHARS,
        doc_title=f"Strategiedokument {source_id}",
        add_overlap=True,  # Use minimal overlap
    )

    if not chunks_with_pages:
        print("⚠️ No chunks created from text")
        return None

    # Apply chunk limit for performance
    if len(chunks_with_pages) > FAST_EXTRACTION_MAX_CHUNKS:
        print(f"⚡ Using first {FAST_EXTRACTION_MAX_CHUNKS} chunks (performance mode)")
        chunks_with_pages = chunks_with_pages[:FAST_EXTRACTION_MAX_CHUNKS]

    print(f"📄 Processing {len(chunks_with_pages)} chunks")

    # Step 2: Extract from each chunk using simplified prompts
    # Initialize Global Entity Registry for consistency across chunks
    from src.processing.global_registry import GlobalEntityRegistry
    global_registry = GlobalEntityRegistry()

    all_results = []
    llm_provider = get_llm_provider()

    for i, (chunk_text, page_numbers) in enumerate(chunks_with_pages):
        print(f"🔍 Processing chunk {i+1}/{len(chunks_with_pages)} (pages {page_numbers})")

        # Get full accumulated extraction state instead of just entity names
        accumulated_json = merge_enhanced_results(all_results) if all_results else None

        # Create context-aware extraction prompt with full JSON structure
        system_message = create_simplified_system_message_with_context()
        main_prompt = create_simplified_extraction_prompt_with_context(chunk_text, source_id, accumulated_json)

        # Enhanced log context
        enhanced_log_context = f"direct_enhanced_{source_id}_chunk_{i+1}"

        try:
            result = llm_provider.query_structured(
                prompt=main_prompt,
                response_model=EnrichedReviewJSON,
                system_message=system_message,
                log_file_path=log_file_path,
                log_context=enhanced_log_context,
            )

            if result:
                # Register action field entities in the global registry
                for action_field in result.action_fields:
                    # Handle both 'name' and 'title' fields (LLM outputs to 'title')
                    original_name = action_field.content.get('name') or action_field.content.get('title', '')
                    if original_name:
                        canonical_name = global_registry.register_entity(original_name)
                        # Write back to BOTH fields for compatibility
                        action_field.content['name'] = canonical_name
                        action_field.content['title'] = canonical_name

                # Add page attribution to all entities
                add_page_attribution_to_enhanced_result(result, page_numbers)
                all_results.append(result)
                print(f"✅ Chunk {i+1}: {len(result.action_fields)} action fields, {len(result.projects)} projects, {len(result.indicators)} indicators")
            else:
                print(f"⚠️ No result from chunk {i+1}")

        except Exception as e:
            print(f"❌ Error processing chunk {i+1}: {e}")
            continue

    if not all_results:
        print("❌ No successful extractions")
        return None

    # Step 3: Merge and deduplicate results
    print(f"🔄 Merging {len(all_results)} chunk results")
    merged_result = merge_enhanced_results(all_results)

    if not merged_result:
        print("❌ Failed to merge results")
        return None

    # Step 4: Apply conservative entity resolution
    print("🧹 Applying entity resolution")
    try:
        final_result = apply_conservative_entity_resolution(merged_result)
        if not final_result:
            print("⚠️ Entity resolution returned None, using merged result")
            final_result = merged_result
    except Exception as e:
        print(f"⚠️ Entity resolution failed: {e}, using merged result")
        final_result = merged_result

    # Step 5: Print Global Entity Registry summary
    global_registry.print_summary()

    # Step 6: Return as dictionary for API response
    extraction_time = time.time() - start_time
    print(f"✅ Direct enhanced extraction completed in {extraction_time:.1f}s")
    print(
        f"📊 Final: {len(final_result.action_fields)} action fields, "
        f"{len(final_result.projects)} projects, {len(final_result.measures)} measures, "
        f"{len(final_result.indicators)} indicators"
    )

    return {
        "extraction_result": final_result.model_dump(),
        "metadata": {
            "extraction_time_seconds": round(extraction_time, 2),
            "chunks_processed": len(chunks_with_pages),
            "method": "direct_enhanced",
            "llm_backend": LLM_BACKEND,
            "chunk_settings": {
                "max_chars": ENHANCED_CHUNK_MAX_CHARS,
                "min_chars": ENHANCED_CHUNK_MIN_CHARS,
                "overlap": ENHANCED_CHUNK_OVERLAP,
            }
        }
    }


def extract_direct_to_enhanced_with_operations(
    page_aware_text: list[tuple[str, int]],
    source_id: str,
    log_file_path: str | None = None,
) -> dict[str, Any] | None:
    """
    Extract directly from PDF text using operations-based approach.
    
    This function implements the operations-based extraction approach:
    1. Chunks text with smaller windows and minimal overlap
    2. For each chunk, LLM returns operations to apply to current state
    3. Operations are applied deterministically to build final state
    4. No copy degradation or context bloat issues
    
    Args:
        page_aware_text: List of (text, page_number) tuples from parser
        source_id: Source identifier for logging
        log_file_path: Optional path to log LLM dialog
        
    Returns:
        Enhanced JSON structure or None if extraction fails
    """
    import os
    import time
    from pathlib import Path

    from src.core.config import (
        ENHANCED_CHUNK_MAX_CHARS,
        ENHANCED_CHUNK_MIN_CHARS,
        ENHANCED_CHUNK_OVERLAP,
        FAST_EXTRACTION_MAX_CHUNKS,
        LLM_BACKEND,
        PROJECT_ROOT,
    )
    from src.core.llm_providers import get_llm_provider
    from src.core.operations_schema import ExtractionOperations
    from src.core.schemas import EnrichedReviewJSON
    from src.extraction.operations_executor import OperationExecutor
    from src.processing.chunker import chunk_for_llm_with_pages

    print(f"🔄 Starting operations-based extraction for {source_id}")
    start_time = time.time()

    if not page_aware_text:
        print("⚠️ No page-aware text provided")
        return None

    # Step 1: Create smaller chunks optimized for focused extraction
    print(f"📝 Chunking with enhanced settings: {ENHANCED_CHUNK_MIN_CHARS}-{ENHANCED_CHUNK_MAX_CHARS} chars, {ENHANCED_CHUNK_OVERLAP*100}% overlap")

    chunks_with_pages = chunk_for_llm_with_pages(
        page_aware_text=page_aware_text,
        max_chars=ENHANCED_CHUNK_MAX_CHARS,
        min_chars=ENHANCED_CHUNK_MIN_CHARS,
        doc_title=f"Strategiedokument {source_id}",
        add_overlap=True,
    )

    if not chunks_with_pages:
        print("⚠️ No chunks created from text")
        return None

    # Apply chunk limit for performance
    if len(chunks_with_pages) > FAST_EXTRACTION_MAX_CHUNKS:
        print(f"⚡ Using first {FAST_EXTRACTION_MAX_CHUNKS} chunks (performance mode)")
        chunks_with_pages = chunks_with_pages[:FAST_EXTRACTION_MAX_CHUNKS]

    print(f"📄 Processing {len(chunks_with_pages)} chunks")

    # Step 2: Initialize empty extraction state and operations executor
    current_state = EnrichedReviewJSON(
        action_fields=[],
        projects=[],
        measures=[],
        indicators=[]
    )

    executor = OperationExecutor()
    llm_provider = get_llm_provider()
    all_operation_logs = []

    # Step 3: Process each chunk with operations
    for i, (chunk_text, page_numbers) in enumerate(chunks_with_pages):
        print(f"🔍 Processing chunk {i+1}/{len(chunks_with_pages)} (pages {page_numbers})")

        # Create operations-focused prompt
        system_message = create_operations_system_message()
        main_prompt = create_operations_extraction_prompt(chunk_text, source_id, current_state, page_numbers)

        # Enhanced log context
        enhanced_log_context = f"operations_{source_id}_chunk_{i+1}"

        try:
            # Get operations from LLM
            operations_result = llm_provider.query_structured(
                prompt=main_prompt,
                response_model=ExtractionOperations,
                system_message=system_message,
                log_file_path=log_file_path,
                log_context=enhanced_log_context,
            )

            if operations_result and operations_result.operations:
                # Filter out invalid operations instead of skipping entire chunk
                from src.extraction.operations_executor import validate_operations
                validation_errors = validate_operations(operations_result.operations, current_state)

                if validation_errors:
                    print(f"⚠️ Chunk {i+1}: {len(validation_errors)} operation validation errors:")
                    for error in validation_errors[:3]:  # Show first 3 errors
                        print(f"   - {error}")
                    if len(validation_errors) > 3:
                        print(f"   - ... and {len(validation_errors) - 3} more errors")

                    # Filter out invalid operations - validate each operation individually
                    valid_operations = []
                    for op in operations_result.operations:
                        single_op_errors = validate_operations([op], current_state)
                        if not single_op_errors:
                            valid_operations.append(op)

                    print(f"   Proceeding with {len(valid_operations)}/{len(operations_result.operations)} valid operations")
                    operations_result.operations = valid_operations

                if operations_result.operations:  # Only proceed if we have valid operations
                    # Apply validated operations to current state
                    try:
                        new_state, operation_log = executor.apply_operations(
                            current_state,
                            operations_result.operations,
                            chunk_index=i
                        )

                        # Only update current_state if operations were successfully applied
                        if operation_log.successful_operations > 0:
                            current_state = new_state
                            all_operation_logs.append(operation_log)

                            print(f"✅ Chunk {i+1}: {operation_log.successful_operations}/{operation_log.total_operations} operations applied")
                            print(f"📊 Current state: {len(current_state.action_fields)} action fields, {len(current_state.projects)} projects, {len(current_state.measures)} measures, {len(current_state.indicators)} indicators")
                        else:
                            print(f"⚠️ Chunk {i+1}: No operations succeeded, keeping previous state")
                            all_operation_logs.append(operation_log)

                    except Exception as op_error:
                        print(f"❌ Chunk {i+1}: Error applying operations: {op_error}")
                        print("   Keeping previous state to preserve integrity")
                        continue
                else:
                    print(f"⚠️ Chunk {i+1}: All operations filtered out, skipping")

            else:
                print(f"⚠️ No operations from chunk {i+1}")

        except Exception as e:
            print(f"❌ Error processing chunk {i+1}: {e}")
            continue

    # Step 4: Final statistics and return
    extraction_time = time.time() - start_time
    operation_summary = executor.get_operation_summary()

    print(f"✅ Operations-based extraction completed in {extraction_time:.1f}s")
    print("📊 Operation Summary:")
    print(f"   - Total operations: {operation_summary['total_operations']}")
    print(f"   - Successful: {operation_summary['successful_operations']}")
    print(f"   - Success rate: {operation_summary['success_rate']:.1%}")
    print(f"   - Entities created: {operation_summary['entities_created']}")
    print(
        f"📊 Final: {len(current_state.action_fields)} action fields, "
        f"{len(current_state.projects)} projects, {len(current_state.measures)} measures, "
        f"{len(current_state.indicators)} indicators"
    )

    # Return in same format as other endpoints for visualization tool compatibility
    return current_state.model_dump()


def create_simplified_system_message() -> str:
    """Create simplified system message focused on descriptions."""
    return """Sie sind ein Experte für die Extraktion von Handlungsfeldern, Projekten und Indikatoren aus deutschen kommunalen Strategiedokumenten.

HIERARCHISCHE STRUKTUR - KRITISCH WICHTIG:
- Handlungsfelder (action_fields): BREITE STRATEGISCHE BEREICHE (max. 8-15 Stück)
  → Projekte: Konkrete Vorhaben innerhalb der Handlungsfelder
    → Maßnahmen: Spezifische Aktionen innerhalb der Projekte
      → Indikatoren: Messbare Kennzahlen

HANDLUNGSFELDER - NUR ÜBERGEORDNETE BEREICHE:
✅ RICHTIG: "Mobilität und Verkehr", "Klimaschutz und Klimaanpassung", "Energie", "Siedlungsentwicklung", "Wirtschaft und Wissenschaft"
❌ FALSCH: "Stadtbahn", "Neuer ZOB", "Green Deal", "Solaranlagen", "Radwegenetz" (das sind PROJEKTE!)

TYPISCHE HANDLUNGSFELDER KOMMUNALER STRATEGIEN:
- Mobilität und Verkehr
- Klimaschutz und Klimaanpassung  
- Energie
- Siedlungs- und Quartiersentwicklung
- Freiraumentwicklung
- Wirtschaft und Wissenschaft
- Digitalisierung
- Soziales und Integration
- Bildung und Betreuung
- Kultur und Sport

PFLICHTFELDER:
- Handlungsfelder: name + description (1-3 Sätze über den strategischen Bereich)
- Projekte: title + description (1-2 Sätze) + type
- Maßnahmen: title + description (1-2 Sätze)
- Indikatoren: title + description (Was wird gemessen?)

OPTIONALE FELDER (NUR mit Quellenbeleg):
- unit: Nur wenn explizit genannt (z.B. "Tonnen CO2/Jahr")
- calculation: Nur wenn Berechnungsmethode beschrieben
- valuesSource: Nur wenn Datenquelle erwähnt

WICHTIGE REGELN:
1. HIERARCHIE BEACHTEN: Spezifische Projekte NIEMALS als Handlungsfelder extrahieren
2. Konservative Extraktion: Nur was eindeutig im Text steht
3. Keine Erfindung von Metadaten (Budget, Termine, Abteilungen)
4. Qualitätsbeschreibungen: Jede Entität braucht aussagekräftige deutsche Beschreibung
5. Verbindungen mit Konfidenz-Scores (0.5-1.0) basierend auf Textkontext

Antworten Sie AUSSCHLIESSLICH mit einem JSON-Objekt, das der EnrichedReviewJSON-Struktur entspricht."""


def create_simplified_extraction_prompt(chunk_text: str, source_id: str) -> str:
    """Create simplified extraction prompt for direct 4-bucket extraction."""
    return f"""Extrahieren Sie aus diesem Textabschnitt die HIERARCHISCH KORREKTEN Strukturen.

KRITISCH: Handlungsfelder sind NUR breite strategische Bereiche (max. 8-15 total), NICHT spezifische Projekte!

BEISPIELE DER HIERARCHIE:
• Handlungsfeld: "Mobilität und Verkehr" 
  → Projekt: "Stadtbahn Regensburg"
  → Projekt: "Radwegenetz Ausbau"
  → Projekt: "Neuer Zentraler Omnibusbahnhof"
  
• Handlungsfeld: "Klimaschutz und Klimaanpassung"
  → Projekt: "Green Deal Regensburg"
  → Projekt: "Klimaneutrale Verwaltung"
  
• Handlungsfeld: "Energie"
  → Projekt: "Solaroffensive"
  → Projekt: "Windkraftausbau"

FALSCHE EXTRAKTION VERMEIDEN:
❌ "Stadtbahn" als Handlungsfeld → ✅ "Mobilität und Verkehr" als Handlungsfeld, "Stadtbahn" als Projekt
❌ "Green Deal" als Handlungsfeld → ✅ "Klimaschutz" als Handlungsfeld, "Green Deal" als Projekt

TEXT:
{chunk_text}

Erstellen Sie die 4-Bucket-Struktur mit KORREKTER HIERARCHIE:

1. action_fields: [{{
   "id": "af_1",
   "content": {{
     "name": "Breiter strategischer Bereich (z.B. 'Mobilität und Verkehr')",
     "description": "1-3 Sätze über diesen ÜBERGEORDNETEN strategischen Bereich"
   }},
   "connections": []
}}]

2. projects: [{{
   "id": "proj_1",
   "content": {{
     "title": "Projekttitel",
     "description": "1-2 Sätze Projektbeschreibung",
     "type": "Infrastructure/Policy/Program/Study"
   }},
   "connections": [{{"target_id": "af_1", "confidence_score": 0.9}}]
}}]

3. measures: [{{
   "id": "msr_1",
   "content": {{
     "title": "Maßnahmentitel",
     "description": "1-2 Sätze was diese Maßnahme beinhaltet"
   }},
   "connections": [{{"target_id": "proj_1", "confidence_score": 0.8}}]
}}]

4. indicators: [{{
   "id": "ind_1",
   "content": {{
     "title": "Indikatorname",
     "description": "Was wird hier gemessen/überwacht",
     "unit": "Nur wenn explizit genannt",
     "calculation": "Nur wenn beschrieben"
   }},
   "connections": [{{"target_id": "msr_1", "confidence_score": 0.9}}]
}}]

KONFIDENZ-SCORES:
- 0.9-1.0: Explizit verbunden oder gruppiert
- 0.7-0.8: Starke thematische Verbindung
- 0.5-0.6: Schwache/inferierte Verbindung

ID-FORMAT: af_1, proj_1, msr_1, ind_1 (fortlaufend nummeriert)"""


def create_simplified_system_message_with_context() -> str:
    """Create context-aware system message that emphasizes consistency with existing entities."""
    return """Sie sind ein Experte für die Extraktion von Handlungsfeldern, Projekten und Indikatoren aus deutschen kommunalen Strategiedokumenten.

HIERARCHISCHE STRUKTUR - KRITISCH WICHTIG:
- Handlungsfelder (action_fields): BREITE STRATEGISCHE BEREICHE (max. 8-15 Stück)
  → Projekte: Konkrete Vorhaben innerhalb der Handlungsfelder
    → Maßnahmen: Spezifische Aktionen innerhalb der Projekte
      → Indikatoren: Messbare Kennzahlen

KONSISTENZ-REGEL (KRITISCH WICHTIG):
- Falls Sie Konzepte finden, die zu bereits bekannten Handlungsfeldern gehören, verwenden Sie die EXAKTEN Namen der bekannten Handlungsfelder
- Erstellen Sie NUR neue Handlungsfelder, wenn sie wirklich einzigartig sind
- Bei ähnlichen Begriffen verwenden Sie die bereits bekannte Variante

Antworten Sie AUSSCHLIESSLICH mit einem JSON-Objekt, das dem vorgegebenen Schema entspricht. KEIN zusätzlicher Text, KEINE Erklärungen, NUR JSON."""


def create_simplified_extraction_prompt_with_context(chunk_text: str, source_id: str, accumulated_json) -> str:
    """Create context-aware extraction prompt that includes full accumulated extraction state."""

    if accumulated_json:
        # Format the full JSON structure for the LLM
        import json
        accumulated_json_str = json.dumps(accumulated_json.model_dump() if hasattr(accumulated_json, 'model_dump') else accumulated_json,
                                        indent=2, ensure_ascii=False)
        context_text = f"""AKTUELLER EXTRAKTIONSSTAND (bisher gefundene Strukturen):
{accumulated_json_str}

WICHTIG: Erweitern Sie diese bestehende Struktur. Verwenden Sie exakte IDs und Namen aus dem obigen JSON. Erstellen Sie Verbindungen zu bestehenden Entities."""
    else:
        context_text = "Noch keine Strukturen extrahiert - dies ist der erste Chunk."

    return f"""Extrahieren Sie aus diesem Textabschnitt die HIERARCHISCH KORREKTEN Strukturen.

{context_text}

KONSISTENZ-ANWEISUNGEN:
- Falls Sie Konzepte finden, die zu den bereits bekannten Handlungsfeldern gehören, verwenden Sie die EXAKTEN Namen der bekannten Handlungsfelder
- Erstellen Sie NUR neue Handlungsfelder, wenn sie wirklich einzigartig sind und nicht den bekannten zugeordnet werden können
- Bei ähnlichen Begriffen (z.B. "Wirtschaft & Wissenschaft" vs "Wirtschaft und Wissenschaft") verwenden Sie die bereits bekannte Variante

KRITISCH: Handlungsfelder sind NUR breite strategische Bereiche (max. 8-15 total), NICHT spezifische Projekte!

TEXT:
{chunk_text}

Erstellen Sie die 4-Bucket-Struktur mit KORREKTER HIERARCHIE und unter Verwendung der bereits bekannten Handlungsfelder, wo zutreffend."""


def create_operations_system_message() -> str:
    """Create system message for operations-based extraction."""
    return """Sie sind ein Experte für die Extraktion von Handlungsfeldern, Projekten, Maßnahmen und Indikatoren aus deutschen kommunalen Strategiedokumenten.

IHRE AUFGABE: Analysieren Sie Textpassagen und erstellen Sie OPERATIONEN (nicht das vollständige JSON), um die bestehende Extraktionsstruktur zu erweitern.

VERFÜGBARE OPERATIONEN:
- CREATE: Neue Entity erstellen (nur wenn wirklich neu und einzigartig) - NIEMALS entity_id angeben, wird automatisch generiert!
 - UPDATE: Bestehende Entity mit zusätzlichen Details erweitern (intelligentes Mergen) - entity_id erforderlich
 - CONNECT: Verbindungen zwischen Entities erstellen - connections erforderlich

HIERARCHISCHE STRUKTUR:
- Handlungsfelder (action_field): BREITE STRATEGISCHE BEREICHE (max. 8-15 total)
  → Projekte: Konkrete Vorhaben innerhalb der Handlungsfelder
    → Maßnahmen: Spezifische Aktionen innerhalb der Projekte
      → Indikatoren: Messbare Kennzahlen für Projekte/Maßnahmen

WICHTIGE PRINZIPIEN:
- Bevorzugen Sie UPDATE gegenüber CREATE für ähnliche/gleichartige Konzepte
- Verwenden Sie exakte Entity-IDs für Verbindungen
- Fügen Sie immer Quellenangaben (source_pages, source_quote) hinzu
- Seien Sie konservativ bei neuen Handlungsfeldern

Antworten Sie AUSSCHLIESSLICH mit der Operations-Liste im JSON-Format."""


def create_operations_extraction_prompt(
    chunk_text: str,
    source_id: str,
    current_state: dict,
    page_numbers: list[int]
) -> str:
    """Create operations-focused extraction prompt."""

    import json

    # Format current state for display
    if (hasattr(current_state, 'action_fields') and
        (current_state.action_fields or current_state.projects or
         current_state.measures or current_state.indicators)):

        current_json_str = json.dumps(
            current_state.model_dump() if hasattr(current_state, 'model_dump') else current_state,
            indent=2,
            ensure_ascii=False
        )
        context_text = f"""AKTUELLER EXTRAKTIONSSTAND:
{current_json_str}

ANWEISUNG: Analysieren Sie den Text und erstellen Sie Operationen zur Erweiterung dieser Struktur."""
    else:
        context_text = "ERSTER CHUNK: Noch keine Entities extrahiert. Beginnen Sie mit CREATE-Operationen."

    page_list = ", ".join(map(str, sorted(page_numbers)))

    return f"""Analysieren Sie diesen Textabschnitt und erstellen Sie OPERATIONEN zur Strukturerweiterung.

{context_text}

TEXTABSCHNITT (Seiten {page_list}):
{chunk_text}

OPERATIONEN-BEISPIELE:

CREATE neue Entity:
{{
  "operation": "CREATE",
  "entity_type": "action_field",
  "content": {{"title": "Mobilität", "description": "Nachhaltige Verkehrslösungen"}},
  "source_pages": [{page_list}],
  "source_quote": "Kurzer relevanter Textauszug (max. 50 Wörter)",
  "confidence": 0.9
}}

UPDATE bestehende Entity (intelligent merging):
{{
  "operation": "UPDATE", 
  "entity_type": "project",
  "entity_id": "proj_1",
  "content": {{"description": "Neue oder erweiterte Beschreibung", "status": "active"}},
  "source_pages": [{page_list}],
  "source_quote": "Zusätzlicher Textauszug (max. 50 Wörter)",
  "confidence": 0.8
}}

CONNECT Entities:
{{
  "operation": "CONNECT",
  "entity_type": "project", 
  "connections": [
    {{"from_id": "proj_2", "to_id": "af_1", "confidence": 0.8}},
    {{"from_id": "proj_2", "to_id": "msr_1", "confidence": 0.9}}
  ],
  "confidence": 0.8
}}

WICHTIG: 
- Verwenden Sie exakte Entity-IDs aus dem aktuellen Stand
- Fügen Sie IMMER source_pages und source_quote hinzu (außer bei CONNECT)
- Seien Sie sparsam mit CREATE für Handlungsfelder
- Verwenden Sie UPDATE für alle Änderungen an bestehenden Entities
- CONNECT kann mehrere Verbindungen gleichzeitig erstellen

Antworten Sie NUR mit der Operations-Liste im JSON-Format:
{{"operations": [...]}}"""


def add_page_attribution_to_enhanced_result(
    result,  # EnrichedReviewJSON
    page_numbers: list[int]
) -> None:
    """Add page attribution to all entities in enhanced result."""
    page_str = f"Seiten {', '.join(map(str, sorted(page_numbers)))}"

    # Add page attribution to all entity types
    for entity_list in [result.action_fields, result.projects, result.measures, result.indicators]:
        for entity in entity_list:
            if "page_source" not in entity.content:
                entity.content["page_source"] = page_str


def merge_enhanced_results(results: list) -> dict | None:  # EnrichedReviewJSON type
    """Merge multiple enhanced results with simple concatenation."""
    if not results:
        return None

    if len(results) == 1:
        return results[0]

    # Import here to avoid circular imports
    from src.core.schemas import EnrichedReviewJSON

    # Simple merge - concatenate all lists
    merged = EnrichedReviewJSON(
        action_fields=[],
        projects=[],
        measures=[],
        indicators=[]
    )

    entity_counter = {"af": 0, "proj": 0, "msr": 0, "ind": 0}

    for result in results:
        # Reassign IDs to avoid conflicts
        for af in result.action_fields:
            entity_counter["af"] += 1
            af.id = f"af_{entity_counter['af']}"
            merged.action_fields.append(af)

        for proj in result.projects:
            entity_counter["proj"] += 1
            proj.id = f"proj_{entity_counter['proj']}"
            merged.projects.append(proj)

        for msr in result.measures:
            entity_counter["msr"] += 1
            msr.id = f"msr_{entity_counter['msr']}"
            merged.measures.append(msr)

        for ind in result.indicators:
            entity_counter["ind"] += 1
            ind.id = f"ind_{entity_counter['ind']}"
            merged.indicators.append(ind)

    return merged


def apply_conservative_entity_resolution(result):  # EnrichedReviewJSON type
    """Apply conservative entity resolution - only merge exact matches."""
    try:
        # Convert to format expected by existing entity resolution
        structures = []

        for af in result.action_fields:
            af_dict = {
                "action_field": af.content.get("title", af.content.get("name", "")),
                "projects": []
            }

            # Find connected projects (using parent→child connections)
            for proj in result.projects:
                # Check if this project is connected from the action field
                if proj.id in [conn.target_id for conn in af.connections]:
                        proj_dict = {
                            "title": proj.content.get("title", ""),
                            "measures": [],
                            "indicators": []
                        }

                        # Find connected measures and indicators (using parent→child connections)
                        for msr in result.measures:
                            # Check if this measure is connected from the project
                            if msr.id in [conn.target_id for conn in proj.connections]:
                                    proj_dict["measures"].append({
                                        "title": msr.content.get("title", ""),
                                        "description": msr.content.get("description", "")
                                    })

                        for ind in result.indicators:
                            # Check if this indicator is connected from the project
                            if ind.id in [conn.target_id for conn in proj.connections]:
                                    proj_dict["indicators"].append({
                                        "title": ind.content.get("title", ""),
                                        "description": ind.content.get("description", "")
                                    })

                        af_dict["projects"].append(proj_dict)

            structures.append(af_dict)

        # Check if structures have meaningful content
        total_projects = sum(len(s.get("projects", [])) for s in structures)
        if total_projects == 0:
            print("⚠️ No connected projects found, bypassing entity resolution to preserve entities")
            return result

        print(f"🔗 Found {total_projects} connected projects in {len(structures)} structures, proceeding with resolution")

        # Apply existing entity resolution
        resolved_structures = apply_entity_resolution(structures)

        # Convert back to enhanced format using the working rebuild function
        return rebuild_enhanced_structure_from_resolved(resolved_structures, result)

    except Exception as e:
        print(f"⚠️ Entity resolution failed, returning unresolved: {e}")
        return result


def rebuild_enhanced_structure_from_resolved_v2(
    resolved_structures: list[dict],
    original_result  # EnrichedReviewJSON type
):
    """Rebuild enhanced structure from resolved data."""
    # For now, return original result since entity resolution is complex
    # TODO: Implement proper rebuilding logic
    return original_result
