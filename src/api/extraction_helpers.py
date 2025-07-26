"""
Helper functions for extraction endpoints.

This module contains refactored helper functions to break down
the large extraction functions in routes.py.
"""

import json
import re
from functools import lru_cache
from typing import Any

from src.core.config import (
    AGGREGATION_CHUNK_SIZE,
    MIN_QUOTE_LENGTH,
    QUOTE_MATCH_THRESHOLD,
)
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
    chunk_size: int = AGGREGATION_CHUNK_SIZE,
    recursion_depth: int = 0,
    max_recursion: int = 2,
    log_file_path: str | None = None,
    log_context_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """
    Handle large datasets by recursively processing them in smaller chunks.
    """
    print(
        f"🔄 Processing {len(all_chunk_results)} action fields in chunks of {chunk_size} (recursion depth: {recursion_depth})"
    )

    if recursion_depth >= max_recursion:
        print(
            f"⚠️ Maximum recursion depth ({max_recursion}) reached - falling back to simple deduplication"
        )
        return simple_deduplication_fallback(all_chunk_results)

    intermediate_results = []
    for i in range(0, len(all_chunk_results), chunk_size):
        chunk = all_chunk_results[i : i + chunk_size]
        print(
            f"📋 Processing chunk {i//chunk_size + 1} with {len(chunk)} action fields"
        )

        chunk_data = json.dumps(chunk, indent=2, ensure_ascii=False)
        log_context = (
            f"{log_context_prefix} - Aggregation Pass {recursion_depth + 1}, Chunk {i//chunk_size + 1} ({len(chunk)} fields)"
            if log_context_prefix
            else None
        )
        result = perform_single_aggregation(chunk_data, log_file_path, log_context)

        if result:
            intermediate_results.extend(result)
            print(
                f"   ✅ Chunk {i//chunk_size + 1} aggregated to {len(result)} action fields"
            )
        else:
            print(
                f"   ❌ Chunk {i//chunk_size + 1} aggregation failed, retaining original chunk data"
            )
            intermediate_results.extend(chunk)

    print(
        f"✅ Pass {recursion_depth + 1} completed: {len(intermediate_results)} intermediate results"
    )

    if len(intermediate_results) > 12 and len(intermediate_results) < len(
        all_chunk_results
    ):
        return chunked_aggregation(
            intermediate_results,
            chunk_size,
            recursion_depth + 1,
            max_recursion,
            log_file_path,
            log_context_prefix,
        )

    elif len(intermediate_results) > 12:
        print("⚠️ Aggregation stalled, falling back to simple deduplication")
        return simple_deduplication_fallback(intermediate_results)

    return intermediate_results


def perform_single_aggregation(
    chunk_data: str, log_file_path: str | None = None, log_context: str | None = None
) -> list[dict[str, Any]] | None:
    """
    Perform a single aggregation pass on JSON data.
    """
    from src.core import MODEL_TEMPERATURE
    from src.core.llm import query_ollama_structured
    from src.core.schemas import ExtractionResult

    system_message = """Sie sind ein Experte für die Konsolidierung von Handlungsfeldern aus deutschen kommunalen Strategiedokumenten.
Ihre Aufgabe ist es, ähnliche Handlungsfelder intelligent zusammenzuführen und eine reduzierte Liste von
maximal 12 konsolidierten Handlungsfeldern zu erstellen.
Antworten Sie AUSSCHLIESSLICH mit einem JSON-Objekt, das der vorgegebenen Struktur entspricht.
KEIN zusätzlicher Text, KEINE Erklärungen, NUR JSON."""

    prompt = f"""Sie erhalten {chunk_data.count('"action_field"')} Handlungsfelder zur Konsolidierung.

Ihre Aufgabe ist es, diese Liste durch intelligente Zusammenführung ähnlicher Bereiche zu reduzieren.

ZIEL: Erstellen Sie eine Liste von maximal 12 konsolidierten Handlungsfeldern (idealerweise 8-12).

ERFOLGSMETRIK: Konsolidieren Sie nur wirklich ähnliche Bereiche. Behalten Sie die Vielfalt und
Granularität der kommunalen Handlungsfelder bei. Streben Sie 10-15 konsolidierte Handlungsfelder an.

STRATEGIE:
1. Analysieren Sie die bereitgestellten Handlungsfelder nach Themenähnlichkeit.
2. Gruppieren Sie verwandte Bereiche unter aussagekräftige Oberkategorien.
3. Verschmelzen Sie die Inhalte vollständig: Alle Projekte, Maßnahmen und Indikatoren der zusammengeführten Felder müssen erhalten bleiben.
4. Verwenden Sie ausschließlich deutsche Fachterminologie. Englische Begriffe sind komplett zu eliminieren.
5. Behalten Sie die Granularität bei: Verschiedene kommunale Fachbereiche sollten getrennt bleiben.

Beachten Sie folgende Konsolidierungsregeln und Beispiele:
✅ "Klimaschutz" + "Energie" + "Nachhaltigkeit" + "Umwelt" → "Klimaschutz, Energie und Umwelt"
✅ "Mobilität" + "Verkehr" + "ÖPNV" + "Radverkehr" → "Mobilität und Verkehr"
✅ "Wohnen" + "Quartiere" + "Stadtentwicklung" + "Bauplanung" → "Wohnen und Quartiersentwicklung"
✅ "Wirtschaft" + "Innovation" + "Wissenschaft" + "Digitalisierung" → "Wirtschaft, Innovation und Digitalisierung"
✅ "Kultur" + "Bildung" + "Sport" + "Freizeit" → "Kultur, Bildung und Sport"
✅ "Soziales" + "Integration" + "Teilhabe" + "Gesundheit" → "Soziales, Integration und Gesundheit"
✅ "Verwaltung" + "Bürgerbeteiligung" + "Transparenz" → "Verwaltung und Bürgerbeteiligung"
✅ "Sicherheit" + "Ordnung" + "Katastrophenschutz" → "Sicherheit und Ordnung"

❌ "Jugendarbeit" und "Seniorenbetreuung" → Bleiben getrennt (verschiedene Zielgruppen)
❌ "Digitalisierung der Verwaltung" und "Digitale Bildung" → Bleiben getrennt (verschiedene Bereiche)
❌ "Stadtplanung" und "Denkmalschutz" → Bleiben getrennt (unterschiedliche Fachbereiche)

Hier sind die Handlungsfelder zur Konsolidierung:
{chunk_data}

Antworten Sie AUSSCHLIESSLICH mit einem JSON-Objekt, das die konsolidierten Handlungsfelder enthält. KEIN zusätzlicher Text, KEINE Erklärungen, NUR JSON.
"""

    try:
        # Check input size
        data_size = len(chunk_data)
        print(f"   🔍 Attempting aggregation with {data_size} characters of JSON")

        result = query_ollama_structured(
            prompt=prompt,
            response_model=ExtractionResult,
            system_message=system_message,
            temperature=MODEL_TEMPERATURE,
            log_file_path=log_file_path,
            log_context=log_context,
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

            print(
                f"   ✅ LLM aggregation successful: {len(aggregated_data)} action fields"
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
    Aggregate extraction results from multiple chunks using proven chunked approach.

    With optimized chunking, we now start with fewer action fields (~16 instead of 50+),
    making the chunked aggregation approach fast and reliable.
    """
    if not all_chunk_results:
        return []

    # With optimized chunking, we typically have 10-20 fields to aggregate
    print(f"🔄 Aggregating {len(all_chunk_results)} extracted action fields...")

    # Use chunked aggregation for all cases - it's proven and reliable
    if len(all_chunk_results) > 12:
        print(
            f"📊 Starting chunked aggregation for {len(all_chunk_results)} action fields"
        )
        return chunked_aggregation(
            all_chunk_results,
            log_file_path=log_file_path,
            log_context_prefix=log_context_prefix,
        )

    # For small datasets, we can still benefit from a single aggregation pass
    print(
        f"📋 Small dataset ({len(all_chunk_results)} fields) - single aggregation pass"
    )
    chunk_data = json.dumps(all_chunk_results, indent=2, ensure_ascii=False)
    result = perform_single_aggregation(
        chunk_data,
        log_file_path,
        (
            f"{log_context_prefix} - Single Pass Aggregation"
            if log_context_prefix
            else "Single Pass Aggregation"
        ),
    )

    if result:
        # Final validation to remove any English contamination
        validated_data = validate_german_only_content(result)
        print(f"✅ Aggregated to {len(validated_data)} clean German action fields")
        return validated_data
    else:
        print("⚠️ Single aggregation failed, using simple deduplication fallback")
        return simple_deduplication_fallback(all_chunk_results)


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
