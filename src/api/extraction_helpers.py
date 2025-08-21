"""
Helper functions for extraction endpoints.

This module contains refactored helper functions to break down
the large extraction functions in routes.py.
"""

import json
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import aiofiles
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from src.core.config import (
    AGGREGATION_CHUNK_SIZE,
)

# ============================================================================
# CONSTANTS AND TEMPLATES
# ============================================================================

# System message templates
DEDUPLICATION_SYSTEM_MESSAGE = """Sie sind ein Experte für die KONSERVATIVE Deduplizierung von Handlungsfeldern aus deutschen kommunalen Strategiedokumenten.
Ihre Hauptaufgabe ist es, DUPLIKATE zu entfernen und die GRANULARITÄT zu erhalten.
Konsolidieren Sie NUR offensichtliche Duplikate oder fast identische Felder.
Behalten Sie die meisten Handlungsfelder bei - Reduzierung um maximal 30-40%.
Antworten Sie AUSSCHLIESSLICH mit einem JSON-Objekt, das der vorgegebenen Struktur entspricht.
KEIN zusätzlicher Text, KEINE Erklärungen, NUR JSON."""

EXTRACTION_SYSTEM_MESSAGE = """Sie sind ein Experte für die Extraktion von Handlungsfeldern,
Projekten und Indikatoren aus deutschen kommunalen Strategiedokumenten.

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

OPERATIONS_SYSTEM_MESSAGE = """Sie sind ein Experte für die Extraktion von Handlungsfeldern, Projekten,
Maßnahmen und Indikatoren aus deutschen kommunalen Strategiedokumenten.

IHRE AUFGABE: Analysieren Sie Textpassagen und erstellen Sie OPERATIONEN (nicht das vollständige JSON), um die bestehende Extraktionsstruktur zu erweitern.

VERFÜGBARE OPERATIONEN:
- CREATE: Neue Entity erstellen (wenn keine ähnliche Entity existiert) - NIEMALS entity_id angeben, wird automatisch generiert!
- UPDATE: Bestehende Entity anreichern (neue Felder hinzufügen, Texte erweitern, Listen ergänzen) - entity_id erforderlich
- CONNECT: Verbindungen zwischen Entities erstellen (auch bestehende Entities können neue Verbindungen erhalten) - connections erforderlich

DUPLIKAT-VERMEIDUNG (HÖCHSTE PRIORITÄT):
- ALLE Entity-Typen im ENTITY REGISTRY prüfen (nicht nur Handlungsfelder!)
- Semantisch ähnliche Entities → UPDATE verwenden
- Verschiedene Schreibweisen des gleichen Konzepts → UPDATE verwenden
- Teilweise Überlappungen → UPDATE verwenden
- NUR CREATE wenn absolut sicher dass Entity neu ist

HIERARCHISCHE STRUKTUR:
- Handlungsfelder (action_field): BREITE STRATEGISCHE BEREICHE
  → Projekte: Konkrete Vorhaben innerhalb der Handlungsfelder
    → Maßnahmen: Spezifische Aktionen innerhalb der Projekte
      → Indikatoren: Messbare Kennzahlen für Projekte/Maßnahmen

KRITISCHE VERBINDUNGSREGELN:
- CONNECT-Operationen zu bestehenden UND neu erstellten Entities sind ERWÜNSCHT!
- NIEMALS Entity-Namen/Titel verwenden - NUR die exakten IDs aus der ID-MAPPING-TABELLE!
- Beispiel: "af_1" ist korrekt, "Mobilität und Verkehr" ist FALSCH
- Beispiel: "proj_2" ist korrekt, "Solarinitiative" ist FALSCH  
- CONNECT-Operationen erfordern Konfidenz ≥ 0.7 und klaren thematischen Zusammenhang

ENTSCHEIDUNGSLOGIK (STRENG BEFOLGEN):
1. Entity bereits im REGISTRY? → UPDATE verwenden
2. Ähnlicher Name im REGISTRY? → UPDATE verwenden
3. Teilweise Überlappung? → UPDATE verwenden
4. Wirklich neu? → Erst dann CREATE
5. AKTIV nach Verbindungsmöglichkeiten suchen: CONNECT zwischen thematisch verwandten Entities

QUALITÄTSPRINZIPIEN:
- ⚠️  KRITISCH: In from_id/to_id NUR IDs verwenden (af_1, proj_2), NIEMALS Titel ("Mobilität")!
- Kopieren Sie IDs EXAKT aus der ID-MAPPING-TABELLE oben
- Fügen Sie immer Quellenangaben (source_pages, source_quote) hinzu  
- Seien Sie konservativ bei neuen Handlungsfeldern
- Bei Unsicherheit: UPDATE verwenden statt CREATE
- Erstellen Sie CONNECT-Operationen wann immer thematische Zusammenhänge erkennbar sind

Antworten Sie AUSSCHLIESSLICH mit der Operations-Liste im JSON-Format."""

# Common exclusion rules for action fields
NOT_CONSOLIDATE_EXAMPLES = """DIESE FÄLLE NICHT konsolidieren:
❌ "Klimaschutz" und "Energie" → Bleiben getrennt (verschiedene Schwerpunkte)
❌ "Umwelt" und "Nachhaltigkeit" → Bleiben getrennt (unterschiedliche Aspekte)
❌ "Mobilität" und "ÖPNV" → Bleiben getrennt (allgemein vs. spezifisch)
❌ "Verkehr" und "Radverkehr" → Bleiben getrennt (allgemein vs. spezifisch)
❌ "Wirtschaft" und "Innovation" → Bleiben getrennt (verschiedene Bereiche)
❌ "Kultur" und "Bildung" → Bleiben getrennt (verschiedene Ressorts)
❌ "Wohnen" und "Stadtentwicklung" → Bleiben getrennt (verschiedene Planungsebenen)
❌ "Soziales" und "Gesundheit" → Bleiben getrennt (verschiedene Zuständigkeiten)
❌ "Integration" und "Teilhabe" → Bleiben getrennt (verschiedene Konzepte)"""

CONSOLIDATE_EXAMPLES = """NUR DIESE FÄLLE konsolidieren:
✅ "Mobilität" + "Mobilität" → "Mobilität" (exaktes Duplikat)
✅ "Verkehr und Mobilität" + "Mobilität und Verkehr" → "Mobilität und Verkehr" (fast identisch)
✅ "Klimaschutz" + "Klimaschutz und Energie" → "Klimaschutz und Energie" (eines ist Teilmenge)"""

# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================


def prepare_chunks_for_extraction(
    page_aware_text: list[tuple[str, int]],
    source_id: str,
    max_chars: int,
    min_chars: int,
    max_chunks: int,
) -> list[tuple[str, list[int]]]:
    """Prepare chunks for extraction with shared logic."""
    from src.core.config import ENHANCED_CHUNK_OVERLAP
    from src.processing.chunker import chunk_for_llm_with_pages

    if not page_aware_text:
        error_msg = "No page-aware text provided"
        raise ValueError(error_msg)

    print(
        f"📝 Chunking with settings: {min_chars}-{max_chars} chars, {ENHANCED_CHUNK_OVERLAP*100}% overlap"
    )

    chunks_with_pages = chunk_for_llm_with_pages(
        page_aware_text=page_aware_text,
        max_chars=max_chars,
        min_chars=min_chars,
        doc_title=f"Strategiedokument {source_id}",
        add_overlap=True,
    )

    if not chunks_with_pages:
        error_msg = "No chunks created from text"
        raise ValueError(error_msg)

    # Apply chunk limit for performance
    if len(chunks_with_pages) > max_chunks:
        print(f"⚡ Using first {max_chunks} chunks (performance mode)")
        chunks_with_pages = chunks_with_pages[:max_chunks]

    print(f"📄 Processing {len(chunks_with_pages)} chunks")
    return chunks_with_pages


def execute_llm_extraction(
    llm_provider,
    prompt: str,
    response_model,
    system_message: str,
    log_file_path: str | None,
    log_context: str,
    chunk_index: int,
    override_num_predict: int | None = None,
):
    """Execute LLM extraction with shared error handling."""
    try:
        result = llm_provider.query_structured(
            prompt=prompt,
            response_model=response_model,
            system_message=system_message,
            log_file_path=log_file_path,
            log_context=log_context,
            override_num_predict=override_num_predict,
        )
        return result
    except Exception as e:
        print(f"❌ Error processing chunk {chunk_index + 1}: {e}")
        return None


def create_unique_entity_id(
    prefix: str, title: str, used_ids: set, id_counters: dict
) -> str:
    """Generate guaranteed unique ID with shared logic."""
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


def format_entity_registry(current_state) -> str:
    """Create a complete, searchable entity registry showing ALL entities."""
    # List ALL entity titles (no truncation)
    action_fields = [af.content.get("title", "") for af in current_state.action_fields]
    projects = [p.content.get("title", "") for p in current_state.projects]
    measures = [m.content.get("title", "") for m in current_state.measures]
    indicators = [i.content.get("title", "") for i in current_state.indicators]

    # Format as readable lists
    registry = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ ENTITY REGISTRY - CHECK BEFORE ANY CREATE OPERATION                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

ACTION FIELDS ({len(action_fields)} total):
{', '.join(action_fields) if action_fields else '[None yet]'}

PROJECTS ({len(projects)} total):
{', '.join(projects) if projects else '[None yet]'}

MEASURES ({len(measures)} total):
{', '.join(measures) if measures else '[None yet]'}

INDICATORS ({len(indicators)} total):
{', '.join(indicators) if indicators else '[None yet]'}

═══════════════════════════════════════════════════════════════════════════════
"""
    return registry


def format_entity_id_mapping(current_state) -> str:
    """Create complete ID lookup table for UPDATE/CONNECT operations."""
    mappings = []

    # Include ALL entities with their IDs
    if current_state.action_fields:
        mappings.append("ACTION FIELD IDs:")
        for af in current_state.action_fields:
            mappings.append(f"  {af.id} → {af.content.get('title', '')}")

    if current_state.projects:
        mappings.append("\nPROJECT IDs:")
        for p in current_state.projects:
            mappings.append(f"  {p.id} → {p.content.get('title', '')}")

    if current_state.measures:
        mappings.append("\nMEASURE IDs:")
        for m in current_state.measures:
            mappings.append(f"  {m.id} → {m.content.get('title', '')}")

    if current_state.indicators:
        mappings.append("\nINDICATOR IDs:")
        for i in current_state.indicators:
            mappings.append(f"  {i.id} → {i.content.get('title', '')}")

    return (
        "\n".join(mappings) if mappings else "No entities yet - use CREATE operations"
    )


def format_context_json(context_data) -> str:
    """Format context as entity registry + ID mappings (not JSON dump)."""
    if not context_data:
        return "ERSTER CHUNK: Noch keine Entities extrahiert. Beginnen Sie mit CREATE-Operationen."

    # Use the new compact registry format
    registry = format_entity_registry(context_data)
    id_mapping = format_entity_id_mapping(context_data)

    return f"""{registry}

{id_mapping}

KRITISCHE REGELN:
1. IMMER prüfen ob Entity schon im REGISTRY existiert
2. Bei ähnlichen Namen → UPDATE statt CREATE
3. Beispiele für Duplikate:
   - "Mobilität und Verkehr" = "Mobilität & Verkehr" = "Verkehrswesen"
   - "Radwegeausbau" = "Ausbau Radwege" = "Radverkehrsnetz"
   - "CO2-Reduktion" = "CO₂-Reduktion" = "Kohlendioxid-Reduktion"
4. NUR CREATE wenn wirklich neu und einzigartig"""


def create_extraction_prompt(
    template_type: str,
    chunk_text: str,
    context_data=None,
    page_numbers: list[int] | None = None,
) -> str:
    """Create extraction prompts using templates."""
    if template_type == "simplified":
        context_text = format_context_json(context_data)
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

    elif template_type == "operations":
        context_text = (
            format_context_json(context_data)
            if context_data
            else "ERSTER CHUNK: Noch keine Entities extrahiert. Beginnen Sie mit CREATE-Operationen."
        )
        page_list = ", ".join(map(str, sorted(page_numbers))) if page_numbers else "N/A"

        return f"""Analysieren Sie diesen Textabschnitt und erstellen Sie OPERATIONEN zur Strukturerweiterung.

{context_text}

VERPFLICHTENDE PRÜFUNG vor jeder CREATE-Operation:
1. Entity bereits im ENTITY REGISTRY? → UPDATE verwenden
2. Ähnlicher Name im REGISTRY? → UPDATE verwenden
3. Teilweise Überlappung? → UPDATE verwenden
4. Wirklich neu? → Erst dann CREATE

VERBINDUNGSREGELN (KRITISCH):
- CONNECT-Operationen sind das ZIEL - erstellen Sie sie aktiv!
- Verbindungen zu bestehenden UND neuen Entities sind gewünscht
- Suchen Sie aktiv nach thematischen Zusammenhängen für Verbindungen

TEXTABSCHNITT (Seiten {page_list}):
{chunk_text}

QUALITÄTSKONTROLLE:
- ⚠️  CONNECT from_id/to_id: NUR IDs aus ID-MAPPING verwenden (af_1, proj_2), NICHT Titel!
- FALSCH: "to_id": "Mobilität und Verkehr" ❌
- RICHTIG: "to_id": "af_1" ✅
- Fügen Sie IMMER source_pages und source_quote hinzu (außer bei CONNECT)
- Konfidenz ≥ 0.7 für CONNECT, ≥ 0.8 für CREATE
- Bevorzugen Sie UPDATE über CREATE bei jeder semantischen Überlappung

Antworten Sie NUR mit der Operations-Liste im JSON-Format:
{{"operations": [...]}}"""

    else:
        error_msg = f"Unknown template type: {template_type}"
        raise ValueError(error_msg)


def add_page_attribution_to_enhanced_result(result, page_numbers: list[int]) -> None:
    """Add page attribution to all entities in enhanced result."""
    page_str = f"Seiten {', '.join(map(str, sorted(page_numbers)))}"

    # Add page attribution to all entity types
    for entity_list in [
        result.action_fields,
        result.projects,
        result.measures,
        result.indicators,
    ]:
        for entity in entity_list:
            if "page_source" not in entity.content:
                entity.content["page_source"] = page_str


def merge_enhanced_results(results: list) -> dict | None:
    """Merge multiple enhanced results with simple concatenation."""
    if not results:
        return None

    if len(results) == 1:
        return results[0]

    # Import here to avoid circular imports
    from src.core.schemas import EnrichedReviewJSON

    # Simple merge - concatenate all lists
    merged = EnrichedReviewJSON(
        action_fields=[], projects=[], measures=[], indicators=[]
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
    from src.core.llm_providers import get_llm_provider
    from src.core.schemas import ExtractionResult

    system_message = DEDUPLICATION_SYSTEM_MESSAGE

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

{NOT_CONSOLIDATE_EXAMPLES}

{CONSOLIDATE_EXAMPLES}

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
    resolved_structures: list[dict[str, Any]], _: Any
) -> Any:
    """
    Rebuild enhanced structure from resolved intermediate structures with unique ID validation.

    Args:
        resolved_structures: Entity-resolved structures in intermediate format

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
        """Generate guaranteed unique ID using shared utility"""
        return create_unique_entity_id(prefix, title, used_ids, id_counters)

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
        msg = (
            "❌ CRITICAL: Duplicate IDs found after rebuild - this should never happen!"
        )
        raise ValueError(msg)

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

    from src.core.config import (
        ENHANCED_CHUNK_MAX_CHARS,
        ENHANCED_CHUNK_MIN_CHARS,
        ENHANCED_CHUNK_OVERLAP,
        FAST_EXTRACTION_MAX_CHUNKS,
        LLM_BACKEND,
    )
    from src.core.llm_providers import get_llm_provider
    from src.core.schemas import EnrichedReviewJSON

    print(f"🔄 Starting direct enhanced extraction for {source_id}")
    start_time = time.time()

    if not page_aware_text:
        print("⚠️ No page-aware text provided")
        return None

    # Step 1: Create smaller chunks optimized for focused extraction
    chunks_with_pages = prepare_chunks_for_extraction(
        page_aware_text,
        source_id,
        ENHANCED_CHUNK_MAX_CHARS,
        ENHANCED_CHUNK_MIN_CHARS,
        FAST_EXTRACTION_MAX_CHUNKS,
    )

    # Step 2: Extract from each chunk using simplified prompts
    # Initialize Global Entity Registry for consistency across chunks
    from src.processing.global_registry import GlobalEntityRegistry

    global_registry = GlobalEntityRegistry()

    all_results = []
    llm_provider = get_llm_provider()

    for i, (chunk_text, page_numbers) in enumerate(chunks_with_pages):
        print(
            f"🔍 Processing chunk {i+1}/{len(chunks_with_pages)} (pages {page_numbers})"
        )

        # Get full accumulated extraction state instead of just entity names
        accumulated_json = merge_enhanced_results(all_results) if all_results else None

        # Create context-aware extraction prompt with full JSON structure
        system_message = EXTRACTION_SYSTEM_MESSAGE
        main_prompt = create_extraction_prompt(
            "simplified", chunk_text, accumulated_json
        )

        # Enhanced log context
        enhanced_log_context = f"direct_enhanced_{source_id}_chunk_{i+1}"

        result = execute_llm_extraction(
            llm_provider,
            main_prompt,
            EnrichedReviewJSON,
            system_message,
            log_file_path,
            enhanced_log_context,
            i,
        )

        if result:
            # Register action field entities in the global registry
            for action_field in result.action_fields:
                # Handle both 'name' and 'title' fields (LLM outputs to 'title')
                original_name = action_field.content.get(
                    "name"
                ) or action_field.content.get("title", "")
                if original_name:
                    canonical_name = global_registry.register_entity(original_name)
                    # Write back to BOTH fields for compatibility
                    action_field.content["name"] = canonical_name
                    action_field.content["title"] = canonical_name

            # Add page attribution to all entities
            add_page_attribution_to_enhanced_result(result, page_numbers)
            all_results.append(result)
            print(
                f"✅ Chunk {i+1}: {len(result.action_fields)} action fields, {len(result.projects)} projects, {len(result.indicators)} indicators"
            )
        else:
            print(f"⚠️ No result from chunk {i+1}")

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
            },
        },
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

    from src.core.config import (
        ENHANCED_CHUNK_MAX_CHARS,
        ENHANCED_CHUNK_MIN_CHARS,
        FAST_EXTRACTION_MAX_CHUNKS,
    )
    from src.core.llm_providers import get_llm_provider
    from src.core.operations_schema import ExtractionOperations
    from src.core.schemas import EnrichedReviewJSON
    from src.extraction.operations_executor import OperationExecutor

    print(f"🔄 Starting operations-based extraction for {source_id}")
    start_time = time.time()

    if not page_aware_text:
        print("⚠️ No page-aware text provided")
        return None

    # Step 1: Create smaller chunks optimized for focused extraction
    chunks_with_pages = prepare_chunks_for_extraction(
        page_aware_text,
        source_id,
        ENHANCED_CHUNK_MAX_CHARS,
        ENHANCED_CHUNK_MIN_CHARS,
        FAST_EXTRACTION_MAX_CHUNKS,
    )

    # Step 2: Initialize empty extraction state and operations executor
    current_state = EnrichedReviewJSON(
        action_fields=[], projects=[], measures=[], indicators=[]
    )

    executor = OperationExecutor()
    llm_provider = get_llm_provider()
    all_operation_logs = []

    # Step 3: Process each chunk with operations
    for i, (chunk_text, page_numbers) in enumerate(chunks_with_pages):
        print(
            f"🔍 Processing chunk {i+1}/{len(chunks_with_pages)} (pages {page_numbers})"
        )

        # Create operations-focused prompt
        system_message = OPERATIONS_SYSTEM_MESSAGE
        main_prompt = create_extraction_prompt(
            "operations", chunk_text, current_state, page_numbers
        )

        # Enhanced log context
        enhanced_log_context = f"operations_{source_id}_chunk_{i+1}"

        operations_result = execute_llm_extraction(
            llm_provider,
            main_prompt,
            ExtractionOperations,
            system_message,
            log_file_path,
            enhanced_log_context,
            i,
        )

        if operations_result and operations_result.operations:
            # Filter out invalid operations instead of skipping entire chunk
            from src.extraction.operations_executor import validate_operations

            validation_errors = validate_operations(
                operations_result.operations, current_state
            )

            if validation_errors:
                print(
                    f"⚠️ Chunk {i+1}: {len(validation_errors)} operation validation errors:"
                )
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

                print(
                    f"   Proceeding with {len(valid_operations)}/{len(operations_result.operations)} valid operations"
                )
                operations_result.operations = valid_operations

            if operations_result.operations:  # Only proceed if we have valid operations
                # Apply validated operations to current state
                try:
                    new_state, operation_log = executor.apply_operations(
                        current_state, operations_result.operations, chunk_index=i
                    )

                    # Only update current_state if operations were successfully applied
                    if operation_log.successful_operations > 0:
                        current_state = new_state
                        all_operation_logs.append(operation_log)

                        print(
                            f"✅ Chunk {i+1}: {operation_log.successful_operations}/{operation_log.total_operations} operations applied"
                        )
                        print(
                            f"📊 Current state: {len(current_state.action_fields)} action fields, "
                            f"{len(current_state.projects)} projects, {len(current_state.measures)} measures, "
                            f"{len(current_state.indicators)} indicators"
                        )
                    else:
                        print(
                            f"⚠️ Chunk {i+1}: No operations succeeded, keeping previous state"
                        )
                        all_operation_logs.append(operation_log)

                except Exception as op_error:
                    print(f"❌ Chunk {i+1}: Error applying operations: {op_error}")
                    print("   Keeping previous state to preserve integrity")
                    continue
            else:
                print(f"⚠️ Chunk {i+1}: All operations filtered out, skipping")

        else:
            print(f"⚠️ No operations from chunk {i+1}")

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


def apply_conservative_entity_resolution(result):  # EnrichedReviewJSON type
    """Apply conservative entity resolution - only merge exact matches."""
    try:
        # Convert to format expected by existing entity resolution
        structures = []

        for af in result.action_fields:
            af_dict = {
                "action_field": af.content.get("title", af.content.get("name", "")),
                "projects": [],
            }

            # Find connected projects (using parent→child connections)
            for proj in result.projects:
                # Check if this project is connected from the action field
                if proj.id in [conn.target_id for conn in af.connections]:
                    proj_dict = {
                        "title": proj.content.get("title", ""),
                        "measures": [],
                        "indicators": [],
                    }

                    # Find connected measures and indicators (using parent→child connections)
                    for msr in result.measures:
                        # Check if this measure is connected from the project
                        if msr.id in [conn.target_id for conn in proj.connections]:
                            proj_dict["measures"].append(
                                {
                                    "title": msr.content.get("title", ""),
                                    "description": msr.content.get("description", ""),
                                }
                            )

                    for ind in result.indicators:
                        # Check if this indicator is connected from the project
                        if ind.id in [conn.target_id for conn in proj.connections]:
                            proj_dict["indicators"].append(
                                {
                                    "title": ind.content.get("title", ""),
                                    "description": ind.content.get("description", ""),
                                }
                            )

                    af_dict["projects"].append(proj_dict)

            structures.append(af_dict)

        # Check if structures have meaningful content
        total_projects = sum(len(s.get("projects", [])) for s in structures)
        if total_projects == 0:
            print(
                "⚠️ No connected projects found, bypassing entity resolution to preserve entities"
            )
            return result

        print(
            f"🔗 Found {total_projects} connected projects in {len(structures)} structures, proceeding with resolution"
        )

        # Apply existing entity resolution
        resolved_structures = apply_entity_resolution(structures)

        # Convert back to enhanced format using the working rebuild function
        return rebuild_enhanced_structure_from_resolved(resolved_structures, result)

    except Exception as e:
        print(f"⚠️ Entity resolution failed, returning unresolved: {e}")
        return result


# ============================================================================
# COMMON ROUTE HELPERS FOR REDUCING ROUTES.PY LINE COUNT
# ============================================================================


def create_error_response(
    endpoint: str,
    error_type: str,
    status_code: int,
    source_id: str,
    exception: Exception,
    start_time: float,
    monitor: Any,
    stage: str,
    suggestion: str | None = None,
) -> JSONResponse:
    """Create standardized error response with logging and monitoring."""
    from src.utils import log_api_response

    monitor.log_error(stage, exception)
    response_time = time.time() - start_time
    log_api_response(endpoint, status_code, response_time)

    content = {
        "error": error_type,
        "detail": str(exception),
        "source_id": source_id,
    }
    if suggestion:
        content["suggestion"] = suggestion

    return JSONResponse(content=content, status_code=status_code)


async def save_json_file(
    data: Any,
    filename: str,
    upload_folder: str,
) -> str:
    """Save JSON data to file and return the full path."""
    file_path = os.path.join(upload_folder, filename)
    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(data, ensure_ascii=False, indent=2))
    return file_path


@contextmanager
def monitor_stage(monitor: Any, stage_name: str, **kwargs):
    """Context manager for monitoring stages."""
    monitor.start_stage(stage_name, **kwargs)
    try:
        yield
    finally:
        monitor.end_stage(stage_name, **kwargs)


def _load_pages_from_file(source_id: str) -> list[tuple[str, int]]:
    """Load page-aware text from the saved pages file."""
    import re

    from src.core import UPLOAD_FOLDER

    pages_filename = os.path.splitext(source_id)[0] + "_pages.txt"
    pages_path = os.path.join(UPLOAD_FOLDER, pages_filename)

    if not os.path.exists(pages_path):
        error_msg = f"Pages file not found: {pages_filename}"
        raise FileNotFoundError(error_msg)

    page_aware_text = []
    with open(pages_path, encoding="utf-8") as f:
        content = f.read()

    # Parse pages using regex
    page_pattern = re.compile(
        r"\[Page (\d+)\]\n(.*?)(?=\n\n\[Page|\Z)",
        re.DOTALL,
    )
    matches = page_pattern.findall(content)

    for page_num_str, page_text in matches:
        try:
            page_num = int(page_num_str)
            page_text = page_text.strip()
            if page_text:  # Only add non-empty pages
                page_aware_text.append((page_text, page_num))
        except ValueError:
            print(f"⚠️ Skipping invalid page number: {page_num_str}")

    return page_aware_text


async def load_and_validate_pages(
    source_id: str, monitor: Any
) -> list[tuple[str, int]]:
    """Load and validate page-aware text, with monitoring."""
    with monitor_stage(monitor, "file_loading", source_id=source_id):
        page_aware_text = _load_pages_from_file(source_id)

        if not page_aware_text:
            pages_filename = os.path.splitext(source_id)[0] + "_pages.txt"
            error_msg = f"No valid page content found in {pages_filename}"
            raise ValueError(error_msg)

        print(f"📄 Loaded {len(page_aware_text)} pages from page-aware text file")
        return page_aware_text


def create_success_response(
    endpoint: str,
    source_id: str,
    extraction_result: dict,
    start_time: float,
    additional_fields: dict | None = None,
) -> JSONResponse:
    """Create standardized success response with logging."""
    from src.utils import log_api_response

    response_time = time.time() - start_time
    log_api_response(endpoint, 200, response_time)

    response_data = {
        "source_id": source_id,
        "extraction_result": extraction_result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if additional_fields:
        response_data.update(additional_fields)

    return JSONResponse(
        content=jsonable_encoder(response_data),
        status_code=200,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
