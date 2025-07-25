import json
import re
from typing import Any

import json5

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
from src.core.constants import ENGLISH_FILTER_TERMS, QUANTITATIVE_PATTERNS

from .prompts import (
    STAGE1_SYSTEM_MESSAGE,
    get_stage1_prompt,
    get_stage2_prompt,
    get_stage2_system_message,
    get_stage3_prompt,
    get_stage3_system_message,
)

# prepare_llm_chunks is imported from semantic_llm_chunker


def extract_action_fields_only(chunks: list[str]) -> list[str]:
    """
    Stage 1: Extract just action field names from all chunks.

    This is a lightweight extraction that only identifies the main categories
    (Handlungsfelder) without extracting projects or details.
    """
    all_action_fields = set()  # Use set to automatically deduplicate

    # Use enhanced prompt for mixed-topic robustness
    system_message = STAGE1_SYSTEM_MESSAGE

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        print(
            f"🔍 Stage 1: Scanning chunk {i+1}/{len(chunks)} for action fields ({len(chunk)} chars)"
        )

        prompt = get_stage1_prompt(chunk)

        result = query_ollama_structured(
            prompt=prompt,
            response_model=ActionFieldList,
            system_message=system_message,
            temperature=MODEL_TEMPERATURE,
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
    """
    if not fields:
        return []

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
    system_message = get_stage2_system_message(action_field)

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        # Remove quick check - with mixed topics, action field might not be explicitly mentioned

        print(
            f"🔎 Stage 2: Searching chunk {i+1}/{len(chunks)} for {action_field} projects"
        )

        prompt = get_stage2_prompt(chunk, action_field)

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
    system_message = get_stage3_system_message(action_field, project_title)

    # Process ALL chunks - indicators might be separated from project mentions
    # in mixed-topic chunks
    print(f"🔬 Stage 3: Analyzing ALL {len(chunks)} chunks for {project_title} details")

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        prompt = get_stage3_prompt(chunk, action_field, project_title)

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


def validate_extraction_schema(data: Any) -> bool:
    """
    Validate that extracted data matches the expected schema.

    Expected schema: List of action field objects with projects.
    """
    if not isinstance(data, list):
        return False

    for item in data:
        if not isinstance(item, dict):
            return False

        # Required fields
        if "action_field" not in item or "projects" not in item:
            return False

        if not isinstance(item["action_field"], str) or not isinstance(
            item["projects"], list
        ):
            return False

        # Check for English content in action field
        action_field = item["action_field"]
        if any(term in action_field for term in ENGLISH_FILTER_TERMS):
            print(f"⚠️ Rejecting English action field: {action_field}")
            return False

        # Validate project structure
        for project in item["projects"]:
            if not isinstance(project, dict):
                return False

            if "title" not in project:
                return False

            if not isinstance(project["title"], str):
                return False

            # Check for English content in project title
            if any(term in project["title"] for term in ENGLISH_FILTER_TERMS):
                print(f"⚠️ Rejecting English project title: {project['title']}")
                return False

            # measures and indicators are optional but must be lists if present
            if "measures" in project and not isinstance(project["measures"], list):
                return False

            if "indicators" in project and not isinstance(project["indicators"], list):
                return False

    return True


def reclassify_measures_to_indicators(
    extracted_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Post-process extracted data to move quantitative measures to indicators.

    Scans measures for numbers, percentages, dates, and other quantitative patterns
    and reclassifies them as indicators.
    """
    # Use patterns from constants
    combined_pattern = "|".join(QUANTITATIVE_PATTERNS)

    for action_field in extracted_data:
        for project in action_field.get("projects", []):
            if project.get("measures"):
                new_measures = []
                indicators = project.get("indicators", [])

                for measure in project["measures"]:
                    # Check if measure contains quantitative data
                    if re.search(combined_pattern, measure, re.IGNORECASE):
                        # Move to indicators
                        indicators.append(measure)
                    else:
                        # Keep as measure
                        new_measures.append(measure)

                project["measures"] = new_measures
                if indicators:  # Only add if there are indicators
                    project["indicators"] = indicators

    return extracted_data


def extract_structures_with_retry(
    chunk_text: str, max_retries: int = EXTRACTION_MAX_RETRIES
) -> list[dict[str, Any]]:
    """
    Extract structures from text using Ollama structured output.
    """
    system_message = """Extrahiere Handlungsfelder und deren Projekte aus kommunalen Strategiedokumenten.

Jedes Handlungsfeld enthält Projekte. Für jedes Projekt extrahiere:
- Titel
- Maßnahmen (konkrete Aktionen, Umsetzungsschritte)
- Indikatoren (ALLE Zahlen, Prozentsätze, Zielwerte, Termine)

WICHTIG: Indikatoren sind IMMER quantitative Angaben:
- Prozentangaben: "55% Reduktion", "um 30% steigern"
- Zeitangaben: "bis 2030", "ab 2025", "jährlich"
- Mengenangaben: "500 Ladepunkte", "18 km", "1000 Wohneinheiten"
- Vergleiche: "Verdopplung", "Halbierung", "30% weniger"
- Aufzählungen mit Zahlen: "24 Frauenzellstraße, 25 Sallern, 26 Stadtamhof"

BEISPIELE für die Unterscheidung:
✓ "24 Frauenzellstraße, 25 Sallern" → INDIKATOR (enthält Zahlen/Standorte)
✓ "500 Ladepunkte" → INDIKATOR (Zahl mit Einheit)
✗ "Verbesserung der Stadtbahninfrastruktur" → MAẞNAHME (keine Zahl)
✗ "Förderung von Innovationen" → MAẞNAHME (keine quantitative Angabe)

REGEL: Enthält der Text eine Zahl, ein Datum oder Prozent? → INDIKATOR. Sonst → MAẞNAHME.

Extrahiere den kompletten Inhalt auf Deutsch."""

    prompt = f"""Extrahiere alle Handlungsfelder und deren Projekte aus diesem kommunalen Strategietext:

{chunk_text.strip()}"""

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

            # Post-process to reclassify measures containing numbers as indicators
            extracted_data = reclassify_measures_to_indicators(extracted_data)
            print(
                "📊 Post-processing: Reclassified quantitative measures as indicators"
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

    system_message = """Erweitere die bestehende Extraktion mit neuen Informationen aus dem kommunalen Dokument.

WICHTIGE REGELN:
1. BEHALTE alle bestehenden Daten - entferne nichts
2. ERGÄNZE bestehende Projekte mit neuen Maßnahmen/Indikatoren
3. VERSCHMELZE doppelte Projekte (gleicher Titel = gleiches Projekt)
4. FÜGE neue Handlungsfelder und Projekte hinzu
5. SUCHE aktiv nach übersehenen Indikatoren

BESONDERER FOKUS auf Indikatoren - finde ALLE quantitativen Informationen:
- Zahlen mit Einheiten: "500 Ladepunkte", "18 km", "1000 Wohneinheiten"
- Prozentangaben: "40% Reduktion", "um 30% senken"
- Zeitziele: "bis 2030", "ab 2025", "innerhalb 5 Jahren"
- Häufigkeiten: "jährlich", "pro Jahr", "monatlich"
- Vergleiche: "Verdopplung", "Halbierung", "30% weniger"

Alles auf Deutsch extrahieren."""

    prompt = f"""Current extraction state has {len(accumulated_data.get('action_fields', []))} action fields:

{json.dumps(accumulated_data, indent=2, ensure_ascii=False)}

Now process this NEW text and enhance the above structure:

{chunk_text.strip()}

Return the COMPLETE enhanced JSON with all existing data plus new findings.
Remember: ENHANCE and ADD, never remove."""

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
