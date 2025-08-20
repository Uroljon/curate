import json
import re
from typing import Any

from src.core import (
    CHUNK_WARNING_THRESHOLD,
    EXTRACTION_MAX_RETRIES,
    MODEL_TEMPERATURE,
    ActionFieldList,
    ExtractionResult,
    ProjectDetails,
    ProjectDetailsEnhanced,
    ProjectList,
    query_ollama_structured,
    query_ollama_with_thinking_mode,
)

from .prompts import (
    STAGE1_SYSTEM_MESSAGE,
    get_stage1_prompt,
    get_stage2_prompt,
    get_stage2_system_message,
    get_stage3_prompt,
    get_stage3_system_message,
)

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




def extract_structures_with_retry(
    chunk_text: str,
    max_retries: int = EXTRACTION_MAX_RETRIES,
    log_file_path: str | None = None,
    log_context: str | None = None,
) -> list[dict[str, Any]]:
    """
    Extract structures from text using Ollama structured output.
    """
    system_message = """Sie sind ein Experte für die Analyse deutscher kommunaler Strategiedokumente.

KRITISCHE ANWEISUNG: Verwenden Sie AUSSCHLIESSLICH Informationen aus dem bereitgestellten Quelldokument.
Nutzen Sie NIEMALS Ihr Vorwissen oder Annahmen - nur den vorliegenden Text.

VERFAHREN (Quote-Before-Answer):
1. ZITATE EXTRAHIEREN: Identifizieren Sie relevante Textpassagen im Dokument
2. ANALYSE: Basieren Sie Ihre Extraktion ausschließlich auf diesen Zitaten
3. VALIDIERUNG: Jeder extrahierte Punkt muss direkt im Quelltext nachweisbar sein

DEUTSCHE VERWALTUNGSSPRACHE - BEISPIELE (nicht limitierend):
✓ Handlungsfelder können vielfältig sein: Mobilität, Klimaschutz, Energie, Wohnen, Bildung,
  Soziales, Wirtschaft, Kultur, Sport, Digitalisierung, Gesundheit, Sicherheit, Verwaltung, etc.
✓ Extrahieren Sie ALLE im Text gefundenen Handlungsfelder, nicht nur die genannten Beispiele
✓ Verwaltungsterminologie: Bescheid, Verordnung, Verwaltungsakt, Maßnahme, Indikator

ABSOLUT VERBOTEN - Englische Begriffe:
✗ "Current State", "Future Vision", "Urban Planning", "Smart City"
✗ Jegliche englische Fachterminologie

EXTRAKTION PRO HANDLUNGSFELD:
- Titel: Prägnante deutsche Bezeichnung (max. 100 Zeichen)
- Maßnahmen: Konkrete Umsetzungsschritte aus dem Dokument
- Indikatoren: Quantitative UND qualitative Zielgrößen aus dem Text

INDIKATOREN (beide Typen erfassen):
- Quantitativ: "55% Reduktion bis 2030", "500 Ladepunkte", "18 km Radwege"
- Qualitativ: "Verbesserung der Luftqualität", "Stärkung des Zusammenhalts"

Antworten Sie nur basierend auf explizit im Dokument gefundenen Informationen.
Falls Informationen nicht verfügbar sind: "Information im Quelldokument nicht verfügbar"."""

    prompt = f"""QUELLDOKUMENT:
========
{chunk_text.strip()}
========

ARBEITSSCHRITTE:

1. RELEVANTE ZITATE IDENTIFIZIEREN:
Suchen Sie alle Textpassagen, die Handlungsfelder, Projekte, Maßnahmen oder Indikatoren erwähnen.

2. DEUTSCHE HANDLUNGSFELDER EXTRAHIEREN:
Basierend nur auf den gefundenen Zitaten - identifizieren Sie ALLE unterschiedlichen Handlungsfelder.
KRITISCH: Extrahieren Sie JEDES einzigartige Handlungsfeld aus diesem Chunk, auch wenn es nur einmal erwähnt wird!
Verschiedene Chunks können verschiedene Handlungsfelder enthalten - erfassen Sie die Vielfalt!

3. PROJEKTE UND DETAILS ZUORDNEN:
Für jedes gefundene Handlungsfeld - extrahieren Sie nur die im Text explizit erwähnten Projekte und Details.

WICHTIG:
- Verwenden Sie ausschließlich Informationen aus dem obigen Quelldokument
- Extrahieren Sie ALLE Handlungsfelder, die Sie im Text finden
- Begrenzen Sie sich NICHT auf Standard-Handlungsfelder
- Jeder Chunk kann unterschiedliche Handlungsfelder enthalten"""

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

    system_message = """Erweitere die bestehende Extraktion mit neuen Informationen aus dem kommunalen Dokument.

WICHTIGE REGELN:
1. BEHALTE alle bestehenden Daten - entferne nichts
2. ERGÄNZE bestehende Projekte mit neuen Maßnahmen/Indikatoren
3. VERSCHMELZE doppelte Projekte (gleicher Titel = gleiches Projekt)
4. FÜGE neue Handlungsfelder und Projekte hinzu
5. SUCHE aktiv nach übersehenen Indikatoren

PROJEKTTITEL REGELN:
- Verwende prägnante, offizielle Bezeichnungen (max. 100 Zeichen)
- RICHTIG: "Stadtbahn Regensburg", "Klimaschutzkonzept 2030", "Digitales Rathaus"
- FALSCH: "Weiterentwicklung der bisherigen Dienstleistungsachse zu einer Dienstleistungs-, Technologie- und Wissenschaftsachse"
- Bei langen Beschreibungen: Extrahiere den Kernnamen oder erstelle eine kurze, treffende Bezeichnung
- Deutsche Komposita sind erlaubt: "Nachhaltigkeitsorientierte Stadtentwicklungskonzeption"

BESONDERER FOKUS auf Indikatoren - finde ALLE quantitativen Informationen:
- Zahlen mit Einheiten: "500 Ladepunkte", "18 km", "1000 Wohneinheiten"
- Prozentangaben: "40% Reduktion", "um 30% senken"
- Zeitziele: "bis 2030", "ab 2025", "innerhalb 5 Jahren"
- Häufigkeiten: "jährlich", "pro Jahr", "monatlich"
- Vergleiche: "Verdopplung", "Halbierung", "30% weniger"

QUALITATIVE INDIKATOREN sind auch wichtig:
- "Deutliche Reduktion der CO2-Emissionen" (später quantifizieren)
- "Erhöhung der Biodiversität" (Zahlen folgen eventuell später)
- "Verbesserung der Luftqualität" (konkrete Werte können später kommen)

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


def _determine_thinking_mode(chunk: str, action_field: str, project_title: str) -> str:
    """
    Determine the appropriate thinking mode based on chunk complexity.

    Args:
        chunk: Text chunk to analyze
        action_field: The action field context
        project_title: The project being analyzed

    Returns:
        str: Thinking mode to use ("analytical", "comparative", "systematic", "contextual")
    """
    chunk_lower = chunk.lower()

    # Count complexity indicators
    complexity_score = 0

    # Multiple projects mentioned - needs comparative analysis
    if chunk_lower.count("projekt") > 1 or chunk_lower.count("maßnahme") > 3:
        complexity_score += 2

    # Contains tables or structured data - needs systematic approach
    if any(indicator in chunk for indicator in ["|-", "|:", "Tabelle", "Tab.", "Nr."]):
        complexity_score += 2

    # Multiple Handlungsfelder - needs contextual understanding
    action_keywords = ["handlungsfeld", "bereich", "themenfeld", "schwerpunkt"]
    if sum(chunk_lower.count(keyword) for keyword in action_keywords) > 1:
        complexity_score += 1

    # Complex numerical data - needs analytical precision

    numerical_patterns = [
        r"\d+%",
        r"\d+\s*Euro",
        r"\d+\s*km",
        r"bis\s+\d{4}",
        r"\d+\.\d+",
    ]
    numerical_matches = sum(
        len(re.findall(pattern, chunk)) for pattern in numerical_patterns
    )
    if numerical_matches > 3:
        complexity_score += 2

    # Mixed measures and indicators - needs comparative analysis
    measure_indicators = [
        "entwicklung",
        "einführung",
        "umsetzung",
        "anzahl",
        "prozent",
        "reduktion",
    ]
    mixed_count = sum(1 for indicator in measure_indicators if indicator in chunk_lower)
    if mixed_count > 4:
        complexity_score += 1

    # Choose thinking mode based on complexity
    if complexity_score >= 5:
        return "contextual"  # Highest complexity - deep context understanding
    elif complexity_score >= 3:
        return "systematic"  # High complexity - methodical approach
    elif complexity_score >= 2:
        return "comparative"  # Medium complexity - compare and contrast
    else:
        return "analytical"  # Lower complexity - standard step-by-step analysis


def _extract_document_hierarchy(chunk: str) -> dict:
    """
    Extract document hierarchy information from a chunk.

    Args:
        chunk: Text chunk to analyze for structure information

    Returns:
        dict: Dictionary with hierarchy information including:
            - current_section: Current section title if found
            - parent_chapter: Parent chapter if identifiable
            - page_number: Page number if available in chunk headers
            - level: Document hierarchy level (1-5)
    """
    from src.utils.text import is_heading

    hierarchy = {
        "current_section": None,
        "parent_chapter": None,
        "page_number": None,
        "level": 0,
    }

    lines = chunk.split("\n")[:10]  # Check first 10 lines for structure info

    # Look for page information in document headers
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Extract page number from headers like "SEITEN: 15-18" or "Seite 23"
        page_match = re.search(
            r"SEITEN?:\s*(\d+(?:-\d+)?)|Seite\s+(\d+)", line, re.IGNORECASE
        )
        if page_match:
            hierarchy["page_number"] = page_match.group(1) or page_match.group(2)

        # Extract section information from ABSCHNITT headers
        section_match = re.search(r"ABSCHNITT:\s*(.+)", line, re.IGNORECASE)
        if section_match:
            hierarchy["current_section"] = section_match.group(1).strip()
            continue

        # Check if line is a heading and extract hierarchy information
        if is_heading(line):
            # Determine hierarchy level based on numbering pattern
            if re.match(r"^\d+\.?\s+", line):  # "1. Hauptkapitel"
                hierarchy["level"] = 1
                hierarchy["parent_chapter"] = line.strip()
            elif re.match(r"^\d+\.\d+\.?\s+", line):  # "1.1 Unterkapitel"
                hierarchy["level"] = 2
                hierarchy["current_section"] = line.strip()
                # Extract parent from numbering
                parent_match = re.match(r"^(\d+)\.", line)
                if parent_match:
                    hierarchy["parent_chapter"] = f"Kapitel {parent_match.group(1)}"
            elif re.match(r"^\d+\.\d+\.\d+\.?\s+", line):  # "1.1.1 Sub-Unterkapitel"
                hierarchy["level"] = 3
                hierarchy["current_section"] = line.strip()
            elif "handlungsfeld" in line.lower():
                hierarchy["level"] = 2  # Handlungsfelder are typically level 2
                hierarchy["current_section"] = line.strip()
            else:
                # General heading
                hierarchy["level"] = max(1, hierarchy["level"])
                if not hierarchy["current_section"]:
                    hierarchy["current_section"] = line.strip()

    # If no specific section found, try to infer from content
    if not hierarchy["current_section"]:
        # Look for action field keywords in the content
        content_lines = chunk.split("\n")[5:15]  # Skip headers, check content
        for line in content_lines:
            line_lower = line.lower().strip()
            if any(
                keyword in line_lower
                for keyword in ["handlungsfeld", "themenfeld", "bereich"]
            ):
                # Extract potential action field name
                field_match = re.search(r'handlungsfeld[:\s]+"?([^".\n]+)', line_lower)
                if field_match:
                    hierarchy["current_section"] = field_match.group(1).strip().title()
                    hierarchy["level"] = 2
                    break

    # Clean up extracted information
    for key in ["current_section", "parent_chapter"]:
        if hierarchy[key]:
            # Remove excessive whitespace and truncate if too long
            hierarchy[key] = re.sub(r"\s+", " ", hierarchy[key])[:100]

    return hierarchy


def _refine_uncertain_classifications(
    chunks: list[str],
    action_field: str,
    project_title: str,
    low_conf_measures: list[tuple[str, float]],
    low_conf_indicators: list[tuple[str, float]],
    confidence_threshold: float,
) -> ProjectDetails | None:
    """
    Perform focused re-analysis of uncertain classifications with enhanced prompts.

    Args:
        chunks: Original text chunks
        action_field: The action field context
        project_title: The project being analyzed
        low_conf_measures: List of (measure, confidence) tuples below threshold
        low_conf_indicators: List of (indicator, confidence) tuples below threshold
        confidence_threshold: Minimum confidence required

    Returns:
        ProjectDetails with refined results that meet confidence threshold, or None
    """
    if not low_conf_measures and not low_conf_indicators:
        return None

    print(
        f"🔍 Refinement: Re-analyzing {len(low_conf_measures + low_conf_indicators)} uncertain items"
    )

    # Create focused prompt for uncertain items
    uncertain_items = [item for item, conf in low_conf_measures + low_conf_indicators]

    focused_system_message = _get_refinement_system_message(
        action_field, project_title, uncertain_items
    )

    refined_measures = {}
    refined_indicators = {}

    # Re-analyze only chunks that contain the uncertain items
    relevant_chunks = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        if any(item.lower()[:20] in chunk_lower for item in uncertain_items):
            relevant_chunks.append(chunk)

    print(f"   📄 Focusing on {len(relevant_chunks)} relevant chunks")

    for _i, chunk in enumerate(relevant_chunks):
        if not chunk.strip():
            continue

        # Extract hierarchy for focused context
        document_hierarchy = _extract_document_hierarchy(chunk)

        # Create focused refinement prompt
        refinement_prompt = _get_refinement_prompt(
            chunk, action_field, project_title, uncertain_items, document_hierarchy
        )

        # Use "contextual" thinking mode for maximum analysis depth
        result = query_ollama_with_thinking_mode(
            prompt=refinement_prompt,
            response_model=ProjectDetailsEnhanced,
            thinking_mode="contextual",  # Highest complexity mode
            system_message=focused_system_message,
            temperature=0.1,  # Lower temperature for more focused analysis
        )

        if result:
            # Only accept items that now meet the confidence threshold
            for measure in result.measures:
                confidence = result.confidence_scores.get(measure, 0.5)
                if confidence >= confidence_threshold:
                    refined_measures[measure] = confidence

            for indicator in result.indicators:
                confidence = result.confidence_scores.get(indicator, 0.5)
                if confidence >= confidence_threshold:
                    refined_indicators[indicator] = confidence

    # Return refined results
    if refined_measures or refined_indicators:
        return ProjectDetails(
            measures=sorted(refined_measures.keys()),
            indicators=sorted(refined_indicators.keys()),
        )

    print("   ❌ Refinement: No items reached confidence threshold")
    return None


def _get_refinement_system_message(
    action_field: str, project_title: str, uncertain_items: list[str]
) -> str:
    """Generate focused system message for refinement analysis."""

    items_list = "\n".join(
        f"• {item}" for item in uncertain_items[:10]
    )  # Limit to first 10

    return f"""Sie sind ein Spezialist für deutsche Kommunalverwaltung mit Fokus auf ZWEIFELHAFTE KLASSIFIKATIONEN.

IHRE AUFGABE: Re-analysieren Sie folgende unsichere Punkte für "{project_title}" im "{action_field}":

{items_list}

ERWEITERTE ANALYSEMETHODEN:

1. ADMINISTRATIVE PRÄZEDENZ - Vergleichen Sie mit typischen Verwaltungsdokumenten
2. ZEITLICHE DIMENSION - Sind Zeitangaben vorhanden? → Meist INDIKATOR
3. QUANTITATIVE DIMENSION - Sind Zahlen/Prozente vorhanden? → Meist INDIKATOR
4. HANDLUNGSDIMENSION - Wird eine konkrete Aktion beschrieben? → Meist MASSNAHME
5. MESSUNGSDIMENSION - Wird ein Zielwert definiert? → Meist INDIKATOR

STRENGE KONFIDENZKRITERIEN:
• Nur klassifizieren wenn EINDEUTIG zuordenbar (Konfidenz ≥ 0.8)
• Bei Unsicherheit: Konfidenz < 0.8 setzen
• Explizite Begründung für jede Entscheidung

FOKUS: Qualität vor Quantität - lieber weniger, aber sichere Ergebnisse."""


def _get_refinement_prompt(
    chunk: str,
    action_field: str,
    project_title: str,
    uncertain_items: list[str],
    document_hierarchy: dict,
) -> str:
    """Generate focused refinement prompt for uncertain classifications."""

    hierarchy_context = ""
    if document_hierarchy and any(document_hierarchy.values()):
        hierarchy_context = f"""
VERSTÄRKTER STRUKTUR-KONTEXT:
• Abschnitt: {document_hierarchy.get('current_section', 'Unbekannt')}
• Kapitel: {document_hierarchy.get('parent_chapter', 'Unbekannt')}
• Seite: {document_hierarchy.get('page_number', 'Unbekannt')}
• Ebene: {document_hierarchy.get('level', 'Unbekannt')}
"""

    items_focus = "\n".join(f"→ {item}" for item in uncertain_items[:8])

    return f"""FOKUSSIERTE NACHANALYSE für "{project_title}":

{hierarchy_context}

UNSICHERE KLASSIFIKATIONEN (erneut prüfen):
{items_focus}

ERWEITERTE DEUTSCHE MUSTER-ANALYSE:

EINDEUTIGE MASSNAHMEN-SIGNALE:
✓ Aktionsverben: "errichten", "schaffen", "entwickeln", "einführen", "umsetzen"
✓ Planungsbegriffe: "Konzept erstellen", "Strategie entwickeln", "Leitfaden erarbeiten"
✓ Amtsdeutsch: "Bereitstellung von", "Durchführung von", "Realisierung von"

EINDEUTIGE INDIKATOR-SIGNALE:
✓ Zahlenwerte: "15 Stationen", "30%", "bis 2030", "€ 2 Mio"
✓ Messgrößen: "Anzahl", "Anteil", "Reduktion um", "Steigerung auf"
✓ Erfolgskennzahlen: "Zielwert", "Kennzahl", "erreichen von"

MEHRDEUTIGE FÄLLE - Entscheidungslogik:
• "Förderung erneuerbarer Energien" → MASSNAHME (allgemeine Handlung)
• "Steigerung auf 50% erneuerbare Energien" → INDIKATOR (messbares Ziel)
• "Verbesserung der Luftqualität" → MASSNAHME (ohne Zahlen)
• "PM10-Werte unter 40 μg/m³" → INDIKATOR (konkreter Messwert)

QUELLDOKUMENT:
========
{chunk.strip()}
========

STRENGE BEWERTUNG: Klassifizieren Sie NUR, wenn eindeutig zuordenbar (Konfidenz ≥ 0.8).
Bei geringster Unsicherheit: Konfidenz < 0.8 setzen."""
