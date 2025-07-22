import json
import re
from typing import Any, Dict, List, Optional

import json5

from embedder import query_chunks
from llm import query_ollama, query_ollama_structured
from schemas import ActionField, ExtractionResult, Project, ActionFieldList, ProjectList, ProjectDetails
from config import CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, CHUNK_WARNING_THRESHOLD, EXTRACTION_MAX_RETRIES, MODEL_TEMPERATURE


def prepare_llm_chunks(
    chunks: list[str], max_chars: int = CHUNK_MAX_CHARS, min_chars: int = CHUNK_MIN_CHARS
) -> list[str]:
    """
    Merge small chunks and split large ones to optimize for LLM context size.
    Operates on character count, but keeps paragraph integrity where possible.
    Enforces hard limits to prevent oversized chunks.
    """
    merged_chunks = []
    current_chunk = []
    current_len = 0

    def split_large_text(text: str, max_size: int) -> list[str]:
        """Recursively split text that exceeds max_size."""
        if len(text) <= max_size:
            return [text]
        
        # Try to split by paragraphs first
        paragraphs = text.split("\n\n")
        if len(paragraphs) > 1:
            result = []
            current = []
            current_size = 0
            
            for para in paragraphs:
                para_size = len(para) + (2 if current else 0)  # Account for \n\n
                
                if para_size > max_size:
                    # Single paragraph too large, need to split it
                    if current:
                        result.append("\n\n".join(current))
                    # Split by sentences or hard split
                    result.extend(split_large_text(para, max_size))
                    current = []
                    current_size = 0
                elif current_size + para_size > max_size:
                    result.append("\n\n".join(current))
                    current = [para]
                    current_size = len(para)
                else:
                    current.append(para)
                    current_size += para_size
                    
            if current:
                result.append("\n\n".join(current))
            return result
        
        # If no paragraphs or single paragraph, try sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 1:
            result = []
            current = []
            current_size = 0
            
            for sent in sentences:
                sent_size = len(sent) + (1 if current else 0)  # Account for space
                
                if sent_size > max_size:
                    # Single sentence too large, hard split
                    if current:
                        result.append(" ".join(current))
                    # Hard split the sentence
                    for i in range(0, len(sent), max_size):
                        result.append(sent[i:i+max_size])
                    current = []
                    current_size = 0
                elif current_size + sent_size > max_size:
                    result.append(" ".join(current))
                    current = [sent]
                    current_size = len(sent)
                else:
                    current.append(sent)
                    current_size += sent_size
                    
            if current:
                result.append(" ".join(current))
            return result
        
        # Last resort: hard split
        return [text[i:i+max_size] for i in range(0, len(text), max_size)]

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # If single chunk exceeds max, split it
        if len(chunk) > max_chars:
            split_chunks = split_large_text(chunk, max_chars)
            for split_chunk in split_chunks:
                if current_chunk and current_len + len(split_chunk) + 2 > max_chars:
                    merged_chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                
                if len(split_chunk) <= max_chars:
                    current_chunk.append(split_chunk)
                    current_len += len(split_chunk) + (2 if current_len > 0 else 0)
                else:
                    # This should not happen with our split logic, but safety check
                    print(f"⚠️ Warning: Chunk still too large after splitting: {len(split_chunk)} chars")
                    merged_chunks.append(split_chunk[:max_chars])
            continue

        # Normal merging logic
        chunk_size = len(chunk) + (2 if current_chunk else 0)  # Account for \n\n separator
        
        if current_len + chunk_size > max_chars:
            if current_len >= min_chars:
                merged_chunks.append("\n\n".join(current_chunk))
                current_chunk = [chunk]
                current_len = len(chunk)
            else:
                # Try to reach min_chars but never exceed max_chars
                current_chunk.append(chunk)
                current_len += chunk_size
                if current_len >= min_chars:
                    merged_chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
        else:
            current_chunk.append(chunk)
            current_len += chunk_size

    if current_chunk:
        merged_chunks.append("\n\n".join(current_chunk))

    # Final validation
    for i, chunk in enumerate(merged_chunks):
        if len(chunk) > max_chars:
            print(f"⚠️ Chunk {i+1} exceeds max_chars ({len(chunk)} > {max_chars}), force splitting...")
            # Force split any remaining oversized chunks
            split_chunks = split_large_text(chunk, max_chars)
            merged_chunks[i:i+1] = split_chunks

    return merged_chunks


def extract_action_fields_only(chunks: List[str]) -> List[str]:
    """
    Stage 1: Extract just action field names from all chunks.
    
    This is a lightweight extraction that only identifies the main categories
    (Handlungsfelder) without extracting projects or details.
    """
    all_action_fields = set()  # Use set to automatically deduplicate
    
    system_message = """Extract ONLY the names of action fields (Handlungsfelder) from German municipal strategy documents.

IMPORTANT: 
- Return ONLY the category names, no projects or details
- Look for main thematic areas like: Klimaschutz, Mobilität, Stadtentwicklung, Digitalisierung, etc.
- Do NOT include project names, measures, or indicators
- Do NOT include numbered items or bullet points that are sub-items

Examples of action fields:
- Klimaschutz
- Mobilität
- Stadtentwicklung
- Digitalisierung
- Wirtschaft und Wissenschaft
- Soziales und Gesellschaft
- Umwelt und Natur
- Energie
- Wohnen

Return a simple list of action field names found in the text."""

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        print(f"🔍 Stage 1: Scanning chunk {i+1}/{len(chunks)} for action fields ({len(chunk)} chars)")
        
        prompt = f"""Find all action fields (Handlungsfelder) in this German municipal text:

{chunk.strip()}

Return ONLY the main category names, not projects or details."""

        result = query_ollama_structured(
            prompt=prompt,
            response_model=ActionFieldList,
            system_message=system_message,
            temperature=MODEL_TEMPERATURE
        )
        
        if result and result.action_fields:
            found_fields = set(result.action_fields)
            print(f"   ✓ Found {len(found_fields)} action fields: {', '.join(sorted(found_fields))}")
            all_action_fields.update(found_fields)
        else:
            print(f"   ✗ No action fields found in chunk {i+1}")
    
    # Convert to sorted list and merge similar fields
    merged_fields = merge_similar_action_fields(list(all_action_fields))
    
    print(f"\n📊 Stage 1 Complete: Found {len(merged_fields)} unique action fields")
    for field in merged_fields:
        print(f"   • {field}")
    
    return merged_fields


def merge_similar_action_fields(fields: List[str]) -> List[str]:
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
    merged = []
    
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


def extract_projects_for_field(chunks: List[str], action_field: str) -> List[str]:
    """
    Stage 2: Extract project names for a specific action field.
    
    Given an action field (e.g., "Klimaschutz"), find all projects
    that belong to this category across all chunks.
    """
    all_projects = set()
    
    system_message = f"""Extract ONLY project names that belong to the action field "{action_field}".

IMPORTANT:
- Return ONLY project titles/names, not measures or descriptions
- Projects are specific initiatives, programs, or named efforts
- Do NOT include general statements or goals
- Do NOT include measures, actions, or indicators

Examples of projects:
- "Stadtbahn Regensburg"
- "Zero Waste Initiative"
- "Masterplan Biodiversität"
- "Energieeffizienz in kommunalen Gebäuden"
- "Digitales Rathaus"

Look for projects specifically related to: {action_field}"""

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        # Quick check if this chunk might contain relevant content
        if action_field.lower() not in chunk.lower():
            continue
            
        print(f"🔎 Stage 2: Searching chunk {i+1}/{len(chunks)} for {action_field} projects")
        
        prompt = f"""Find all projects related to the action field "{action_field}" in this text:

{chunk.strip()}

Return ONLY the project names that belong to {action_field}."""

        result = query_ollama_structured(
            prompt=prompt,
            response_model=ProjectList,
            system_message=system_message,
            temperature=MODEL_TEMPERATURE
        )
        
        if result and result.projects:
            found_projects = set(result.projects)
            print(f"   ✓ Found {len(found_projects)} projects")
            all_projects.update(found_projects)
    
    # Remove duplicates and sort
    unique_projects = sorted(list(all_projects))
    
    print(f"   📋 Total {len(unique_projects)} projects for {action_field}")
    
    return unique_projects


def extract_project_details(chunks: List[str], action_field: str, project_title: str) -> ProjectDetails:
    """
    Stage 3: Extract measures and indicators for a specific project.
    
    This is the most focused extraction, looking for specific details
    about a single project within an action field.
    """
    all_measures = set()
    all_indicators = set()
    
    system_message = f"""Extract measures and indicators for the project "{project_title}" in the action field "{action_field}".

DEFINITIONS:
- Maßnahmen (measures): Concrete actions, steps, implementations
- Indikatoren (indicators): Numbers, percentages, dates, targets, KPIs

CRITICAL: Focus on finding INDICATORS - these are quantitative metrics!

INDICATOR PATTERNS:
- Percentages: "40% Reduktion", "um 30% reduzieren", "Anteil von 65%"
- Time targets: "bis 2030", "ab 2025", "innerhalb von 5 Jahren"
- Quantities: "500 Ladepunkte", "18 km Streckenlänge", "1000 Wohneinheiten"
- Frequencies: "jährlich", "pro Jahr", "monatlich"
- Comparisons: "Verdopplung", "Halbierung", "30% weniger"

Example for a project "Stadtbahn":
Measures:
- "Planung der Trassenführung"
- "Bau der Haltestellen"
- "Integration in bestehenden ÖPNV"

Indicators:
- "18 km Streckenlänge"
- "24 Haltestellen"
- "Inbetriebnahme bis 2028"
- "Modal Split Erhöhung um 15%"

Look ONLY for information about: {project_title}"""

    # Search in chunks that mention both the action field and project
    relevant_chunks = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        chunk_lower = chunk.lower()
        # Check if chunk contains project name (fuzzy match)
        project_words = project_title.lower().split()
        if any(word in chunk_lower for word in project_words if len(word) > 3):
            relevant_chunks.append((i, chunk))
    
    if not relevant_chunks:
        print(f"   ⚠️ No chunks found mentioning project: {project_title}")
        return ProjectDetails()
    
    print(f"🔬 Stage 3: Analyzing {len(relevant_chunks)} chunks for {project_title} details")
    
    for i, chunk in relevant_chunks:
        prompt = f"""Extract measures and indicators for the project "{project_title}" from this text:

{chunk.strip()}

Focus on finding:
1. Maßnahmen (measures) - what will be done
2. Indikatoren (indicators) - numbers, targets, dates, percentages

Return ONLY information directly related to {project_title}."""

        result = query_ollama_structured(
            prompt=prompt,
            response_model=ProjectDetails,
            system_message=system_message,
            temperature=MODEL_TEMPERATURE
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
        measures=sorted(list(all_measures)),
        indicators=sorted(list(all_indicators))
    )
    
    print(f"   📊 Total: {len(details.measures)} measures, {len(details.indicators)} indicators")
    
    return details


def build_structure_prompt(chunk_text: str) -> str:
    return f"""Extract from this German text and output JSON:

{chunk_text.strip()}

Output the JSON array now:
["""


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

        # Validate project structure
        for project in item["projects"]:
            if not isinstance(project, dict):
                return False

            if "title" not in project:
                return False

            if not isinstance(project["title"], str):
                return False

            # measures and indicators are optional but must be lists if present
            if "measures" in project and not isinstance(project["measures"], list):
                return False

            if "indicators" in project and not isinstance(project["indicators"], list):
                return False

    return True


def extract_json_from_response(response: str) -> list[dict[str, Any]] | None:
    """
    Extract and validate JSON from LLM response with aggressive cleaning.
    """
    # Clean the response first - remove any conversational text
    cleaned = response.strip()

    # Remove common conversational prefixes/suffixes
    prefixes_to_remove = [
        "Here",
        "Hier",
        "This",
        "Das",
        "Es",
        "The",
        "I'll",
        "Let",
        "Based",
        "Looking",
        "What",
        "Wie",
        "Municipal",
        "German",
    ]

    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            # Find the first [ or { after the prefix
            start_idx = max(cleaned.find("["), cleaned.find("{"))
            if start_idx > 0:
                cleaned = cleaned[start_idx:]
                break

    # Method 1: Direct JSON array extraction (most aggressive)
    array_patterns = [
        r"\[[\s\S]*?\]",  # Complete array
        r"\{[\s\S]*?\}",  # Single object (will wrap)
    ]

    for pattern in array_patterns:
        matches = re.findall(pattern, cleaned, re.DOTALL)
        for match in matches:
            try:
                data = json5.loads(match)
                if isinstance(data, dict):
                    data = [data]  # Wrap single object
                if validate_extraction_schema(data):
                    return data
            except:
                continue

    # Method 2: Extract everything between first [ and last ]
    first_bracket = cleaned.find("[")
    last_bracket = cleaned.rfind("]")
    if first_bracket >= 0 and last_bracket > first_bracket:
        try:
            json_str = cleaned[first_bracket : last_bracket + 1]
            data = json5.loads(json_str)
            if validate_extraction_schema(data):
                return data
        except:
            pass

    # Method 3: Try to find object and wrap
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        try:
            json_str = cleaned[first_brace : last_brace + 1]
            data = json5.loads(json_str)
            if isinstance(data, dict):
                data = [data]
            if validate_extraction_schema(data):
                return data
        except:
            pass

    return None


def extract_structures_with_retry(
    chunk_text: str, max_retries: int = EXTRACTION_MAX_RETRIES
) -> list[dict[str, Any]]:
    """
    Extract structures from text using Ollama structured output.
    """
    system_message = """Extract German municipal action fields (Handlungsfelder) and their projects from the text.

CRITICAL: You MUST actively search for and extract indicators/KPIs. These are NUMBERS, PERCENTAGES, DATES, and TARGETS.

Definitions:
- Maßnahmen (measures): Concrete actions, steps, implementations (verbs like "errichten", "ausbauen", "fördern")
- Indikatoren (indicators): QUANTITATIVE metrics, targets, deadlines, percentages, numbers

INDICATOR PATTERNS TO FIND:
- Percentages: "40% Reduktion", "um 30% reduzieren", "Anteil von 65%", "Quote von 25%"
- Time targets: "bis 2030", "ab 2025", "innerhalb von 5 Jahren", "jährlich", "pro Jahr"
- Quantities: "500 Ladepunkte", "18 km Streckenlänge", "1000 Wohneinheiten", "50 Hektar"
- Reductions: "Reduzierung um 55%", "Senkung auf 20%", "Verringerung der Emissionen um 40%"
- Increases: "Steigerung auf 70%", "Erhöhung um 25%", "Verdopplung bis 2030"
- Ratios: "Modal Split 70/30", "Versiegelungsgrad max. 30%", "Grünflächenanteil min. 40%"

Example with FULL indicator extraction:
{
  "action_fields": [{
    "action_field": "Klimaschutz",
    "projects": [{
      "title": "CO2-Neutralität",
      "measures": ["Umstellung auf erneuerbare Energien", "Gebäudesanierung", "Ausbau Fernwärme"],
      "indicators": ["CO2-Reduktion 55% bis 2030", "100% Ökostrom bis 2035", "Sanierungsquote 3% pro Jahr"]
    }]
  }, {
    "action_field": "Mobilität",
    "projects": [{
      "title": "Verkehrswende",
      "measures": ["Ausbau Radwegenetz", "Einführung Stadtbahn", "Park&Ride-Anlagen"],
      "indicators": ["Modal Split 70% Umweltverbund", "500 neue Fahrradstellplätze jährlich", "30% weniger PKW-Verkehr bis 2030"]
    }]
  }]
}

REMEMBER: If you see ANY number, percentage, date target, or quantitative goal - it's an indicator!"""

    prompt = f"""Extract all action fields and their projects from this German municipal strategy text:

{chunk_text.strip()}

Extract action fields with their projects, measures, and indicators."""

    # Validate chunk size
    if len(chunk_text) > CHUNK_WARNING_THRESHOLD:
        print(f"⚠️ WARNING: Chunk size ({len(chunk_text)} chars) exceeds recommended limit of {CHUNK_WARNING_THRESHOLD} chars!")
        print(f"   This may cause JSON parsing issues or incomplete responses.")
    
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
                action_field_dict = {"action_field": af.action_field, "projects": []}
                for project in af.projects:
                    project_dict = {"title": project.title}
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

    # Fallback to old method as last resort
    print("🔄 Falling back to legacy extraction method...")
    prompt = build_structure_prompt(chunk_text)
    raw_response = query_ollama(prompt, system_message=system_message)
    extracted_data = extract_json_from_response(raw_response)

    if extracted_data:
        print(f"✅ Legacy method extracted {len(extracted_data)} action fields")
        return extracted_data

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

    system_message = """You are enhancing an existing extraction with new information from German municipal strategy documents.

CRITICAL RULES:
1. PRESERVE all existing data - never remove anything
2. ENHANCE existing projects with new measures/indicators if found
3. MERGE duplicate projects (same title = same project, combine their data)
4. ADD new action fields and projects not yet captured
5. ACTIVELY SEARCH for indicators that may have been missed

SPECIAL FOCUS ON INDICATORS:
Look for ANY quantitative information:
- Numbers with units: "500 Ladepunkte", "18 km", "1000 Wohneinheiten"
- Percentages: "40% Reduktion", "um 30% senken", "Anteil von 65%"
- Time targets: "bis 2030", "ab 2025", "innerhalb 5 Jahren"
- Frequencies: "jährlich", "pro Jahr", "monatlich"
- Comparisons: "Verdopplung", "Halbierung", "30% weniger"

When merging projects:
- Combine all unique measures
- Combine all unique indicators
- Look for indicators that relate to existing projects but weren't extracted before

Example of enhanced extraction with found indicators:
Existing: {"title": "Verkehrswende", "measures": ["Ausbau Radwegenetz"], "indicators": []}
New text mentions: "Das Radwegenetz soll bis 2030 auf 500 km ausgebaut werden mit jährlich 50 km Neubau"
Result: {"title": "Verkehrswende", "measures": ["Ausbau Radwegenetz"], "indicators": ["500 km Radwegenetz bis 2030", "50 km Neubau jährlich"]}"""

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
        result_dict = {"action_fields": []}
        for af in enhanced_result.action_fields:
            af_dict = {"action_field": af.action_field, "projects": []}
            for project in af.projects:
                proj_dict = {"title": project.title}
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
