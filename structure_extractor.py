import json
import json5
import re
from typing import List, Dict, Any, Optional
from llm import query_ollama, query_ollama_structured
from embedder import query_chunks
from schemas import ActionField, ExtractionResult, Project


def prepare_llm_chunks(
    chunks: list[str], max_chars: int = 12000, min_chars: int = 8000
) -> list[str]:
    """
    Merge small chunks and split large ones to optimize for LLM context size.
    Operates on character count, but keeps paragraph integrity.
    """
    merged_chunks = []
    current_chunk = []
    current_len = 0

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if len(chunk) > max_chars:
            # Split this single chunk by paragraph if it's too big
            paragraphs = chunk.split("\n\n")
            paragraph_chunk = []
            paragraph_len = 0
            for para in paragraphs:
                para_len = len(para)
                if paragraph_len + para_len > max_chars:
                    if paragraph_chunk:
                        merged_chunks.append("\n\n".join(paragraph_chunk))
                    paragraph_chunk = [para]
                    paragraph_len = para_len
                else:
                    paragraph_chunk.append(para)
                    paragraph_len += para_len
            if paragraph_chunk:
                merged_chunks.append("\n\n".join(paragraph_chunk))
            continue

        if current_len + len(chunk) > max_chars:
            if current_len >= min_chars:
                merged_chunks.append("\n\n".join(current_chunk))
                current_chunk = [chunk]
                current_len = len(chunk)
            else:
                current_chunk.append(chunk)
                current_len += len(chunk)
        else:
            current_chunk.append(chunk)
            current_len += len(chunk)

    if current_chunk:
        merged_chunks.append("\n\n".join(current_chunk))

    return merged_chunks


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


def extract_json_from_response(response: str) -> Optional[List[Dict[str, Any]]]:
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
    chunk_text: str, max_retries: int = 1
) -> List[Dict[str, Any]]:
    """
    Extract structures from text using Ollama structured output.
    """
    system_message = """Extract German municipal action fields (Handlungsfelder) and their projects from the text.

IMPORTANT: For each project, actively look for:
- Maßnahmen (measures/actions): Specific steps, actions, or implementations mentioned
- Indikatoren (indicators/KPIs): Percentages, targets, deadlines, metrics mentioned

Example of correct extraction:
{
  "action_fields": [{
    "action_field": "Mobilität",
    "projects": [{
      "title": "Stadtbahn Regensburg",
      "measures": ["Errichtung des Kernnetzes", "Ausbau zu einem Gesamtnetz", "Einführung bis 2030"],
      "indicators": ["Modal Split 70% bis 2040", "Reduzierung MIV um 30%", "18 km Streckenlänge"]
    }]
  }]
}

Extract ALL details found in the text. Look for:
- Action fields: Klimaschutz, Mobilität, Stadtentwicklung, Digitalisierung, Bildung, Ökologie, etc.
- Project titles: Named initiatives, programs, or projects
- Measures: Actions with verbs like "schaffen", "entwickeln", "fördern", "umsetzen", "ausbauen"
- Indicators: Numbers, percentages, dates, targets like "bis 2030", "20%", "reduzieren um"

Be comprehensive but only extract what's explicitly stated in the text."""

    prompt = f"""Extract all action fields and their projects from this German municipal strategy text:

{chunk_text.strip()}

Extract action fields with their projects, measures, and indicators."""

    for attempt in range(max_retries):
        print(
            f"📝 Extraction attempt {attempt + 1}/{max_retries} for chunk ({len(chunk_text)} chars)"
        )

        # Use structured output with Pydantic model
        result = query_ollama_structured(
            prompt=prompt,
            response_model=ExtractionResult,
            system_message=system_message,
            temperature=0.0,  # Zero temperature for deterministic extraction
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
                print(f"🔄 Retrying...")

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
    accumulated_data: Dict[str, Any],
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
) -> Dict[str, Any]:
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

    system_message = """You are enhancing an existing extraction with new information.

CRITICAL RULES:
1. PRESERVE all existing data - never remove anything
2. ENHANCE existing projects with new measures/indicators if found
3. MERGE duplicate projects (same title = same project, combine their data)
4. ADD new action fields and projects not yet captured
5. Maintain clean JSON structure

When you find a project that already exists:
- Add any new measures to the existing measures list
- Add any new indicators to the existing indicators list
- Don't duplicate measures/indicators that already exist

Example of merging:
Existing: {"title": "Stadtbahn", "measures": ["Bau Kernnetz"], "indicators": ["2030"]}
New info: "Stadtbahn soll auch Ausbaunetz bekommen"
Result: {"title": "Stadtbahn", "measures": ["Bau Kernnetz", "Ausbaunetz"], "indicators": ["2030"]}"""

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
        temperature=0.0,
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
        print(f"⚠️ Enhancement failed, keeping previous state")
        return accumulated_data
