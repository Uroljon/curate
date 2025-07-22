"""
Enhanced extraction prompts that are robust to mixed-topic chunks.
"""

# Stage 1: Find ALL action fields
STAGE1_SYSTEM_MESSAGE = """Extract ALL action fields (Handlungsfelder/Bereiche/Themenfelder/Schwerpunkte) from German municipal documents.

IMPORTANT: 
- This chunk may contain MULTIPLE action fields mixed together
- Return ALL category names found, even if they appear briefly
- Look for ANY thematic area, using various naming conventions:
- LANGUAGE: Always extract and return content in GERMAN as it appears in the source text
  * Handlungsfeld/Handlungsfelder
  * Bereich/Bereiche
  * Themenfeld/Themenfelder  
  * Schwerpunkt/Schwerpunkte
  * Sektor/Sektoren
  * Or any other clear thematic grouping

Common action fields include (but are not limited to):
- Klimaschutz / Klimawandel / Klimaanpassung
- Mobilität / Verkehr / ÖPNV
- Stadtentwicklung / Städtebau / Quartiersentwicklung
- Digitalisierung / Smart City / IT
- Wirtschaft / Wissenschaft / Innovation
- Soziales / Gesellschaft / Teilhabe
- Umwelt / Natur / Biodiversität
- Energie / Erneuerbare Energien
- Wohnen / Wohnungsbau
- Bildung / Kultur
- Gesundheit / Sport

Also look for implicit topic shifts even without explicit markers.
Return ALL distinct action fields found in the text."""


# Stage 2: Extract projects for a specific field
def get_stage2_system_message(action_field: str) -> str:
    return f"""Extract ALL projects that belong to the action field "{action_field}".

CRITICAL INSTRUCTIONS:
1. This chunk may discuss MULTIPLE action fields
2. Focus ONLY on projects related to "{action_field}"
3. IGNORE projects that belong to other action fields
4. A project may be mentioned in a mixed context - extract it if it relates to "{action_field}"
5. LANGUAGE: Always extract and return content in GERMAN as it appears in the source text

WHAT ARE PROJECTS:
- Named initiatives, programs, or specific efforts
- Has a clear title or designation
- Examples: "Stadtbahn Regensburg", "Masterplan Biodiversität", "Digitales Rathaus"

LOOK FOR PROJECTS EVEN IF:
- They appear in a section about multiple topics
- They are mentioned briefly or in passing
- The chunk primarily discusses another action field
- They appear before or after the explicit mention of "{action_field}"

DO NOT INCLUDE:
- General goals or statements
- Individual measures without a project name
- Indicators or metrics
- Projects that clearly belong to OTHER action fields"""


# Stage 3: Extract details for a specific project
def get_stage3_system_message(action_field: str, project_title: str) -> str:
    return f"""Extract measures and indicators for the project "{project_title}" in action field "{action_field}".

CRITICAL CONTEXT AWARENESS:
1. This chunk may contain information about MULTIPLE projects and action fields
2. Focus ONLY on measures/indicators related to "{project_title}"
3. Indicators may appear ANYWHERE in the chunk - before, after, or separated from the project mention
4. Look for cross-references: "wie oben erwähnt", "siehe auch", "vgl."
5. LANGUAGE: Always extract and return content in GERMAN as it appears in the source text - do NOT translate

DEFINITIONS:
- Maßnahmen (measures): Concrete actions, steps, implementations for THIS project
- Indikatoren (indicators): Quantitative metrics, targets, KPIs for THIS project

INDICATOR PATTERNS TO FIND:
- Percentages: "55% Reduktion", "Anteil von 70%", "um 30% steigern"
- Absolute numbers: "500 Ladepunkte", "18 km", "1000 Wohneinheiten"
- Time targets: "bis 2030", "ab 2025", "innerhalb von 5 Jahren", "jährlich"
- Comparisons: "Verdopplung", "Halbierung", "30% weniger", "Steigerung um"
- Rates: "pro Jahr", "je Einwohner", "täglich", "€/m²"
- Compound metrics: "3 von 5", "mindestens 50%"

SPECIAL INSTRUCTIONS:
- If an indicator appears without explicit project attribution but contextually relates to "{project_title}", include it
- If measures are listed generally but apply to "{project_title}", include them
- Look for indicators that might be in summary sections, tables, or overview paragraphs
- Include partial information - better to capture incomplete data than miss it entirely"""


# Enhanced prompts for each stage
def get_stage1_prompt(chunk: str) -> str:
    return f"""Analyze this text from a German municipal strategy document and find ALL action fields/thematic areas:

{chunk.strip()}

Remember:
- This chunk may contain multiple topics mixed together
- Look for both explicit markers and implicit topic changes
- Return ALL distinct action fields, even if mentioned briefly
- Include variations like Handlungsfeld, Bereich, Themenfeld, etc."""


def get_stage2_prompt(chunk: str, action_field: str) -> str:
    return f"""Find ALL projects specifically related to "{action_field}" in this text:

{chunk.strip()}

IMPORTANT REMINDERS:
- This chunk may discuss multiple action fields
- Only extract projects that belong to "{action_field}"
- Projects may be scattered throughout the text
- Look for project names even in mixed contexts
- Return ONLY project names/titles for "{action_field}" """


def get_stage3_prompt(chunk: str, action_field: str, project_title: str) -> str:
    return f"""Extract measures and indicators for the project "{project_title}" (action field: {action_field}):

{chunk.strip()}

REMEMBER:
- Focus ONLY on "{project_title}" - ignore other projects
- Indicators may appear anywhere in the text
- Look for quantitative metrics (%, numbers, dates, targets)
- Include measures that specifically implement "{project_title}"
- This chunk may contain mixed topics - filter carefully"""