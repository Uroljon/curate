import json
from llm import query_ollama
from embedder import query_chunks


def build_structure_prompt(chunk_text: str) -> str:
    return f"""
The following is a section of a German municipality's long-term strategic development plan.

Please identify any of the following if present:
- Action Field (e.g., "Klimaschutz")
- Projects within that field (titles)
- Measures under each project (if available)
- Indicators used to measure progress (if available)

Return a JSON like this (or empty if not applicable):

{{
  "action_field": "Klimaschutz",
  "projects": [
    {{
      "title": "Erneuerbare Energie Ausbau",
      "measures": ["Machbarkeitsstudie", "Stakeholder Konsultation"],
      "indicators": ["MWh produziert", "CO2-Reduktion (%)"]
    }}
  ]
}}

Text:
\"\"\"
{chunk_text.strip()}
\"\"\"
"""
