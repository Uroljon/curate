import json
from llm import query_ollama
from embedder import query_chunks

def prepare_llm_chunks(chunks: list[str], max_chars: int = 30000, min_chars: int = 8000) -> list[str]:
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
    return f"""
Respond ONLY with a **JSON array** of the following format. Do NOT include any explanation or additional text.
If no relevant data is found, return an empty array: `[]`
The following is a section of a German municipality's long-term strategic development plan.

Please identify any of the following if present:
- Action Field (e.g., "Klimaschutz")
- Projects within that field (titles)
- Measures under each project (if available)
- Indicators used to measure progress (if available)

Expected format:
[
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
]

Text:
\"\"\"
{chunk_text.strip()}
\"\"\"
"""
