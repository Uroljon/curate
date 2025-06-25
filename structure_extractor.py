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
Antworten Sie AUSSCHLIESSLICH mit einem **JSON-Array** im folgenden Format. Fügen Sie KEINE Erläuterungen oder zusätzlichen Texte hinzu.
Wenn keine relevanten Daten gefunden werden, geben Sie ein leeres Array zurück: `[]`
Erwartetes Format:
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
Im Folgenden finden Sie einen Auszug aus dem langfristigen strategischen Entwicklungsplan einer deutschen Gemeinde.
Bitte identifizieren Sie alle folgenden Elemente, sofern vorhanden:
- Aktionsfeld (z. B. „Klimaschutz”)
- Projekte innerhalb dieses Feldes (Titel)
- Maßnahmen im Rahmen jedes Projekts (sofern verfügbar)
- Indikatoren zur Messung des Fortschritts (sofern verfügbar)

Text:
\"\"\"
{chunk_text.strip()}
\"\"\"
"""
