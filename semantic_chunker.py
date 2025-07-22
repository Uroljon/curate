import re
from typing import List


def is_heading(line: str) -> bool:
    """Heuristic: check if a line looks like a section heading.
    
    Enhanced for German municipal documents with common heading patterns.
    """
    line = line.strip()
    if len(line) > 100 or len(line) < 3:
        return False
    
    # Numbered sections (e.g. "1. Klimaschutz", "2.1 Maßnahmen", "III. Ziele")
    if re.match(r"^\d{1,2}(\.\d{1,2})*\.?\s+\w", line):
        return True
    if re.match(r"^[IVX]+\.\s+\w", line):  # Roman numerals
        return True
    if re.match(r"^[a-z]\)\s+\w", line):  # Letter enumeration "a) Ziel"
        return True
    
    # German document structure keywords
    german_structure_keywords = [
        "Kapitel", "Abschnitt", "Teil", "Anlage", "Anhang",
        "Maßnahmen:", "Projekte:", "Ziele:", "Indikatoren:",
        "Handlungsfeld:", "Handlungsfelder:", "Zielstellung:",
        "Ausgangslage:", "Umsetzung:", "Zeitplan:", "Monitoring:"
    ]
    for keyword in german_structure_keywords:
        if line.startswith(keyword):
            return True
    
    # All uppercase (common for main sections)
    if line.isupper() and len(line.split()) <= 8 and len(line) > 3:
        return True
    
    # Title Case (allowing German characters and common prepositions)
    # Allow lowercase words like "für", "und", "der", etc. in the middle
    if re.match(r"^[A-ZÄÖÜ][a-zäöüß]+(\s+([A-ZÄÖÜ\-][a-zäöüß]+|für|und|der|die|das|von|zu|mit|in|auf))*$", line):
        # Extra check: should be relatively short and not a regular sentence
        word_count = len(line.split())
        # Must have at least 2 words or be longer than 10 chars for single words
        if (word_count >= 2 or len(line) > 10) and word_count <= 6 and not line.endswith(('.', ',', ';')):
            return True
    
    # Lines ending with colon (often introduce sections)
    if line.endswith(':') and 3 < len(line) < 50:
        return True
    
    return False


def split_by_heading(text: str) -> list[str]:
    """Split text by section headings, preserving OCR tags.
    
    Handles multi-line German headings by checking if consecutive lines
    form a heading pattern together.
    """
    lines = text.splitlines()
    chunks = []
    current = []
    i = 0

    while i < len(lines):
        line = lines[i]
        
        # Handle OCR page markers
        if re.match(r"\[OCR Page \d+\]", line):
            if current:
                chunks.append("\n".join(current).strip())
                current = []
            current.append(line)
            i += 1
            continue

        # Check for multi-line headings (common in German documents)
        is_multiline_heading = False
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Check if current + next line form a heading pattern
            # e.g., "Handlungsfeld 1:" on one line, "Klimaschutz" on next
            if (line.strip().endswith(':') and len(line.strip()) < 30 and 
                next_line and len(next_line) < 50 and 
                next_line and not (next_line[0].islower() if next_line else False)):
                is_multiline_heading = True
            # Check for numbered heading continuing on next line
            elif (re.match(r"^\d{1,2}(\.\d{1,2})*\.?\s*$", line.strip()) and
                  next_line and not (next_line[0].islower() if next_line else False)):
                is_multiline_heading = True

        if is_heading(line) and current:
            chunks.append("\n".join(current).strip())
            current = [line]
            if is_multiline_heading:
                i += 1
                current.append(lines[i])
        elif is_multiline_heading and current:
            chunks.append("\n".join(current).strip())
            current = [line, lines[i + 1]]
            i += 1
        else:
            current.append(line)
        
        i += 1

    if current:
        chunks.append("\n".join(current).strip())

    return chunks


def merge_short_chunks(chunks: list[str], min_chars=3000, max_chars=5000) -> list[str]:
    """Merge small chunks to optimize for embedding and LLM processing."""
    merged = []
    buffer = ""

    def char_count(text):
        return len(text)

    for chunk in chunks:
        if not chunk.strip():
            continue

        combined = buffer + "\n\n" + chunk if buffer else chunk
        if char_count(combined) < max_chars:
            buffer = combined
        else:
            if buffer:
                merged.append(buffer.strip())
            buffer = chunk

    if buffer.strip():
        merged.append(buffer.strip())

    return merged


def smart_chunk(cleaned_text: str, max_chars: int = 5000) -> list[str]:
    chunks = split_by_heading(cleaned_text)
    final_chunks = merge_short_chunks(chunks, max_chars=max_chars)
    return final_chunks
