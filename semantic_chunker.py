import re
from typing import List


def is_heading(line: str) -> bool:
    """Heuristic: check if a line looks like a section heading."""
    line = line.strip()
    if len(line) > 100 or len(line) < 3:
        return False
    if re.match(r"^\d{1,2}(\.\d{1,2})*\.?\s+\w", line):
        return True  # e.g. "1. Klimaschutz" or "2.1 Maßnahmen"
    if line.isupper():
        return True
    if re.match(r"^[A-ZÄÖÜ][a-zäöüß]+(\s+[A-ZÄÖÜ][a-zäöüß]+)*$", line):
        return True  # Title Case
    return False


def split_by_heading(text: str) -> list[str]:
    """Split text by section headings, preserving OCR tags."""
    lines = text.splitlines()
    chunks = []
    current = []

    for line in lines:
        if re.match(r"\[OCR Page \d+\]", line):
            if current:
                chunks.append("\n".join(current).strip())
                current = []
            current.append(line)
            continue

        if is_heading(line) and current:
            chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

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
