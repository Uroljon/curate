import re
from typing import List


def is_heading(line: str) -> bool:
    """Heuristic: check if a line looks like a section heading."""
    line = line.strip()
    if len(line) > 100 or len(line) < 3:
        return False
    if re.match(r"^\d{1,2}(\.\d{1,2})*\s+[A-ZÄÖÜ]", line):
        return True  # e.g. "2.1 Maßnahmen"
    if line.isupper():
        return True
    if re.match(r"^[A-ZÄÖÜ][a-zäöüß]+(\s+[A-ZÄÖÜ][a-zäöüß]+)*$", line):
        return True  # Title Case
    return False


def split_by_heading(text: str) -> List[str]:
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


def merge_short_chunks(chunks: List[str], min_words=100, max_words=300) -> List[str]:
    """Merge small or overly long chunks to be LLM-friendly."""
    merged = []
    buffer = ""

    def word_count(text):
        return len(text.split())

    for chunk in chunks:
        if not chunk.strip():
            continue

        combined = buffer + "\n\n" + chunk if buffer else chunk
        if word_count(combined) < max_words:
            buffer = combined
        else:
            if buffer:
                merged.append(buffer.strip())
            buffer = chunk

    if buffer.strip():
        merged.append(buffer.strip())

    return merged


def smart_chunk(cleaned_text: str, max_words: int = 300) -> List[str]:
    chunks = split_by_heading(cleaned_text)
    final_chunks = merge_short_chunks(chunks, max_words=max_words)
    return final_chunks
