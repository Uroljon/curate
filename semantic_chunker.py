# Take full PDF text (already extracted by parser.py)
# Split it intelligently into semantic chunks (like chapters/sections) using line spacing, heading patterns, and fuzzy keyword detection.

# semantic_chunker.py

import re
from typing import List

def clean_ocr_text(text: str) -> str:
    """Preserve metadata like [OCR Page X], but clean other OCR junk."""
    # Remove isolated page numbers (but keep tags like [OCR Page 3])
    text = re.sub(r'\n\d+\n', '\n', text)
    return text

def paragraph_split(text: str) -> List[str]:
    """Split raw text into paragraphs using double newlines"""
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paras

def rolling_chunks(paragraphs: List[str], max_words=300) -> List[str]:
    """Group paragraphs into ~max_words chunks. Includes [OCR Page X]."""
    chunks = []
    current = []

    for p in paragraphs:
        if re.match(r"\[OCR Page \d+\]", p):
            # If page marker, flush current chunk and start new
            if current:
                chunks.append("\n\n".join(current))
                current = []
            current.append(p)
            continue

        words = p.split()
        current_word_count = sum(len(c.split()) for c in current)

        if current_word_count + len(words) > max_words and current:
            chunks.append("\n\n".join(current))
            current = [p]
        else:
            current.append(p)

    if current:
        chunks.append("\n\n".join(current))

    return chunks

def smart_chunk(text: str, max_words: int = 300) -> List[str]:
    text = clean_ocr_text(text)
    paragraphs = paragraph_split(text)
    chunks = rolling_chunks(paragraphs, max_words)
    return chunks
