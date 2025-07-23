"""
Document chunking functionality for CURATE.

This module provides two main chunking strategies:
1. Semantic chunking for embeddings (chunk_for_embedding)
2. LLM-optimized chunking (chunk_for_llm)

With optional structure-aware chunking when PDF path is available.
"""

import re
from pathlib import Path
from typing import Optional, Union

# Indicator patterns for German municipal documents
INDICATOR_PATTERNS = {
    "percentage": r"\d+(?:[,\.]\d+)?\s*%",
    "currency": r"\d+(?:[,\.]\d+)?\s*(?:Millionen|Mio\.?|Tsd\.?|Mrd\.?)?\s*(?:Euro|EUR|€)",
    "measurement": r"\d+(?:[,\.]\d+)?\s*(?:km|m²|qm|MW|GW|kW|ha|t|kg|m)",
    "time_target": r"(?:bis|ab|von|nach|vor|zum|zur)\s+(?:20\d{2}|Jahr\s+20\d{2})",
    "quantity": r"\d+(?:[,\.]\d+)?\s*(?:neue|mehr|weniger|zusätzliche)",
    "rate": r"\d+(?:[,\.]\d+)?\s*%?\s*(?:pro|je|per)\s+(?:Jahr|Monat|Tag|Einwohner|km)",
    "compound": r"\d+(?:[,\.]\d+)?\s*(?:von|aus)\s+\d+",  # "5 von 10", "3 aus 20"
}


def contains_indicator_context(
    text: str, position: int, window: int = 150
) -> tuple[bool, int | None]:
    """
    Check if there's an indicator pattern near a split position.

    Args:
        text: The full text
        position: The proposed split position
        window: How many characters before/after to check

    Returns:
        (has_indicator, safe_split_position)
    """
    # Get context around the split point
    start = max(0, position - window)
    end = min(len(text), position + window)
    context = text[start:end]

    # Check each indicator pattern
    for _pattern_name, pattern in INDICATOR_PATTERNS.items():
        matches = list(re.finditer(pattern, context, re.IGNORECASE))

        for match in matches:
            # Get actual position in original text
            match_start = start + match.start()
            match_end = start + match.end()

            # If split would occur within or very close to indicator
            if match_start - 50 <= position <= match_end + 50:
                # Find a safer split position
                # Try after the indicator context
                safe_pos = match_end + 50
                if safe_pos < len(text):
                    # Look for sentence end
                    sentence_end = text.find(". ", match_end, safe_pos + 100)
                    if sentence_end != -1:
                        return True, sentence_end + 2

                # Try before the indicator context
                safe_pos = match_start - 50
                if safe_pos > 0:
                    # Look for sentence end before indicator
                    sentence_end = text.rfind(". ", safe_pos - 100, match_start)
                    if sentence_end != -1:
                        return True, sentence_end + 2

                return True, None

    return False, position


def find_safe_split_point(text: str, target_pos: int, max_chars: int) -> int:
    """
    Find a safe position to split text that doesn't break indicator context.

    Args:
        text: The text to split
        target_pos: The desired split position
        max_chars: Maximum allowed characters

    Returns:
        Safe split position
    """
    # First check if there's an indicator near the target position
    has_indicator, safe_pos = contains_indicator_context(text, target_pos)

    if has_indicator and safe_pos:
        # Use the safe position if it's within bounds
        if safe_pos <= max_chars:
            return safe_pos

    # Try to find paragraph boundary near target
    para_before = text.rfind("\n\n", max(0, target_pos - 200), target_pos)
    para_after = text.find("\n\n", target_pos, min(len(text), target_pos + 200))

    if para_before != -1 and (target_pos - para_before) < 200:
        return para_before + 2
    elif para_after != -1 and (para_after - target_pos) < 200:
        return para_after

    # Try to find sentence boundary
    sentence_before = text.rfind(". ", max(0, target_pos - 100), target_pos)
    if sentence_before != -1:
        return sentence_before + 2

    # Last resort: return original position
    return target_pos


def is_heading(line: str) -> bool:
    """
    Detect if a line is likely a heading.

    Criteria:
    - Numbered sections (1.1, 2.3.4, etc.)
    - Title case lines
    - All uppercase lines
    - Lines starting with specific keywords
    - Lines ending with a colon
    """
    line = line.strip()
    if not line:
        return False

    # Check for numbered sections (e.g., "1.1", "2.3.4")
    if re.match(r"^\d+(\.\d+)*\.?\s+", line):
        return True

    # Check for Roman numerals
    if re.match(r"^[IVXLCDM]+\.\s+", line):
        return True

    # Check for letter sections (e.g., "a)", "b)")
    if re.match(r"^[a-z]\)\s+", line, re.IGNORECASE):
        return True

    # German section keywords
    german_keywords = [
        "Kapitel",
        "Abschnitt",
        "Teil",
        "Artikel",
        "Paragraph",
        "Handlungsfeld",
        "Maßnahme",
        "Projekt",
        "Ziel",
        "Indikator",
        "Anhang",
        "Anlage",
        "Übersicht",
        "Zusammenfassung",
    ]
    if any(line.startswith(keyword) for keyword in german_keywords):
        return True

    # Check if it's all uppercase (but not too short)
    if len(line) > 3 and line.isupper():
        return True

    # Check if it's title case (most words capitalized)
    words = line.split()
    if len(words) >= 2:
        capitalized = sum(1 for word in words if word and word[0].isupper())
        if capitalized / len(words) > 0.7:
            return True

    # Check if line ends with colon (often indicates a heading)
    if line.endswith(":") and len(line) < 100:
        return True

    return False


def extract_chunk_topic(chunk: str) -> dict:
    """
    Extract topic/heading information from a chunk.

    Returns:
        Dict with 'heading' and optionally 'action_field' if detected
    """
    lines = chunk.strip().split("\n")

    # Look for heading in first few lines
    for _i, line in enumerate(lines[:5]):
        line = line.strip()
        if not line:
            continue

        if is_heading(line):
            # Check if it's a Handlungsfeld
            if "Handlungsfeld" in line or re.match(r"^\d+\.\s+[A-ZÄÖÜ]", line):
                # Try to extract the action field name
                # Remove numbering and keywords
                field_name = re.sub(r"^\d+(\.\d+)*\.?\s*", "", line)
                field_name = re.sub(
                    r"^Handlungsfeld\s*:?\s*", "", field_name, flags=re.IGNORECASE
                )
                field_name = field_name.strip()

                if field_name:
                    return {"heading": line, "action_field": field_name}

            return {"heading": line}

    # No clear heading found
    # Try to extract from content
    full_text = " ".join(lines[:10])  # Look at first 10 lines

    # Common patterns for action fields in content
    patterns = [
        r"im Handlungsfeld[:\s]+([^\.]+)",
        r"Handlungsfeld[:\s]+([^\.]+)",
        r"für das Handlungsfeld[:\s]+([^\.]+)",
        r"zum Handlungsfeld[:\s]+([^\.]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            action_field = match.group(1).strip()
            # Clean up common suffixes
            action_field = re.sub(r"\s*[,;:].*$", "", action_field)
            return {"action_field": action_field}

    return {}


def split_by_heading(text: str) -> list[str]:
    """
    Split text into chunks at heading boundaries.

    This creates natural document sections based on structure.
    """
    lines = text.split("\n")
    chunks = []
    current_chunk: list[str] = []

    for _i, line in enumerate(lines):
        # Check if this line is a heading
        if is_heading(line):
            # If we have accumulated content, save it as a chunk
            if current_chunk and len("\n".join(current_chunk).strip()) > 100:
                chunks.append("\n".join(current_chunk))

            # Start new chunk with this heading
            current_chunk = [line]
        else:
            # Add line to current chunk
            current_chunk.append(line)

    # Don't forget the last chunk
    if current_chunk and len("\n".join(current_chunk).strip()) > 100:
        chunks.append("\n".join(current_chunk))

    return chunks


def split_large_chunk(text: str, max_chars: int) -> list[str]:
    """
    Split a large chunk into smaller pieces at safe boundaries.

    Respects:
    - Paragraph boundaries
    - Indicator context
    - Sentence boundaries
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text

    while len(remaining) > max_chars:
        # Find split point
        target_pos = max_chars

        # Check for indicator context
        split_pos = find_safe_split_point(remaining, target_pos, max_chars)

        # Extract chunk
        chunk = remaining[:split_pos].strip()
        if chunk:
            chunks.append(chunk)

        # Update remaining text
        remaining = remaining[split_pos:].strip()

    # Add final chunk
    if remaining:
        chunks.append(remaining)

    return chunks


def split_by_lines(text: str, max_chars: int) -> list[str]:
    """
    Split text by accumulating lines until size limit.

    This is a fallback when no clear structure is found.
    """
    lines = text.split("\n")
    chunks = []
    current_chunk: list[str] = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1  # +1 for newline

        # Check if adding this line would exceed limit
        if current_size + line_size > max_chars and current_chunk:
            # Check for indicator context at boundary
            chunk_text = "\n".join(current_chunk)
            has_indicator, _ = contains_indicator_context(
                chunk_text + "\n" + line, len(chunk_text)
            )

            if has_indicator:
                # Include the line with indicator
                current_chunk.append(line)
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_size = 0
            else:
                # Safe to split here
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size

    # Add remaining lines
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def merge_short_chunks(chunks: list[str], min_chars=3000, max_chars=5000) -> list[str]:
    """Merge chunks that are too short."""
    if not chunks:
        return []

    merged = []
    current = chunks[0]

    for next_chunk in chunks[1:]:
        # Check if we should merge
        if len(current) < min_chars and len(current) + len(next_chunk) + 2 <= max_chars:
            # Safe to merge
            current = current + "\n\n" + next_chunk
        else:
            # Save current and start new
            merged.append(current)
            current = next_chunk

    # Don't forget the last chunk
    if current:
        merged.append(current)

    return merged


def chunk_for_embedding(cleaned_text: str, max_chars: int = 5000) -> list[str]:
    """
    Smart chunking for embeddings that respects document structure.

    This is the main entry point for semantic chunking.
    Used for creating chunks that will be embedded for retrieval.

    Args:
        cleaned_text: The cleaned text to chunk
        max_chars: Maximum characters per chunk (default: 5000)

    Returns:
        List of text chunks
    """
    # Step 1: Try to split by headings first
    heading_chunks = split_by_heading(cleaned_text)

    # Step 2: Process each heading-based chunk
    processed_chunks = []
    for chunk in heading_chunks:
        if len(chunk) <= max_chars:
            processed_chunks.append(chunk)
        else:
            # Split large chunks
            sub_chunks = split_large_chunk(chunk, max_chars)
            processed_chunks.extend(sub_chunks)

    # Step 3: Merge short chunks
    final_chunks = merge_short_chunks(
        processed_chunks,
        min_chars=int(max_chars * 0.6),  # 60% of max as minimum
        max_chars=max_chars,
    )

    return final_chunks


# Alias for backward compatibility
smart_chunk = chunk_for_embedding


def chunk_for_llm(
    chunks: list[str], max_chars: int = 20000, min_chars: int = 15000
) -> list[str]:
    """
    Prepare chunks for LLM processing using simple size-based merging.

    This takes the output of chunk_for_embedding and merges them
    into larger chunks optimized for LLM context windows.

    Args:
        chunks: List of text chunks from semantic chunker
        max_chars: Maximum characters per LLM chunk
        min_chars: Target minimum characters per LLM chunk

    Returns:
        List of merged chunks optimized for LLM processing
    """
    if not chunks:
        return []

    result_chunks = []
    current_merge: list[str] = []
    current_size = 0

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        chunk_size = len(chunk)

        # Decide whether to merge with current or start new
        if not current_merge:
            # First chunk
            current_merge = [chunk]
            current_size = chunk_size
        elif chunk_size > max_chars:
            # Single chunk too large - flush current and add oversized chunk as-is
            if current_merge:
                result_chunks.append("\n\n".join(current_merge))
            result_chunks.append(chunk)  # Add oversized chunk as-is
            current_merge = []
            current_size = 0
        elif current_size + chunk_size + 2 > max_chars:
            # Would exceed max size - flush current and start new
            result_chunks.append("\n\n".join(current_merge))
            current_merge = [chunk]
            current_size = chunk_size
        else:
            # Can merge
            current_merge.append(chunk)
            current_size += chunk_size + 2  # +2 for \n\n separator

    # Don't forget the last merge
    if current_merge:
        result_chunks.append("\n\n".join(current_merge))

    # Post-process: try to merge very small chunks
    final_chunks: list[str] = []
    for _i, chunk in enumerate(result_chunks):
        chunk = chunk.strip()
        if not chunk:
            continue

        # If chunk is too small and we have a previous chunk, try to merge
        if (
            len(chunk) < min_chars
            and final_chunks
            and len(final_chunks[-1]) + len(chunk) + 2 <= max_chars
        ):
            # Merge with previous chunk
            final_chunks[-1] = final_chunks[-1] + "\n\n" + chunk
        else:
            final_chunks.append(chunk)

    return final_chunks


# Alias for backward compatibility
prepare_llm_chunks = chunk_for_llm


def analyze_chunk_quality(chunks: list[str], stage: str = "unknown") -> dict:
    """Analyze the quality of prepared chunks."""
    if not chunks:
        return {"error": "No chunks provided"}

    sizes = [len(chunk) for chunk in chunks]

    # Count chunks with OCR page markers
    chunks_with_pages = sum(1 for chunk in chunks if "[OCR Page" in chunk)

    # Basic size statistics
    return {
        "stage": stage,
        "total_chunks": len(chunks),
        "total_chars": sum(sizes),
        "size_stats": {
            "min": min(sizes),
            "max": max(sizes),
            "avg": sum(sizes) / len(sizes),
            "median": sorted(sizes)[len(sizes) // 2],
        },
        "structural_stats": {
            "chunks_with_headings": sum(
                1
                for chunk in chunks
                if any(is_heading(line) for line in chunk.split("\n")[:5])
            ),
            "chunks_with_ocr_pages": chunks_with_pages,
            "chunks_with_indicators": sum(
                1
                for chunk in chunks
                if any(
                    re.search(pattern, chunk, re.IGNORECASE)
                    for pattern in INDICATOR_PATTERNS.values()
                )
            ),
        },
    }


def chunk_for_embedding_enhanced(
    text_or_path: str | Path,
    max_chars: int = 5000,
    use_structure_aware: bool = True,
    pdf_path: str | None = None,
) -> list[str]:
    """
    Enhanced chunking that can use structure-aware approach when PDF is available.

    Args:
        text_or_path: Either extracted text (str) or path to PDF file
        max_chars: Maximum characters per chunk
        use_structure_aware: Whether to use structure-aware chunking if possible
        pdf_path: Optional PDF path if text_or_path is text

    Returns:
        List of chunks (strings only for compatibility)
    """
    # Check if we can use structure-aware chunking
    if use_structure_aware:
        # If text_or_path is a path
        if (
            isinstance(text_or_path, str | Path)
            and Path(text_or_path).exists()
            and str(text_or_path).endswith(".pdf")
        ):
            try:
                from .structure_chunker_v2 import create_structure_aware_chunks

                # Use structure-aware chunking with appropriate parameters
                chunks = create_structure_aware_chunks(
                    str(text_or_path),
                    max_chunk_size=max_chars,
                    min_chunk_size=int(max_chars * 0.3),  # 30% of max
                    overlap_size=200,
                    max_pages=None,
                )

                # Convert to list of text for compatibility
                return [chunk["text"] for chunk in chunks]

            except ImportError:
                # Fall back to basic chunking
                pass
            except Exception as e:
                print(f"Structure-aware chunking failed: {e}")
                # Fall back to basic chunking

        # If we have a pdf_path provided separately
        elif pdf_path and Path(pdf_path).exists():
            try:
                from .structure_chunker_v2 import create_structure_aware_chunks

                chunks = create_structure_aware_chunks(
                    pdf_path,
                    max_chunk_size=max_chars,
                    min_chunk_size=int(max_chars * 0.3),
                    overlap_size=200,
                    max_pages=None,
                )

                return [chunk["text"] for chunk in chunks]

            except:
                # Fall back to basic chunking
                pass

    # Fall back to basic text-based chunking
    if isinstance(text_or_path, str | Path) and Path(text_or_path).exists():
        # It's a file path, need to extract text first
        from .parser import extract_text_with_ocr_fallback

        text, _ = extract_text_with_ocr_fallback(str(text_or_path))
    else:
        # It's already text
        text = str(text_or_path)

    return chunk_for_embedding(text, max_chars)
