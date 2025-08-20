"""
Document chunking functionality for CURATE.

This module provides two main chunking strategies:
1. Semantic chunking for embeddings (chunk_for_embedding)
2. LLM-optimized chunking (chunk_for_llm)

With optional structure-aware chunking when PDF path is available.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.constants import INDICATOR_PATTERNS
from src.utils.text import is_heading

# Optional imports for structure-aware chunking

if TYPE_CHECKING:
    from unstructured.documents.elements import Element, Title
    from unstructured.partition.pdf import partition_pdf
else:
    try:
        from unstructured.documents.elements import Element, Title
        from unstructured.partition.pdf import partition_pdf

    except ImportError:
        Element = Any
        Title = Any
        partition_pdf = Any


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


def add_overlap_to_chunks(
    chunks: list[str], overlap_percent: float = 0.15
) -> list[str]:
    """
    Add overlap between consecutive chunks to prevent information loss.

    Args:
        chunks: List of text chunks
        overlap_percent: Percentage of chunk size to use as overlap (default: 0.15 = 15%)

    Returns:
        List of chunks with overlap added
    """
    if not chunks or len(chunks) < 2:
        return chunks

    overlapped_chunks = []

    for i, chunk in enumerate(chunks):
        current_chunk = chunk

        # Add overlap from next chunk (except for last chunk)
        if i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            overlap_size = int(len(current_chunk) * overlap_percent)

            if overlap_size > 0 and len(next_chunk) > overlap_size:
                # Get overlap text from beginning of next chunk
                overlap_text = next_chunk[:overlap_size]

                # Find good sentence boundary to avoid cutting mid-sentence
                # Look for sentence endings within reasonable range
                sentence_endings = [". ", ".\n", "! ", "!\n", "? ", "?\n"]
                best_cut = overlap_size

                for ending in sentence_endings:
                    pos = overlap_text.rfind(ending)
                    if pos > overlap_size * 0.5:  # At least 50% of desired overlap
                        best_cut = pos + len(ending)
                        break

                # If no good sentence boundary found, look for paragraph break
                if best_cut == overlap_size:
                    para_break = overlap_text.rfind("\n\n")
                    if (
                        para_break > overlap_size * 0.3
                    ):  # At least 30% of desired overlap
                        best_cut = para_break + 2

                overlap_text = overlap_text[:best_cut].strip()

                if overlap_text:
                    # Add overlap with clear separator
                    current_chunk = (
                        current_chunk
                        + "\n\n[OVERLAP_START]\n"
                        + overlap_text
                        + "\n[OVERLAP_END]"
                    )

        overlapped_chunks.append(current_chunk)

    return overlapped_chunks


def add_overlap_to_page_chunks(
    chunks: list[tuple[str, list[int]]], overlap_percent: float = 0.15
) -> list[tuple[str, list[int]]]:
    """
    Add overlap between consecutive page-aware chunks to prevent information loss.

    Args:
        chunks: List of (chunk_text, page_numbers) tuples
        overlap_percent: Percentage of chunk size to use as overlap (default: 0.15 = 15%)

    Returns:
        List of page-aware chunks with overlap added
    """
    if not chunks or len(chunks) < 2:
        return chunks

    overlapped_chunks = []

    for i, (chunk_text, chunk_pages) in enumerate(chunks):
        current_chunk = chunk_text

        # Add overlap from next chunk (except for last chunk)
        if i < len(chunks) - 1:
            next_chunk_text, _ = chunks[i + 1]
            overlap_size = int(len(current_chunk) * overlap_percent)

            if overlap_size > 0 and len(next_chunk_text) > overlap_size:
                # Get overlap text from beginning of next chunk
                overlap_text = next_chunk_text[:overlap_size]

                # Find good sentence boundary to avoid cutting mid-sentence
                sentence_endings = [". ", ".\n", "! ", "!\n", "? ", "?\n"]
                best_cut = overlap_size

                for ending in sentence_endings:
                    pos = overlap_text.rfind(ending)
                    if pos > overlap_size * 0.5:  # At least 50% of desired overlap
                        best_cut = pos + len(ending)
                        break

                # If no good sentence boundary found, look for paragraph break
                if best_cut == overlap_size:
                    para_break = overlap_text.rfind("\n\n")
                    if (
                        para_break > overlap_size * 0.3
                    ):  # At least 30% of desired overlap
                        best_cut = para_break + 2

                overlap_text = overlap_text[:best_cut].strip()

                if overlap_text:
                    # Add overlap with clear separator
                    current_chunk = (
                        current_chunk
                        + "\n\n[OVERLAP_START]\n"
                        + overlap_text
                        + "\n[OVERLAP_END]"
                    )

        overlapped_chunks.append((current_chunk, chunk_pages))

    return overlapped_chunks


def add_overlap_to_dict_chunks(
    chunks: list[dict[str, Any]], overlap_percent: float = 0.15
) -> list[dict[str, Any]]:
    """
    Add overlap between consecutive dictionary-based chunks to prevent information loss.

    Args:
        chunks: List of chunk dictionaries with 'text' key
        overlap_percent: Percentage of chunk size to use as overlap (default: 0.15 = 15%)

    Returns:
        List of chunk dictionaries with overlap added to text
    """
    if not chunks or len(chunks) < 2:
        return chunks

    overlapped_chunks = []

    for i, chunk in enumerate(chunks):
        current_chunk = chunk.copy()
        current_text = current_chunk["text"]

        # Add overlap from next chunk (except for last chunk)
        if i < len(chunks) - 1:
            next_chunk_text = chunks[i + 1]["text"]
            overlap_size = int(len(current_text) * overlap_percent)

            if overlap_size > 0 and len(next_chunk_text) > overlap_size:
                # Get overlap text from beginning of next chunk
                overlap_text = next_chunk_text[:overlap_size]

                # Find good sentence boundary to avoid cutting mid-sentence
                sentence_endings = [". ", ".\n", "! ", "!\n", "? ", "?\n"]
                best_cut = overlap_size

                for ending in sentence_endings:
                    pos = overlap_text.rfind(ending)
                    if pos > overlap_size * 0.5:  # At least 50% of desired overlap
                        best_cut = pos + len(ending)
                        break

                # If no good sentence boundary found, look for paragraph break
                if best_cut == overlap_size:
                    para_break = overlap_text.rfind("\n\n")
                    if (
                        para_break > overlap_size * 0.3
                    ):  # At least 30% of desired overlap
                        best_cut = para_break + 2

                overlap_text = overlap_text[:best_cut].strip()

                if overlap_text:
                    # Add overlap with clear separator
                    current_chunk["text"] = (
                        current_text
                        + "\n\n[OVERLAP_START]\n"
                        + overlap_text
                        + "\n[OVERLAP_END]"
                    )

        overlapped_chunks.append(current_chunk)

    return overlapped_chunks


def chunk_for_embedding(
    cleaned_text: str, max_chars: int = 5000, add_overlap: bool = False
) -> list[str]:
    """
    Smart chunking for embeddings that respects document structure.

    This is the main entry point for semantic chunking.
    Used for creating chunks that will be embedded for retrieval.

    Args:
        cleaned_text: The cleaned text to chunk
        max_chars: Maximum characters per chunk (default: 5000)
        add_overlap: Whether to add 15% overlap between chunks (default: False)

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

    # Step 4: Add overlap if requested
    if add_overlap:
        final_chunks = add_overlap_to_chunks(final_chunks)

    return final_chunks


def chunk_for_llm_with_pages(
    page_aware_text: list[tuple[str, int]],
    max_chars: int = 20000,
    min_chars: int = 15000,
    doc_title: str = "Dokument",
    add_overlap: bool = False,
) -> list[tuple[str, list[int]]]:
    """
    Prepare page-aware chunks for LLM processing with context headers.

    This takes page-aware text and creates larger chunks optimized for LLM context windows
    while preserving information about which pages each chunk came from and adding
    contextual headers for better LLM understanding.

    Args:
        page_aware_text: List of (text, page_number) tuples from parser
        max_chars: Maximum characters per LLM chunk
        min_chars: Target minimum characters per LLM chunk
        doc_title: Title of the document for context headers
        add_overlap: Whether to add 15% overlap between chunks (default: False)

    Returns:
        List of (chunk_text, page_numbers) tuples where chunk_text includes context header
    """
    if not page_aware_text:
        return []

    result_chunks: list[tuple[str, list[int]]] = []
    current_texts: list[str] = []
    current_pages: set[int] = set()
    current_size = 0

    for page_text, page_num in page_aware_text:
        page_text = page_text.strip()
        if not page_text:
            continue

        text_size = len(page_text)

        # Decide whether to merge with current chunk or start new
        if not current_texts:
            # First chunk
            current_texts = [page_text]
            current_pages = {page_num}
            current_size = text_size
        elif text_size > max_chars:
            # Single page too large - flush current and add oversized page as-is
            if current_texts:
                result_chunks.append(
                    ("\n\n".join(current_texts), sorted(current_pages))
                )
            result_chunks.append((page_text, [page_num]))
            current_texts = []
            current_pages = set()
            current_size = 0
        elif current_size + text_size + 2 > max_chars:
            # Would exceed max size - flush current and start new
            result_chunks.append(("\n\n".join(current_texts), sorted(current_pages)))
            current_texts = [page_text]
            current_pages = {page_num}
            current_size = text_size
        else:
            # Can merge
            current_texts.append(page_text)
            current_pages.add(page_num)
            current_size += text_size + 2  # +2 for \n\n separator

    # Don't forget the last chunk
    if current_texts:
        result_chunks.append(("\n\n".join(current_texts), sorted(current_pages)))

    # Post-process: try to merge very small chunks
    final_chunks: list[tuple[str, list[int]]] = []
    for chunk_text, chunk_pages in result_chunks:
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        # If chunk is too small and we have a previous chunk, try to merge
        if (
            len(chunk_text) < min_chars
            and final_chunks
            and len(final_chunks[-1][0]) + len(chunk_text) + 2 <= max_chars
        ):
            # Merge with previous chunk
            prev_text, prev_pages = final_chunks[-1]
            merged_text = prev_text + "\n\n" + chunk_text
            merged_pages = sorted(set(prev_pages + chunk_pages))
            final_chunks[-1] = (merged_text, merged_pages)
        else:
            final_chunks.append((chunk_text, chunk_pages))

    # Add context headers to each chunk
    chunks_with_headers: list[tuple[str, list[int]]] = []
    for chunk_text, chunk_pages in final_chunks:
        # Create context header with clear structure
        page_range = (
            f"{min(chunk_pages)}-{max(chunk_pages)}"
            if len(chunk_pages) > 1
            else str(chunk_pages[0])
        )

        # Try to extract section info from the beginning of the chunk
        section_info = ""
        lines = chunk_text.splitlines()[:5]  # Check first 5 lines for section headers
        for line in lines:
            line = line.strip()
            if line and is_heading(line):
                # Clean up the section title
                section_info = line[:100]  # Limit length
                break

        # Create a clearer, more structured context header
        context_lines = [
            "=" * 80,
            f"DOKUMENT: {doc_title}",
            f"SEITEN: {page_range}",
        ]

        if section_info:
            context_lines.append(f"ABSCHNITT: {section_info}")

        context_lines.extend(["=" * 80, ""])  # Empty line before content

        context_header = "\n".join(context_lines)

        # Add header to chunk
        chunk_with_header = context_header + chunk_text
        chunks_with_headers.append((chunk_with_header, chunk_pages))

    # Add overlap if requested
    if add_overlap:
        chunks_with_headers = add_overlap_to_page_chunks(chunks_with_headers)

    return chunks_with_headers
