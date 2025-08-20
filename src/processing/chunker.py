"""
Document chunking functionality for CURATE.

This module provides two main chunking strategies:
1. Semantic chunking for embeddings (chunk_for_embedding)
2. LLM-optimized chunking (chunk_for_llm)

With optional structure-aware chunking when PDF path is available.
"""

import re
from pathlib import Path
from typing import Any

from src.utils.text import is_heading

# Optional imports for structure-aware chunking removed - not used in current implementation








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
