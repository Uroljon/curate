"""
Structure-aware document chunking using Unstructured.io.

This module provides advanced chunking that respects document structure,
preserves hierarchical context, and optimizes for German municipal documents.
"""

import re
from pathlib import Path
from typing import Any, Optional

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element, Title
from unstructured.partition.pdf import partition_pdf


def extract_hierarchical_path(element: Element) -> str:
    """
    Extract hierarchical path from element metadata.

    For example: "2. Handlungsfeld Klimaschutz > 2.1 Energieeffizienz"
    """
    # Get parent information if available (unused for now)

    # Build path from element text
    if isinstance(element, Title):
        return element.text

    return ""


def add_overlap_to_chunks(
    chunks: list[dict[str, Any]], overlap_chars: int = 200
) -> list[dict[str, Any]]:
    """
    Add overlap between consecutive chunks for better context continuity.

    Args:
        chunks: List of chunk dictionaries
        overlap_chars: Number of characters to overlap (default: 200)

    Returns:
        List of chunks with overlap added
    """
    if not chunks or len(chunks) < 2:
        return chunks

    enhanced_chunks = []

    for i, chunk in enumerate(chunks):
        text = chunk["text"]

        # Add suffix from next chunk
        if i < len(chunks) - 1:
            next_text = chunks[i + 1]["text"]
            if len(next_text) > overlap_chars:
                # Find a good break point (sentence end)
                suffix = next_text[:overlap_chars]
                last_period = suffix.rfind(". ")
                if last_period > 0:
                    suffix = suffix[: last_period + 2]
                text = text + "\n\n[...]\n\n" + suffix

        # Add prefix from previous chunk
        if i > 0:
            prev_text = chunks[i - 1]["text"]
            if len(prev_text) > overlap_chars:
                # Find a good break point (sentence end)
                prefix = prev_text[-overlap_chars:]
                first_period = prefix.find(". ")
                if first_period > 0 and first_period < len(prefix) - 2:
                    prefix = prefix[first_period + 2 :]
                text = prefix + "\n\n[...]\n\n" + text

        enhanced_chunk = chunk.copy()
        enhanced_chunk["text"] = text
        enhanced_chunk["has_overlap"] = i > 0 or i < len(chunks) - 1
        enhanced_chunks.append(enhanced_chunk)

    return enhanced_chunks


def structure_aware_chunk(
    pdf_path: str,
    max_characters: int = 2000,
    min_characters: int = 500,
    overlap_chars: int = 200,
    combine_under_n_chars: int = 300,
    include_metadata: bool = True,
) -> list[dict[str, Any]]:
    """
    Create structure-aware chunks from a PDF using Unstructured.io.

    This function:
    1. Extracts structured elements from PDF
    2. Chunks by title/heading to preserve document structure
    3. Adds overlap between chunks for context continuity
    4. Preserves hierarchical metadata

    Args:
        pdf_path: Path to the PDF file
        max_characters: Maximum characters per chunk (default: 2000)
        min_characters: Minimum characters before creating new chunk (default: 500)
        overlap_chars: Characters to overlap between chunks (default: 200)
        combine_under_n_chars: Combine small sections under this size (default: 300)
        include_metadata: Whether to include metadata in chunks (default: True)

    Returns:
        List of chunk dictionaries with text and metadata
    """
    # Ensure path exists
    if not Path(pdf_path).exists():
        error_msg = f"PDF file not found: {pdf_path}"
        raise FileNotFoundError(error_msg)

    # Extract structured elements from PDF
    # Using "fast" strategy for initial testing (change to "auto" for production)
    elements = partition_pdf(
        filename=pdf_path,
        strategy="fast",  # auto, hi_res, or fast
        languages=["deu"],  # German language support
        include_page_breaks=True,
        infer_table_structure=False,  # Faster without table inference
    )

    # Chunk by title to preserve document structure
    # This respects headings and creates semantic chunks
    chunks = chunk_by_title(
        elements,
        max_characters=max_characters,
        combine_text_under_n_chars=combine_under_n_chars,
        multipage_sections=True,  # Allow sections to span pages
        overlap=0,  # We'll add custom overlap later
    )

    # Convert chunks to our format
    structured_chunks: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        # Extract text content
        if hasattr(chunk, "text"):
            text = chunk.text
        else:
            # Chunk might be a composite element
            text = str(chunk)

        # Build metadata
        metadata = {
            "chunk_index": i,
            "chunk_id": f"chunk_{i}",
        }

        if include_metadata and hasattr(chunk, "metadata"):
            chunk_meta = chunk.metadata

            # Add page information
            if hasattr(chunk_meta, "page_number"):
                metadata["page_number"] = chunk_meta.page_number

            # Add section/heading information
            if hasattr(chunk_meta, "section"):
                metadata["section"] = chunk_meta.section

            if hasattr(chunk_meta, "parent_id"):
                metadata["parent_section"] = chunk_meta.parent_id

            # Detect if this is a heading/title
            if hasattr(chunk, "__class__") and "Title" in chunk.__class__.__name__:
                metadata["is_heading"] = True
                metadata["heading_level"] = _detect_heading_level(text)

            # Add filename
            if hasattr(chunk_meta, "filename"):
                metadata["source_file"] = chunk_meta.filename

        # Detect action field (Handlungsfeld)
        action_field = _extract_action_field(text)
        if action_field:
            metadata["action_field"] = action_field

        # Check for indicators
        indicator_count = _count_indicators(text)
        if indicator_count > 0:
            metadata["indicator_count"] = indicator_count
            metadata["has_indicators"] = True

        structured_chunks.append(
            {
                "text": text.strip(),
                "metadata": metadata,
                "char_count": len(text),
            }
        )

    # Filter out very small chunks (likely just headings or fragments)
    structured_chunks = [
        chunk
        for chunk in structured_chunks
        if (chunk.get("char_count", 0) >= min_characters)
        or (chunk.get("metadata", {}).get("is_heading", False))
    ]

    # Add overlap between chunks
    if overlap_chars > 0:
        structured_chunks = add_overlap_to_chunks(structured_chunks, overlap_chars)

    return structured_chunks


def _detect_heading_level(text: str) -> int:
    """Detect the hierarchical level of a heading."""
    # Check for numbered sections
    match = re.match(r"^(\d+(?:\.\d+)*)", text.strip())
    if match:
        # Count dots to determine level
        number = match.group(1)
        return number.count(".") + 1

    # Check for special keywords
    if any(
        keyword in text.lower()
        for keyword in ["handlungsfeld", "kapitel", "abschnitt", "teil"]
    ):
        return 1

    # Default level
    return 2


def _extract_action_field(text: str) -> str | None:
    """Extract action field (Handlungsfeld) from text."""
    # Direct heading patterns
    patterns = [
        r"Handlungsfeld[:\s]+([^\n.]+)",
        r"^\d+\.\s*([A-ZÄÖÜ][^:\n]+)(?:\s|$)",  # Numbered sections
        r"^([A-ZÄÖÜ][^:\n]+):?\s*$",  # Title case lines
    ]

    lines = text.split("\n")
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                field = match.group(1).strip()
                # Clean up common suffixes
                field = re.sub(r"\s*[,;:].*$", "", field)
                if 3 < len(field) < 100:  # Reasonable length
                    return field

    # Check in content
    content_patterns = [
        r"im Handlungsfeld[:\s]+([^\.,]+)",
        r"für das Handlungsfeld[:\s]+([^\.,]+)",
        r"zum Handlungsfeld[:\s]+([^\.,]+)",
    ]

    full_text = " ".join(lines[:10])
    for pattern in content_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def _count_indicators(text: str) -> int:
    """Count potential indicators in text."""
    # Patterns from the original chunker
    patterns = [
        r"\d+(?:[,\.]\d+)?\s*%",  # Percentages
        r"\d+(?:[,\.]\d+)?\s*(?:Millionen|Mio\.?|Tsd\.?|Mrd\.?)\s*(?:Euro|EUR|€)",  # Currency
        r"\d+(?:[,\.]\d+)?\s*(?:km|m²|qm|MW|GW|kW|ha|t|kg|m)",  # Measurements
        r"(?:bis|ab|von|nach|vor|zum|zur)\s+(?:20\d{2}|Jahr\s+20\d{2})",  # Time targets
        r"\d+(?:[,\.]\d+)?\s*(?:neue|mehr|weniger|zusätzliche)",  # Quantities
    ]

    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        count += len(matches)

    return count


def chunk_pdf_with_overlap(
    pdf_path: str, target_chunk_size: int = 1500, overlap_size: int = 200
) -> list[dict[str, Any]]:
    """
    Convenience function for chunking with sensible defaults for German municipal documents.

    Args:
        pdf_path: Path to PDF file
        target_chunk_size: Target size for chunks in characters
        overlap_size: Size of overlap between chunks

    Returns:
        List of structured chunks with overlap
    """
    return structure_aware_chunk(
        pdf_path=pdf_path,
        max_characters=target_chunk_size,
        min_characters=target_chunk_size // 3,
        overlap_chars=overlap_size,
        combine_under_n_chars=300,
        include_metadata=True,
    )
