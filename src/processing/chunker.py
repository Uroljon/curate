"""
Document chunking functionality for CURATE.

This module provides two main chunking strategies:
1. Semantic chunking for embeddings (chunk_for_embedding)
2. LLM-optimized chunking (chunk_for_llm)

With optional structure-aware chunking when PDF path is available.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from src.core.constants import ACTION_FIELD_PATTERNS, INDICATOR_PATTERNS
from src.utils.text import is_heading

# Optional imports for structure-aware chunking
UNSTRUCTURED_AVAILABLE = False

if TYPE_CHECKING:
    from unstructured.documents.elements import Element, Title
    from unstructured.partition.pdf import partition_pdf
else:
    try:
        from unstructured.documents.elements import Element, NarrativeText, Title
        from unstructured.partition.pdf import partition_pdf

        UNSTRUCTURED_AVAILABLE = True
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

    # Use patterns from constants
    for pattern in ACTION_FIELD_PATTERNS:
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


class StructureAwareChunker:
    """Advanced chunker that respects document structure while avoiding over-fragmentation."""

    def __init__(
        self,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 500,
        overlap_size: int = 200,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size

        # Patterns for significant headings (German municipal documents)
        self.heading_patterns = [
            r"^\d+\.?\s+Handlungsfeld",  # Action fields
            r"^\d+\.\d+\.?\s+",  # Numbered sections (2.1, 3.4.1, etc.)
            r"^Handlungsfeld\s+\d+",
            r"^Kapitel\s+\d+",
            r"^Abschnitt\s+\d+",
            r"^Teil\s+[A-Z]",
            r"^Anhang\s+[A-Z\d]",
        ]

        # Keywords that indicate major sections
        self.section_keywords = [
            "handlungsfeld",
            "maßnahmen",
            "projekte",
            "indikatoren",
            "ziele",
            "strategische ziele",
            "ausgangslage",
            "herausforderungen",
        ]

    def is_significant_heading(self, element: Element) -> bool:
        """Determine if an element is a significant heading worth creating a chunk boundary."""
        if not isinstance(element, Title):
            return False

        text = element.text.strip()
        if not text:
            return False

        # Check against heading patterns
        for pattern in self.heading_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True

        # Check for section keywords
        text_lower = text.lower()
        for keyword in self.section_keywords:
            if keyword in text_lower:
                return True

        # Check if it's a major numbered section (but not sub-sub-sections)
        if re.match(r"^\d+\.\s+[A-ZÄÖÜ]", text):  # e.g., "1. Introduction"
            return True

        # Avoid treating every small title as a heading
        if len(text) < 10 or (text.isupper() and len(text) < 20):
            return False

        return False

    def group_elements_by_section(self, elements: list[Element]) -> list[list[Element]]:
        """Group elements into sections based on significant headings."""
        sections = []
        current_section: list[Element] = []

        for element in elements:
            if self.is_significant_heading(element):
                # Save previous section if it has content
                if current_section:
                    sections.append(current_section)
                # Start new section with this heading
                current_section = [element]
            else:
                # Add to current section
                current_section.append(element)

        # Don't forget the last section
        if current_section:
            sections.append(current_section)

        return sections

    def section_to_chunks(self, section: list[Element]) -> list[dict[str, Any]]:
        """Convert a section of elements into appropriately sized chunks."""
        chunks: list[dict[str, Any]] = []
        current_text: list[str] = []
        current_size = 0
        section_heading = None

        # Extract section heading if present
        if section and isinstance(section[0], Title):
            section_heading = section[0].text.strip()

        for element in section:
            # Skip page breaks and headers/footers
            element_type = type(element).__name__
            if element_type in ["PageBreak", "Header", "Footer"]:
                continue

            text = element.text.strip()
            if not text:
                continue

            text_size = len(text)

            # Check if adding this would exceed max size
            if current_size + text_size > self.max_chunk_size and current_text:
                # Create chunk from accumulated text
                chunk_text = "\n\n".join(current_text)
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            "section_heading": section_heading,
                            "char_count": len(chunk_text),
                        },
                    }
                )

                # Start new chunk
                current_text = [text]
                current_size = text_size
            else:
                # Add to current chunk
                current_text.append(text)
                current_size += text_size + 2  # +2 for \n\n

        # Create final chunk if there's content
        if current_text:
            chunk_text = "\n\n".join(current_text)
            # Only create chunk if it meets minimum size or is the only chunk in section
            if len(chunk_text) >= self.min_chunk_size or not chunks:
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            "section_heading": section_heading,
                            "char_count": len(chunk_text),
                        },
                    }
                )
            elif chunks:
                # Merge with previous chunk if too small
                last_chunk = chunks[-1]
                last_chunk["text"] = last_chunk["text"] + "\n\n" + chunk_text
                last_chunk["metadata"]["char_count"] = len(last_chunk["text"])

        return chunks

    def add_overlap(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add overlap between consecutive chunks."""
        if not chunks or len(chunks) < 2 or self.overlap_size <= 0:
            return chunks

        enhanced_chunks = []

        for i, chunk in enumerate(chunks):
            text = chunk["text"]

            # Add prefix from previous chunk
            if i > 0:
                prev_text = chunks[i - 1]["text"]
                if len(prev_text) > self.overlap_size:
                    prefix = prev_text[-self.overlap_size :]
                    # Find sentence boundary
                    last_period = prefix.find(". ")
                    if 0 < last_period < len(prefix) - 2:
                        prefix = prefix[last_period + 2 :]
                    text = f"[...{prefix}]\n\n{text}"

            # Add suffix from next chunk
            if i < len(chunks) - 1:
                next_text = chunks[i + 1]["text"]
                if len(next_text) > self.overlap_size:
                    suffix = next_text[: self.overlap_size]
                    # Find sentence boundary
                    first_period = suffix.rfind(". ")
                    if first_period > 0:
                        suffix = suffix[: first_period + 2]
                    text = f"{text}\n\n[{suffix}...]"

            enhanced_chunk = chunk.copy()
            enhanced_chunk["text"] = text
            enhanced_chunk["has_overlap"] = True
            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    def chunk_pdf(
        self, pdf_path: str, max_pages: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Create structure-aware chunks from a PDF.

        Args:
            pdf_path: Path to the PDF file
            max_pages: Maximum number of pages to process (for testing)

        Returns:
            List of chunks with text and metadata
        """
        # Extract elements from PDF
        partition_kwargs = {
            "filename": pdf_path,
            "strategy": "fast",  # Use "auto" or "hi_res" for better OCR
            "include_page_breaks": True,
        }

        if max_pages:
            partition_kwargs["max_partition"] = max_pages

        elements = partition_pdf(
            filename=str(partition_kwargs["filename"]),
            strategy=str(partition_kwargs["strategy"]),
            languages=["deu"],  # German language support
            include_page_breaks=bool(partition_kwargs["include_page_breaks"]),
            max_partition=partition_kwargs.get("max_partition") if max_pages else None,
        )

        # Group elements by section
        sections = self.group_elements_by_section(elements)

        # Convert sections to chunks
        all_chunks = []
        for section in sections:
            chunks = self.section_to_chunks(section)
            all_chunks.extend(chunks)

        # Add chunk indices and detect features
        for i, chunk in enumerate(all_chunks):
            chunk["metadata"]["chunk_index"] = i

            # Detect action field
            action_field = self._extract_action_field(chunk["text"])
            if action_field:
                chunk["metadata"]["action_field"] = action_field

            # Count indicators
            indicator_count = self._count_indicators(chunk["text"])
            if indicator_count > 0:
                chunk["metadata"]["indicator_count"] = indicator_count
                chunk["metadata"]["has_indicators"] = True

        # Add overlap
        if self.overlap_size > 0:
            all_chunks = self.add_overlap(all_chunks)

        return all_chunks

    def _extract_action_field(self, text: str) -> str | None:
        """Extract action field from text."""
        patterns = [
            r"Handlungsfeld[:\s]+([^\n.]+)",
            r"^\d+\.\s*([A-ZÄÖÜ][^:\n]+)(?:\s|$)",
        ]

        lines = text.split("\n")
        for line in lines[:10]:  # Check first 10 lines
            for pattern in patterns:
                match = re.search(pattern, line.strip(), re.IGNORECASE)
                if match:
                    field = match.group(1).strip()
                    if 3 < len(field) < 100:
                        return field

        return None

    def _count_indicators(self, text: str) -> int:
        """Count indicators in text."""
        patterns = [
            r"\d+(?:[,\.]\d+)?\s*%",  # Percentages
            r"\d+(?:[,\.]\d+)?\s*(?:Millionen|Mio\.?|Tsd\.?)\s*(?:Euro|EUR|€)",
            r"\d+(?:[,\.]\d+)?\s*(?:km|m²|MW|GW|kW|ha|t)",
            r"(?:bis|ab|zum)\s+20\d{2}",
        ]

        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            count += len(matches)

        return count


def create_structure_aware_chunks(
    pdf_path: str,
    max_chunk_size: int = 2000,
    min_chunk_size: int = 500,
    overlap_size: int = 200,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    """
    Convenience function to create structure-aware chunks.

    Args:
        pdf_path: Path to PDF file
        max_chunk_size: Maximum chunk size in characters
        min_chunk_size: Minimum chunk size in characters
        overlap_size: Size of overlap between chunks
        max_pages: Limit pages for testing (None = all pages)

    Returns:
        List of chunks with metadata
    """
    if not UNSTRUCTURED_AVAILABLE:
        error_msg = (
            "The 'unstructured' library is required for structure-aware chunking. "
            "Please install it with: pip install unstructured[pdf]"
        )
        raise ImportError(error_msg)

    chunker = StructureAwareChunker(
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        overlap_size=overlap_size,
    )

    return chunker.chunk_pdf(pdf_path, max_pages=max_pages)


def chunk_for_embedding_with_pages(
    page_aware_text: list[tuple[str, int]],
    max_chars: int = 5000,
) -> list[dict[str, Any]]:
    """
    Page-aware chunking that preserves page number information.

    Args:
        page_aware_text: List of (text, page_number) tuples
        max_chars: Maximum characters per chunk

    Returns:
        List of chunks with metadata: [{"text": str, "pages": List[int], "chunk_id": int}]
    """
    chunks_with_pages = []
    chunk_id = 0

    # First, combine all page texts with page markers
    combined_segments = []
    for text, page_num in page_aware_text:
        combined_segments.append((text, page_num))

    # Convert to single text while tracking page boundaries
    full_text = ""
    page_positions = []  # List of (start_pos, end_pos, page_num)
    current_pos = 0

    for text, page_num in combined_segments:
        if full_text:  # Add separator between pages
            full_text += "\n\n"
            current_pos += 2

        start_pos = current_pos
        full_text += text
        current_pos += len(text)
        end_pos = current_pos

        page_positions.append((start_pos, end_pos, page_num))

    # Use existing chunking logic
    text_chunks = chunk_for_embedding(full_text, max_chars)

    # Map chunks back to pages
    for chunk_text in text_chunks:
        chunk_start = full_text.find(chunk_text)
        chunk_end = chunk_start + len(chunk_text)

        # Find which pages this chunk spans
        chunk_pages = set()
        for start_pos, end_pos, page_num in page_positions:
            # Check if chunk overlaps with this page
            if not (chunk_end <= start_pos or chunk_start >= end_pos):
                chunk_pages.add(page_num)

        chunks_with_pages.append({
            "text": chunk_text,
            "pages": sorted(chunk_pages),
            "chunk_id": chunk_id
        })
        chunk_id += 1

    return chunks_with_pages


def chunk_for_embedding_enhanced(
    text_or_path: str | Path | list[tuple[str, int]],
    max_chars: int = 5000,
    use_structure_aware: bool = True,
    pdf_path: str | None = None,
) -> list[str]:
    """
    Enhanced chunking that can use structure-aware approach when PDF is available.

    Args:
        text_or_path: Either extracted text (str), path to PDF file, or page-aware text list
        max_chars: Maximum characters per chunk
        use_structure_aware: Whether to use structure-aware chunking if possible
        pdf_path: Optional PDF path if text_or_path is text

    Returns:
        List of chunks (strings only for compatibility)
    """
    # Handle page-aware text input (new functionality)
    if isinstance(text_or_path, list) and all(isinstance(item, tuple) and len(item) == 2 for item in text_or_path):
        # This is page-aware text format: list[tuple[str, int]]
        print(f"Using page-aware text chunking with {len(text_or_path)} pages")
        page_chunks = chunk_for_embedding_with_pages(text_or_path, max_chars)
        return [chunk["text"] for chunk in page_chunks]  # Return only text for compatibility

    # Check if we can use structure-aware chunking
    if (
        use_structure_aware
        and pdf_path
        and Path(pdf_path).exists()
        and UNSTRUCTURED_AVAILABLE
    ):
        # We have a PDF path provided separately (most common case from API)
        try:
            print(f"Using structure-aware chunking for PDF: {pdf_path}")
            chunks = create_structure_aware_chunks(
                pdf_path,
                max_chunk_size=max_chars,
                min_chunk_size=int(max_chars * 0.3),
                overlap_size=200,
                max_pages=None,
            )

            print(f"Structure-aware chunking created {len(chunks)} chunks")
            return [chunk["text"] for chunk in chunks]

        except Exception as e:
            print(f"Structure-aware chunking failed: {e}")
            print("Falling back to improved semantic chunking...")
            # Fall back to basic chunking

    # Check if text_or_path might be a PDF path (less common case)
    if (
        use_structure_aware
        and isinstance(text_or_path, str | Path)
        and UNSTRUCTURED_AVAILABLE
    ):
        text_or_path_str = str(text_or_path)
        # Only check if it looks like a path (has .pdf extension and reasonable length)
        if text_or_path_str.endswith(".pdf") and len(text_or_path_str) < 500:
            try:
                # Try to check if it's a valid file path
                if Path(text_or_path_str).exists():
                    print(f"Using structure-aware chunking for PDF: {text_or_path_str}")
                    chunks = create_structure_aware_chunks(
                        text_or_path_str,
                        max_chunk_size=max_chars,
                        min_chunk_size=int(max_chars * 0.3),
                        overlap_size=200,
                        max_pages=None,
                    )

                    return [chunk["text"] for chunk in chunks]
            except:
                # Not a valid path or chunking failed, treat as text
                pass

    # Fall back to basic text-based chunking
    # At this point, text_or_path should be text content
    text = str(text_or_path)

    # If it still looks like a path that exists, extract text first
    if len(text) < 500 and text.endswith(".pdf"):
        try:
            if Path(text).exists():
                from .parser import extract_text_legacy

                text, _ = extract_text_legacy(text)
        except:
            # Not a valid path, use as text
            pass

    print(f"Using basic text chunking, creating chunks from {len(text)} characters")
    return chunk_for_embedding(text, max_chars)
