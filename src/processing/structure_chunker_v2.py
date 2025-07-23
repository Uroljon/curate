"""
Improved structure-aware document chunking.

This version better handles the excessive fragmentation from Unstructured
by being more selective about what constitutes a chunk boundary.
"""

import re
from pathlib import Path
from typing import Any, Optional

from unstructured.documents.elements import Element, NarrativeText, Title
from unstructured.partition.pdf import partition_pdf


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
    chunker = StructureAwareChunker(
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        overlap_size=overlap_size,
    )

    return chunker.chunk_pdf(pdf_path, max_pages=max_pages)
