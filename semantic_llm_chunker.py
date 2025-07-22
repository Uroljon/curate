"""
Semantic-aware LLM chunk preparation that respects document structure.

This module provides functions to prepare chunks for LLM processing while
maintaining semantic boundaries identified during initial chunking.
"""

import re
from typing import List, Tuple

from semantic_chunker import is_heading


def chunk_starts_with_heading(chunk: str) -> bool:
    """Check if a chunk starts with a heading."""
    lines = chunk.strip().split('\n')
    if not lines:
        return False
    return is_heading(lines[0])


def contains_section_marker(chunk: str) -> bool:
    """Check if chunk contains important section markers."""
    section_markers = [
        "Handlungsfeld:", "Handlungsfelder:",
        "Maßnahmen:", "Projekte:", "Ziele:", "Indikatoren:",
        "Ausgangslage:", "Umsetzung:", "Zeitplan:", "Monitoring:",
        "Kapitel", "Abschnitt"
    ]
    
    chunk_lower = chunk.lower()
    return any(marker.lower() in chunk_lower for marker in section_markers)


def get_chunk_priority(chunk: str) -> int:
    """
    Assign priority to chunks based on content importance.
    Higher priority chunks should not be split or merged carelessly.
    """
    priority = 0
    
    # High priority for chunks with section headers
    if chunk_starts_with_heading(chunk):
        priority += 10
    
    # High priority for chunks with key markers
    if contains_section_marker(chunk):
        priority += 8
    
    # Medium priority for chunks with indicators
    indicator_patterns = [
        r'\d+\s*%',  # Percentages
        r'bis\s+20\d{2}',  # Year targets
        r'\d+\s*(km|m²|MW|ha|t)',  # Units
        r'CO2|CO₂',  # Climate indicators
    ]
    
    for pattern in indicator_patterns:
        if re.search(pattern, chunk):
            priority += 2
    
    return priority


def prepare_semantic_llm_chunks(
    chunks: List[str], 
    max_chars: int = 20000, 
    min_chars: int = 15000,
    force_boundaries: bool = True
) -> List[str]:
    """
    Prepare chunks for LLM processing while respecting semantic boundaries.
    
    Args:
        chunks: List of semantically chunked text pieces
        max_chars: Maximum characters per LLM chunk
        min_chars: Minimum characters per LLM chunk (target)
        force_boundaries: If True, never merge chunks that start with headings
    
    Returns:
        List of optimized chunks for LLM processing
    """
    if not chunks:
        return []
    
    result_chunks = []
    current_merge = []
    current_size = 0
    
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        
        chunk_size = len(chunk)
        
        # Check if this chunk starts a new section
        starts_new_section = chunk_starts_with_heading(chunk)
        
        # Decide whether to merge with current or start new
        if not current_merge:
            # First chunk
            current_merge = [chunk]
            current_size = chunk_size
        elif chunk_size > max_chars:
            # Single chunk too large - flush current and handle separately
            if current_merge:
                result_chunks.append("\n\n".join(current_merge))
            result_chunks.append(chunk)  # Add oversized chunk as-is
            current_merge = []
            current_size = 0
        elif force_boundaries and starts_new_section and current_size > 0:
            # This chunk starts a new section - don't merge
            if current_size >= min_chars or len(current_merge) > 1:
                # Current merge is good enough
                result_chunks.append("\n\n".join(current_merge))
                current_merge = [chunk]
                current_size = chunk_size
            else:
                # Current merge too small, but we respect boundaries
                # Check if we can merge without exceeding max
                if current_size + chunk_size + 2 <= max_chars:
                    # Exception: merge if result still under max
                    current_merge.append(chunk)
                    current_size += chunk_size + 2
                else:
                    # Flush small chunk and start new
                    result_chunks.append("\n\n".join(current_merge))
                    current_merge = [chunk]
                    current_size = chunk_size
        elif current_size + chunk_size + 2 > max_chars:
            # Would exceed max size
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
    
    # Post-process: ensure quality
    final_chunks = []
    for chunk in result_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        
        # If chunk is too small and doesn't start with heading, 
        # try to merge with previous
        if (len(chunk) < min_chars and 
            not chunk_starts_with_heading(chunk) and 
            final_chunks and 
            len(final_chunks[-1]) + len(chunk) + 2 <= max_chars):
            final_chunks[-1] = final_chunks[-1] + "\n\n" + chunk
        else:
            final_chunks.append(chunk)
    
    return final_chunks


def analyze_chunk_quality(chunks: List[str], stage: str = "unknown") -> dict:
    """Analyze the quality of prepared chunks."""
    if not chunks:
        return {"error": "No chunks provided"}
    
    sizes = [len(chunk) for chunk in chunks]
    
    # Count chunks with section markers
    chunks_with_sections = sum(1 for chunk in chunks if contains_section_marker(chunk))
    
    # Count chunks starting with headings
    chunks_with_headings = sum(1 for chunk in chunks if chunk_starts_with_heading(chunk))
    
    # Find mixed sections (multiple different Handlungsfelder in one chunk)
    mixed_sections = 0
    for chunk in chunks:
        handlungsfeld_count = chunk.count("Handlungsfeld:")
        if handlungsfeld_count > 1:
            mixed_sections += 1
    
    return {
        "stage": stage,
        "total_chunks": len(chunks),
        "size_stats": {
            "avg": sum(sizes) / len(sizes),
            "min": min(sizes),
            "max": max(sizes),
            "total": sum(sizes)
        },
        "heading_stats": {
            "chunks_with_sections": chunks_with_sections,
            "chunks_with_headings": chunks_with_headings,
            "chunks_starting_with_heading": chunks_with_headings
        },
        "mixed_sections": mixed_sections,
        "size_distribution": {
            "<10k": sum(1 for s in sizes if s < 10000),
            "10k-15k": sum(1 for s in sizes if 10000 <= s < 15000),
            "15k-20k": sum(1 for s in sizes if 15000 <= s < 20000),
            ">20k": sum(1 for s in sizes if s > 20000),
        }
    }