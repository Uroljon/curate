"""
Simple LLM chunk preparation that respects document structure.

This module provides functions to prepare chunks for LLM processing using
size-based merging while respecting basic structural boundaries.
"""

import re
from typing import List


def prepare_llm_chunks(
    chunks: List[str], 
    max_chars: int = 20000, 
    min_chars: int = 15000
) -> List[str]:
    """
    Prepare chunks for LLM processing using simple size-based merging.
    
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
    current_merge = []
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
    final_chunks = []
    for i, chunk in enumerate(result_chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        
        # If chunk is too small and we have a previous chunk, try to merge
        if (len(chunk) < min_chars and 
            final_chunks and 
            len(final_chunks[-1]) + len(chunk) + 2 <= max_chars):
            # Merge with previous chunk
            final_chunks[-1] = final_chunks[-1] + "\n\n" + chunk
        else:
            final_chunks.append(chunk)
    
    return final_chunks


def analyze_chunk_quality(chunks: List[str], stage: str = "unknown") -> dict:
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
        "size_stats": {
            "avg": sum(sizes) / len(sizes),
            "min": min(sizes),
            "max": max(sizes),
            "total": sum(sizes)
        },
        "structural_stats": {
            "chunks_with_page_markers": chunks_with_pages,
        },
        "size_distribution": {
            "<10k": sum(1 for s in sizes if s < 10000),
            "10k-15k": sum(1 for s in sizes if 10000 <= s < 15000),
            "15k-20k": sum(1 for s in sizes if 15000 <= s < 20000),
            ">20k": sum(1 for s in sizes if s > 20000),
        }
    }