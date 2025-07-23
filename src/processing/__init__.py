"""Document processing functionality for CURATE."""

from .chunker import (
    chunk_for_embedding,
    chunk_for_llm,
    smart_chunk,  # Backward compatibility
    prepare_llm_chunks,  # Backward compatibility
    analyze_chunk_quality,
    contains_indicator_context,
    extract_chunk_topic,
    INDICATOR_PATTERNS,
)
from .embedder import (
    embed_chunks,
    query_chunks,
    get_all_chunks_for_document,
)
from .parser import extract_text_with_ocr_fallback

__all__ = [
    # Parser exports
    "extract_text_with_ocr_fallback",
    # Embedder exports
    "embed_chunks",
    "query_chunks", 
    "get_all_chunks_for_document",
    # Chunker exports
    "chunk_for_embedding",
    "chunk_for_llm",
    "smart_chunk",
    "prepare_llm_chunks",
    "analyze_chunk_quality",
    "contains_indicator_context",
    "extract_chunk_topic",
    "INDICATOR_PATTERNS",
]