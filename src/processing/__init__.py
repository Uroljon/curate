"""Document processing functionality for CURATE."""

from .chunker import (
    INDICATOR_PATTERNS,
    analyze_chunk_quality,
    chunk_for_embedding,
    chunk_for_embedding_enhanced,
    chunk_for_llm,
    contains_indicator_context,
    extract_chunk_topic,
    prepare_llm_chunks,  # Backward compatibility
    smart_chunk,  # Backward compatibility
)
from .embedder import (
    embed_chunks,
    get_all_chunks_for_document,
    query_chunks,
)
from .parser import extract_text_with_ocr_fallback

__all__ = [
    "INDICATOR_PATTERNS",
    "analyze_chunk_quality",
    # Chunker exports
    "chunk_for_embedding",
    "chunk_for_embedding_enhanced",
    "chunk_for_llm",
    "contains_indicator_context",
    # Embedder exports
    "embed_chunks",
    "extract_chunk_topic",
    # Parser exports
    "extract_text_with_ocr_fallback",
    "get_all_chunks_for_document",
    "prepare_llm_chunks",
    "query_chunks",
    "smart_chunk",
]
