"""Document processing functionality for CURATE."""

from .parser import extract_text_with_ocr_fallback

__all__ = sorted(
    [
        # Chunker exports
        "INDICATOR_PATTERNS",
        "analyze_chunk_quality",
        "chunk_for_embedding",
        "chunk_for_embedding_enhanced",
        "chunk_for_embedding_with_pages",
        "chunk_for_llm",
        "chunk_for_llm_with_pages",
        "contains_indicator_context",
        "extract_chunk_topic",
        # Parser exports
        "extract_text_legacy",
        "extract_text_with_ocr_fallback",
    ]
)
