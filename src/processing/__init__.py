"""Document processing functionality for CURATE."""

from .parser import extract_text_with_ocr_fallback

__all__ = sorted(
    [
        # Chunker exports
        "INDICATOR_PATTERNS",
        "chunk_for_embedding",
        "chunk_for_embedding_with_pages",
        "chunk_for_llm_with_pages",
        "contains_indicator_context",
        # Parser exports
        "extract_text_legacy",
        "extract_text_with_ocr_fallback",
    ]
)
