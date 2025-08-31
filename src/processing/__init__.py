"""Document processing functionality for CURATE."""

from .chunker import chunk_for_llm_with_pages
from .parser import extract_text_with_ocr_fallback

__all__ = sorted(
    [
        # Chunker exports
        "chunk_for_llm_with_pages",
        # Parser exports
        "extract_text_with_ocr_fallback",
    ]
)
