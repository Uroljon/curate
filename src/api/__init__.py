"""API module for CURATE."""

from .routes import enhance_structure, extract_enhanced, extract_structure, upload_pdf

__all__ = [
    "enhance_structure",
    "extract_enhanced",
    "extract_structure",
    "upload_pdf",
]
