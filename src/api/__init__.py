"""API module for CURATE."""

from .routes import extract_structure, upload_pdf, enhance_structure

__all__ = [
    "extract_structure",
    "upload_pdf",
    "enhance_structure",
]
