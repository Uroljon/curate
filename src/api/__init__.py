"""API module for CURATE."""

from .routes import extract_structure, extract_structure_fast, upload_pdf

__all__ = [
    "extract_structure",
    "extract_structure_fast",
    "upload_pdf",
]
