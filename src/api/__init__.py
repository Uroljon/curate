"""API module for CURATE."""

from .routes import upload_pdf, extract_structure, extract_structure_fast

__all__ = [
    "upload_pdf",
    "extract_structure", 
    "extract_structure_fast",
]