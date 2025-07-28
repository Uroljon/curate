"""API module for CURATE."""

from .routes import extract_structure, upload_pdf

__all__ = [
    "extract_structure",
    "upload_pdf",
]
