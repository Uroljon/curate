"""API module for CURATE."""

from .routes import (
    extract_structure,
    extract_structure_fast,
    upload_pdf,
    upload_pdf_with_pages,
)

__all__ = [
    "extract_structure",
    "extract_structure_fast",
    "upload_pdf",
    "upload_pdf_with_pages",
]
