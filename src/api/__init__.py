"""API module for CURATE."""

from .routes import (
    extract_enhanced,
    extract_enhanced_operations,
    upload_pdf,
)

__all__ = [
    "extract_enhanced",
    "extract_enhanced_operations",
    "upload_pdf",
]
