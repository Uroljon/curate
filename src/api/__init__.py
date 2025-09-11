"""API module for CURATE."""

from .routes import (
    extract_enhanced,
    extract_enhanced_operations,
    extract_incremental,
    compute_diff,
    upload_pdf,
)

__all__ = [
    "extract_enhanced",
    "extract_enhanced_operations",
    "extract_incremental",
    "compute_diff",
    "upload_pdf",
]
