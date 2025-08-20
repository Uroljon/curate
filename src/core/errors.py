"""
Custom exceptions for CURATE.

This module defines all custom exceptions used throughout the codebase
to provide clear error handling and debugging information.
"""

from typing import Any


class CurateError(Exception):
    """Base exception for all CURATE-specific errors."""

    pass


class ExtractionError(CurateError):
    """Raised when extraction process fails."""

    def __init__(
        self,
        message: str,
        stage: str | None = None,
        chunk_index: int | None = None,
        source_id: str | None = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.chunk_index = chunk_index
        self.source_id = source_id


class OCRError(CurateError):
    """Raised when OCR processing fails."""

    def __init__(
        self,
        message: str,
        page_number: int | None = None,
        confidence: float | None = None,
    ):
        super().__init__(message)
        self.page_number = page_number
        self.confidence = confidence


class LLMError(CurateError):
    """Raised when LLM communication fails."""

    def __init__(
        self,
        message: str,
        model: str | None = None,
        prompt_size: int | None = None,
        response_code: int | None = None,
    ):
        super().__init__(message)
        self.model = model
        self.prompt_size = prompt_size
        self.response_code = response_code


# Helper functions for consistent error messages
