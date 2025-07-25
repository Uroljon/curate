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


class ChunkingError(CurateError):
    """Raised when document chunking fails."""

    def __init__(self, message: str, document_path: str | None = None):
        super().__init__(message)
        self.document_path = document_path


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


class ValidationError(CurateError):
    """Raised when data validation fails."""

    def __init__(self, message: str, field_name: str | None = None, value: Any = None):
        super().__init__(message)
        self.field_name = field_name
        self.value = value


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


class EmbeddingError(CurateError):
    """Raised when embedding generation or storage fails."""

    def __init__(
        self, message: str, chunk_id: str | None = None, collection: str | None = None
    ):
        super().__init__(message)
        self.chunk_id = chunk_id
        self.collection = collection


class FileProcessingError(CurateError):
    """Raised when file processing operations fail."""

    def __init__(
        self, message: str, file_path: str | None = None, operation: str | None = None
    ):
        super().__init__(message)
        self.file_path = file_path
        self.operation = operation


class ConfigurationError(CurateError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_key: str | None = None):
        super().__init__(message)
        self.config_key = config_key


class TimeoutError(CurateError):
    """Raised when an operation exceeds its timeout."""

    def __init__(self, message: str, operation: str | None = None, timeout: int | None = None):
        super().__init__(message)
        self.operation = operation
        self.timeout = timeout


class LanguageDetectionError(CurateError):
    """Raised when language detection fails or finds unsupported language."""

    def __init__(
        self,
        message: str,
        detected_language: str | None = None,
        expected_language: str | None = None,
    ):
        super().__init__(message)
        self.detected_language = detected_language
        self.expected_language = expected_language


# Helper functions for consistent error messages
def format_extraction_error(
    stage: str, chunk_index: int, error: Exception
) -> ExtractionError:
    """Create a formatted extraction error."""
    return ExtractionError(
        f"Extraction failed at stage '{stage}' for chunk {chunk_index}: {error!s}",
        stage=stage,
        chunk_index=chunk_index,
    )


def format_llm_error(model: str, error: Exception) -> LLMError:
    """Create a formatted LLM error."""
    return LLMError(f"LLM '{model}' request failed: {error!s}", model=model)


def format_ocr_error(page: int, error: Exception) -> OCRError:
    """Create a formatted OCR error."""
    return OCRError(f"OCR failed on page {page}: {error!s}", page_number=page)
