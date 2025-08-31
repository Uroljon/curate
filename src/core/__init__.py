"""Core functionality for CURATE."""

# Chunking configuration
# Model configuration
# Storage configuration
# Extraction configuration
# Text processing configuration
from .config import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_WARNING_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    EXTRACTION_MAX_RETRIES,
    FAST_EXTRACTION_ENABLED,
    FAST_EXTRACTION_MAX_CHUNKS,
    INDICATOR_AWARE_CHUNKING,
    INDICATOR_WINDOW_SIZE,
    LOG_DIR,
    MIN_CHARS_FOR_VALID_PAGE,
    MODEL_NAME,
    MODEL_TEMPERATURE,
    MODEL_TIMEOUT,
    OCR_LANGUAGE,
    OLLAMA_API_URL,
    OLLAMA_CHAT_URL,
    SEMANTIC_CHUNK_MAX_CHARS,
    SEMANTIC_CHUNK_MIN_CHARS,
    SEMANTIC_CHUNK_TARGET_CHARS,
    SPELL_CHECK_LANGUAGES,
    SPELL_CHECK_THRESHOLD,
    STRUCTURED_OUTPUT_OPTIONS,
    SUPPORTED_LANGUAGES,
    SYMBOL_FILTER_THRESHOLD,
    UPLOAD_FOLDER,
    USE_CHAIN_OF_THOUGHT,
)

# LLM functionality
from .llm import query_ollama_structured, query_ollama_with_thinking_mode

# Data schemas
from .schemas import (
    ActionField,
    ActionFieldList,
    ExtractionResult,
    Project,
    ProjectDetails,
    ProjectDetailsEnhanced,
    ProjectList,
)

__all__ = [
    "CHUNK_MAX_CHARS",
    "CHUNK_MIN_CHARS",
    "CHUNK_WARNING_THRESHOLD",
    "CONFIDENCE_THRESHOLD",
    "EXTRACTION_MAX_RETRIES",
    "FAST_EXTRACTION_ENABLED",
    "FAST_EXTRACTION_MAX_CHUNKS",
    "INDICATOR_AWARE_CHUNKING",
    "INDICATOR_WINDOW_SIZE",
    "LOG_DIR",
    "MIN_CHARS_FOR_VALID_PAGE",
    "MODEL_NAME",
    "MODEL_TEMPERATURE",
    "MODEL_TIMEOUT",
    "OCR_LANGUAGE",
    "OLLAMA_API_URL",
    "OLLAMA_CHAT_URL",
    "SEMANTIC_CHUNK_MAX_CHARS",
    "SEMANTIC_CHUNK_MIN_CHARS",
    "SEMANTIC_CHUNK_TARGET_CHARS",
    "SPELL_CHECK_LANGUAGES",
    "SPELL_CHECK_THRESHOLD",
    "STRUCTURED_OUTPUT_OPTIONS",
    "SUPPORTED_LANGUAGES",
    "SYMBOL_FILTER_THRESHOLD",
    "UPLOAD_FOLDER",
    "USE_CHAIN_OF_THOUGHT",
    "ActionField",
    "ActionFieldList",
    "ExtractionResult",
    "Project",
    "ProjectDetails",
    "ProjectDetailsEnhanced",
    "ProjectList",
    "query_ollama_structured",
    "query_ollama_with_thinking_mode",
]
