"""Core functionality for CURATE."""

# Chunking configuration
# Model configuration
# Storage configuration
# Extraction configuration
# Text processing configuration
from .config import (
    CHROMA_DIR,
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_WARNING_THRESHOLD,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
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
)

# LLM functionality
from .llm import query_ollama_structured

# Data schemas
from .schemas import (
    ActionField,
    ActionFieldList,
    ExtractionResult,
    Project,
    ProjectDetails,
    ProjectList,
)

__all__ = [
    # Storage configuration
    "CHROMA_DIR",
    # Chunking configuration
    "CHUNK_MAX_CHARS",
    "CHUNK_MIN_CHARS",
    "CHUNK_WARNING_THRESHOLD",
    "COLLECTION_NAME",
    "EMBEDDING_MODEL",
    # Extraction configuration
    "EXTRACTION_MAX_RETRIES",
    "FAST_EXTRACTION_ENABLED",
    "FAST_EXTRACTION_MAX_CHUNKS",
    "INDICATOR_AWARE_CHUNKING",
    "INDICATOR_WINDOW_SIZE",
    "LOG_DIR",
    # Text processing configuration
    "MIN_CHARS_FOR_VALID_PAGE",
    # Model configuration
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
    # Schema exports
    "ActionField",
    "ActionFieldList",
    "ExtractionResult",
    "Project",
    "ProjectDetails",
    "ProjectList",
    # LLM exports
    "query_ollama_structured",
]
