"""Core functionality for CURATE."""

from .config import *
from .llm import query_ollama, query_ollama_structured
from .schemas import (
    ActionField,
    ActionFieldList,
    ExtractionResult,
    Project,
    ProjectDetails,
    ProjectList,
)

__all__ = [
    # Config exports
    "MODEL_NAME",
    "MODEL_TEMPERATURE",
    "MODEL_TIMEOUT",
    "OLLAMA_API_URL",
    "OLLAMA_CHAT_URL",
    "CHUNK_MAX_CHARS",
    "CHUNK_MIN_CHARS",
    "CHUNK_WARNING_THRESHOLD",
    "SEMANTIC_CHUNK_MAX_CHARS",
    "SEMANTIC_CHUNK_TARGET_CHARS",
    "SEMANTIC_CHUNK_MIN_CHARS",
    "GENERATION_OPTIONS",
    "STRUCTURED_OUTPUT_OPTIONS",
    "EXTRACTION_MAX_RETRIES",
    "MIN_CHARS_FOR_VALID_PAGE",
    "OCR_LANGUAGE",
    "SPELL_CHECK_THRESHOLD",
    "SYMBOL_FILTER_THRESHOLD",
    "SUPPORTED_LANGUAGES",
    "SPELL_CHECK_LANGUAGES",
    "CHROMA_DIR",
    "COLLECTION_NAME",
    "EMBEDDING_MODEL",
    "UPLOAD_FOLDER",
    "INDICATOR_AWARE_CHUNKING",
    "INDICATOR_WINDOW_SIZE",
    "FAST_EXTRACTION_ENABLED",
    "FAST_EXTRACTION_MAX_CHUNKS",
    "FAST_EXTRACTION_PARALLEL",
    "RETRIEVAL_MODE",
    "RETRIEVAL_MIN_SCORE",
    "RETRIEVAL_TOP_K",
    "RETRIEVAL_QUERIES",
    "OUTPUT_FOLDER",
    # LLM exports
    "query_ollama",
    "query_ollama_structured",
    # Schema exports
    "ActionField",
    "ActionFieldList", 
    "ExtractionResult",
    "Project",
    "ProjectDetails",
    "ProjectList",
]