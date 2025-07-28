# config.py
"""
Central configuration file for CURATE PDF extraction system.
All tunable parameters should be defined here.
"""

import os
from pathlib import Path

# Get the project root directory (where config.py is located)
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Goes up to curate/

# Model Configuration
MODEL_NAME = "qwen3:14b"  # Options: "qwen2.5:7b", "qwen2.5:14b", "llama3:8b", etc.
MODEL_TEMPERATURE = 0.2  # Research-backed: 0.2-0.3 for PDF extraction (balances determinism with flexibility)
MODEL_TIMEOUT = 600  # seconds (10 minutes for larger models)

# API Configuration

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
OLLAMA_API_URL = f"http://{OLLAMA_HOST}/api/generate"
OLLAMA_CHAT_URL = f"http://{OLLAMA_HOST}/api/chat"

# Chunk Configuration - Optimized to prevent LLM output truncation
CHUNK_MAX_CHARS = (
    15000  # Maximum characters per chunk for LLM (reduced to prevent JSON truncation)
)
CHUNK_MIN_CHARS = 12000  # Minimum characters per chunk for LLM
CHUNK_WARNING_THRESHOLD = 20000  # Warn if chunk exceeds this size

# Semantic Chunk Configuration (for initial document chunking)
SEMANTIC_CHUNK_MAX_CHARS = 7500  # Maximum characters per semantic chunk
SEMANTIC_CHUNK_TARGET_CHARS = 5000  # Target size for semantic chunks
SEMANTIC_CHUNK_MIN_CHARS = 1000  # Minimum characters per semantic chunk

# Structured Output Configuration - Research-backed optimal parameters
STRUCTURED_OUTPUT_OPTIONS = {
    # Note: temperature is not included here, use MODEL_TEMPERATURE when calling Ollama
    "top_p": 0.4,  # Research shows 0.3-0.5 optimal for factual extraction
    "top_k": 40,  # Focused candidate set for reliable JSON generation
    "num_predict": 3000,  # Larger for complex German municipal documents
    "num_ctx": 40960,  # Increased context for qwen3 to prevent truncation errors
    "stop": ["</json>", "```"],
    "keep_alive": "24h",  # Keep model loaded in memory for 24 hours
}

# Extraction Configuration
EXTRACTION_MAX_RETRIES = 3  # Number of retry attempts for failed extractions

# Text Processing Configuration
MIN_CHARS_FOR_VALID_PAGE = (
    10  # Minimum characters to consider page as text (not scanned)
)
OCR_LANGUAGE = "deu"  # Tesseract language code for German
SPELL_CHECK_THRESHOLD = 0.6  # Ratio of misspelled words to filter OCR noise
SYMBOL_FILTER_THRESHOLD = 0.3  # Minimum ratio of letters in a line to keep it
SUPPORTED_LANGUAGES = ["en", "de"]  # Languages to accept in OCR text
SPELL_CHECK_LANGUAGES = {"de": "de", "en": "en"}  # Language codes for spell checker

# ChromaDB Configuration
CHROMA_DIR = str(PROJECT_ROOT / "data" / "chroma_store")
COLLECTION_NAME = "document_chunks"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Upload Configuration
UPLOAD_FOLDER = str(PROJECT_ROOT / "data" / "uploads")

# Logging Configuration
LOG_DIR = str(PROJECT_ROOT / "logs")

# Indicator Configuration
INDICATOR_AWARE_CHUNKING = True  # Preserve indicators with their context
INDICATOR_WINDOW_SIZE = 150  # Characters to check around split points

# Fast Extraction Configuration
FAST_EXTRACTION_ENABLED = True  # Enable fast single-pass extraction endpoint
FAST_EXTRACTION_MAX_CHUNKS = 50  # Limit chunks for speed (0 = no limit)

# Source Attribution Configuration
QUOTE_MATCH_THRESHOLD = 0.4  # Minimum score for fuzzy quote matching (0-1)
MIN_QUOTE_LENGTH = 15  # Minimum characters for a quote to be considered for matching

# Aggregation Configuration
AGGREGATION_CHUNK_SIZE = (
    15  # Number of action fields to process in each aggregation batch
)
