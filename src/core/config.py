# config.py
"""
Central configuration file for CURATE PDF extraction system.
All tunable parameters should be defined here.
"""

import os
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv not installed, skip loading .env file
    pass

# Get the project root directory (where config.py is located)
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Goes up to curate/

# API Configuration
# LLM Backend Selection ('ollama' or 'vllm')
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# Model Configuration
MODEL_NAME = "qwen3:14b"  # Options: "qwen3:7b", "qwen3:14b", "llama3:8b", etc.
# Temperature settings - different for different backends
# For Qwen3 models: 0.7 for non-thinking mode (JSON), 0.6 for thinking mode
# For other models: 0.2-0.3 for PDF extraction
MODEL_TEMPERATURE = 0.7 if LLM_BACKEND == "vllm" else 0.2
MODEL_TIMEOUT = 600  # seconds (10 minutes for larger models)

# Ollama Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
OLLAMA_API_URL = f"http://{OLLAMA_HOST}/api/generate"
OLLAMA_CHAT_URL = f"http://{OLLAMA_HOST}/api/chat"

# vLLM Configuration
VLLM_HOST = os.getenv("VLLM_HOST", "10.67.142.34:8001")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_MAX_TOKENS = int(
    os.getenv("VLLM_MAX_TOKENS", "15000")
)  # Optimized for 32K context (qwen3:14b-AWQ) - allows large structured outputs

# Model name mappings between Ollama and vLLM
MODEL_MAPPINGS = {
    "qwen3:14b": "Qwen/Qwen3-14B-AWQ",  # Updated to new AWQ model
    "qwen3:7b": "Qwen/Qwen3-7B-Instruct",
    "qwen3:8b": "Qwen/Qwen3-8B-Instruct",
}

# Chunk Configuration - Optimized to prevent LLM output truncation
# Adjust chunk size based on backend
if LLM_BACKEND == "vllm":
    # With 32K context (Qwen3-14B-AWQ), we have much more room:
    # - Input chunk (~3K tokens per 12K chars)
    # - JSON schema + prompts (~1K tokens)
    # - Output JSON (~4-6K tokens)
    # Total: ~8-10K tokens, leaving plenty of headroom
    CHUNK_MAX_CHARS = 12000  # Can use larger chunks with 32K context
    CHUNK_MIN_CHARS = 10000
else:
    CHUNK_MAX_CHARS = 15000  # Larger chunks for Ollama with bigger context windows
    CHUNK_MIN_CHARS = 12000

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
    "num_predict": 30720,  # Increased to 75% of context window to prevent JSON truncation
    "num_ctx": 40960,  # Matches qwen3:14b context length
    "stop": ["</json>", "```"],
    "keep_alive": "24h",  # Keep model loaded in memory for 24 hours
}

# Extraction Configuration
EXTRACTION_MAX_RETRIES = 3  # Number of retry attempts for failed extractions
USE_CHAIN_OF_THOUGHT = (
    True  # Enable Chain-of-Thought prompting for better classification
)
CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence score for classification (0.0-1.0)

# Text Processing Configuration
MIN_CHARS_FOR_VALID_PAGE = (
    50  # Minimum characters to consider page as text (not scanned)
)
OCR_LANGUAGE = "deu"  # Tesseract language code for German
SPELL_CHECK_THRESHOLD = 0.6  # Ratio of misspelled words to filter OCR noise
SYMBOL_FILTER_THRESHOLD = 0.3  # Minimum ratio of letters in a line to keep it
SUPPORTED_LANGUAGES = ["en", "de"]  # Languages to accept in OCR text
SPELL_CHECK_LANGUAGES = {"de": "de", "en": "en"}  # Language codes for spell checker

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
