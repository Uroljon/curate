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
# LLM Backend Selection ('ollama', 'vllm', 'openai', 'openrouter', or 'gemini')
LLM_BACKEND = os.getenv("LLM_BACKEND", "openrouter")

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

# External API Configuration (OpenAI, Gemini, etc.)
EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", "")  # OpenAI API key or Gemini API key
EXTERNAL_BASE_URL = os.getenv("EXTERNAL_BASE_URL", None)  # Custom base URL (optional)
EXTERNAL_MODEL_NAME = os.getenv("EXTERNAL_MODEL_NAME", "gpt-4o")  # Default to GPT-4o
EXTERNAL_MAX_TOKENS = int(
    os.getenv("EXTERNAL_MAX_TOKENS", "4096")
)  # Conservative default

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")  # OpenRouter API key
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "openai/o4-mini-high")  # Default model
OPENROUTER_MAX_TOKENS = int(
    os.getenv("OPENROUTER_MAX_TOKENS", "65536")
)  # Increased default to reduce truncation with large JSON outputs

# Recommended model configurations for different providers
EXTERNAL_MODEL_CONFIGS = {
    "openai": {
        "gpt-4o": {"max_tokens": 4096, "temperature": 0.1},
        "gpt-4o-mini": {"max_tokens": 16384, "temperature": 0.1},
        "gpt-4-turbo": {"max_tokens": 4096, "temperature": 0.1},
        "o1-preview": {
            "max_tokens": 32768,
            "temperature": 1.0,
        },  # o1 models don't support temperature control
        "o1-mini": {"max_tokens": 65536, "temperature": 1.0},
    },
    "gemini": {
        "gemini-2.0-flash-exp": {"max_tokens": 8192, "temperature": 0.1},
        "gemini-1.5-pro": {"max_tokens": 8192, "temperature": 0.1},
        "gemini-1.5-flash": {"max_tokens": 8192, "temperature": 0.1},
    },
    "openrouter": {
        # OpenAI models via OpenRouter
        "openai/gpt-4o": {"max_tokens": 4096, "temperature": 0.1},
        "openai/gpt-4o-mini": {"max_tokens": 16384, "temperature": 0.1},
        "openai/o4-mini-high": {"max_tokens": 100000, "temperature": 0.1},  # 200K context, cheaper and better
        "openai/o1-preview": {"max_tokens": 32768, "temperature": 1.0},
        "openai/o1-mini": {"max_tokens": 65536, "temperature": 1.0},
        # Anthropic models via OpenRouter
        "anthropic/claude-3.5-sonnet": {"max_tokens": 8192, "temperature": 0.1},
        "anthropic/claude-3-haiku": {"max_tokens": 8192, "temperature": 0.1},
        # Google models via OpenRouter
        "google/gemini-2.0-flash-exp": {"max_tokens": 8192, "temperature": 0.1},
        "google/gemini-pro": {"max_tokens": 8192, "temperature": 0.1},
        # Cost-effective alternatives
        "meta-llama/llama-3.1-405b-instruct": {"max_tokens": 4096, "temperature": 0.1},
        "qwen/qwen-2.5-72b-instruct": {"max_tokens": 8192, "temperature": 0.1},
    },
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
elif LLM_BACKEND in ["openai", "gemini", "openrouter"]:
    # External APIs generally have large context windows
    # GPT-4o: 128K context, Gemini 2.0: 2M context, Claude 3.5: 200K context
    # Use conservative sizing for cost optimization
    CHUNK_MAX_CHARS = 18000  # Larger chunks for external APIs with big contexts
    CHUNK_MIN_CHARS = 15000
else:
    CHUNK_MAX_CHARS = 15000  # Larger chunks for Ollama with bigger context windows
    CHUNK_MIN_CHARS = 12000

CHUNK_WARNING_THRESHOLD = 20000  # Warn if chunk exceeds this size

# Enhanced Extraction Configuration (for direct extraction to 4-bucket structure)
if LLM_BACKEND == "vllm":
    ENHANCED_CHUNK_MAX_CHARS = 10000  # Smaller chunks for focused extraction
    ENHANCED_CHUNK_MIN_CHARS = 8000
elif LLM_BACKEND in ["openai", "gemini", "openrouter"]:
    ENHANCED_CHUNK_MAX_CHARS = 12000  # Slightly larger for external APIs
    ENHANCED_CHUNK_MIN_CHARS = 10000
else:
    ENHANCED_CHUNK_MAX_CHARS = 10000  # Conservative for Ollama
    ENHANCED_CHUNK_MIN_CHARS = 8000

ENHANCED_CHUNK_OVERLAP = 0.1  # 10% overlap (reduced from 15%)

# Semantic Chunk Configuration (for initial document chunking)
SEMANTIC_CHUNK_MAX_CHARS = 7500  # Maximum characters per semantic chunk
SEMANTIC_CHUNK_TARGET_CHARS = 5000  # Target size for semantic chunks
SEMANTIC_CHUNK_MIN_CHARS = 1000  # Minimum characters per semantic chunk

# Structured Output Configuration - Research-backed optimal parameters
STRUCTURED_OUTPUT_OPTIONS = {
    # Note: temperature is not included here, use MODEL_TEMPERATURE when calling Ollama
    "top_p": 0.4,  # Research shows 0.3-0.5 optimal for factual extraction
    "top_k": 40,  # Focused candidate set for reliable JSON generation
    "num_predict": 50000,  # Increased to allow full JSON responses without truncation
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

# Entity Resolution Configuration
ENTITY_RESOLUTION_ENABLED = True  # Enable entity resolution to fix node fragmentation
ENTITY_RESOLUTION_SIMILARITY_THRESHOLD = (
    0.65  # Reduced from 0.75 to catch more variations like "und" vs "&"
)
ENTITY_RESOLUTION_AUTO_MERGE_THRESHOLD = (
    0.90  # Auto-merge threshold for very similar entities (0-1)
)
ENTITY_RESOLUTION_RESOLVE_ACTION_FIELDS = True  # Resolve action field duplicates
ENTITY_RESOLUTION_RESOLVE_PROJECTS = True  # Resolve project duplicates

# Extraction Consistency Configuration
EXTRACTION_CONSISTENCY_ENABLED = (
    True  # Enable consistency validation to fix uneven edge distribution
)
EXTRACTION_CONSISTENCY_MIN_MEASURES_PER_PROJECT = 1  # Minimum measures per project
EXTRACTION_CONSISTENCY_MIN_INDICATORS_PER_PROJECT = 1  # Minimum indicators per project
