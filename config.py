# config.py
"""
Central configuration file for CURATE PDF extraction system.
All tunable parameters should be defined here.
"""

# Model Configuration
MODEL_NAME = "qwen2.5:7b"  # Options: "qwen2.5:7b", "qwen2.5:14b", "llama3:8b", etc.
MODEL_TEMPERATURE = 0.0  # 0.0 for deterministic output
MODEL_TIMEOUT = 180  # seconds

# API Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

# Chunk Configuration
CHUNK_MAX_CHARS = 20000  # Maximum characters per chunk
CHUNK_MIN_CHARS = 15000  # Minimum characters per chunk
CHUNK_WARNING_THRESHOLD = 25000  # Warn if chunk exceeds this size

# LLM Generation Configuration
GENERATION_OPTIONS = {
    "temperature": 0.3,  # For non-structured generation
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.05,
    "num_predict": 800,
    "stop": ["```", "</json>"],
}

# Structured Output Configuration
STRUCTURED_OUTPUT_OPTIONS = {
    "temperature": MODEL_TEMPERATURE,
    "top_p": 0.9,
    "num_predict": 2000,  # Larger for structured output
    "stop": ["</json>", "```"],
}

# Extraction Configuration
EXTRACTION_MAX_RETRIES = 1  # Number of retry attempts for failed extractions

# Text Processing Configuration
MIN_CHARS_FOR_VALID_PAGE = 10  # Minimum characters to consider page as text (not scanned)
OCR_LANGUAGE = "deu"  # Tesseract language code for German
SPELL_CHECK_THRESHOLD = 0.6  # Ratio of misspelled words to filter OCR noise
SYMBOL_FILTER_THRESHOLD = 0.3  # Minimum ratio of letters in a line to keep it
SUPPORTED_LANGUAGES = ["en", "de"]  # Languages to accept in OCR text
SPELL_CHECK_LANGUAGES = {"de": "de", "en": "en"}  # Language codes for spell checker

# ChromaDB Configuration
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "document_chunks"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Upload Configuration
UPLOAD_FOLDER = "uploads"

# Semantic Chunking Configuration
SEMANTIC_CHUNK_MAX_CHARS = 5000  # Target size for initial chunks
SEMANTIC_CHUNK_MIN_CHARS = 3000  # Minimum size before merging

# Output Configuration
OUTPUT_FOLDER = "outputs"