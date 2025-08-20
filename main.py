# main.py
import os

# Set this before importing transformers/sentence-transformers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI

from src.api import (
    extract_enhanced,
    extract_enhanced_operations,
    upload_pdf,
)

app = FastAPI()

# Register routes
app.post("/upload")(upload_pdf)
app.get("/extract_enhanced")(extract_enhanced)
app.get("/extract_enhanced_operations")(extract_enhanced_operations)
