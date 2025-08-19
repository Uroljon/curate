# main.py
import os

# Set this before importing transformers/sentence-transformers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI

from src.api import enhance_structure, extract_enhanced, extract_enhanced_operations, extract_structure, upload_pdf

app = FastAPI()

# Register routes
app.post("/upload")(upload_pdf)
app.get("/extract_structure")(extract_structure)
app.get("/enhance_structure")(enhance_structure)
app.get("/extract_enhanced")(extract_enhanced)
app.get("/extract_enhanced_operations")(extract_enhanced_operations)
