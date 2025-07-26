# main.py
import os

# Set this before importing transformers/sentence-transformers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI

from src.api import extract_structure, extract_structure_fast, upload_pdf, upload_pdf_with_pages

app = FastAPI()

# Register routes
app.post("/upload")(upload_pdf)
app.post("/upload_with_pages")(upload_pdf_with_pages)
app.get("/extract_structure")(extract_structure)
app.get("/extract_structure_fast")(extract_structure_fast)
