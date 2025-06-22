# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
from parser import extract_text_with_ocr_fallback
from llm import query_ollama

app = FastAPI()

UPLOAD_FOLDER = "uploads"

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    extracted_text = extract_text_with_ocr_fallback(file_path)

    short_text = extracted_text[:3000]  # Truncate for now

    # Query LLM for summary
    summary_prompt = f"Summarize the following document content:\n\n{short_text}"
    summary = query_ollama(summary_prompt)

    return JSONResponse(content={
        "summary": summary,
        "text": short_text
    })
