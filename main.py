# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
from parser import extract_text_with_ocr_fallback
from embedder import embed_chunks, query_chunks
from semantic_chunker import smart_chunk

app = FastAPI()
UPLOAD_FOLDER = "uploads"

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    extracted_text = extract_text_with_ocr_fallback(file_path)
    chunks = smart_chunk(extracted_text)

    # Store embeddings
    embed_chunks(chunks, source_id=file.filename)

    # send chunk length as proof
    return JSONResponse(content={
        "chunks": len(chunks)
    })

from structure_extractor import build_structure_prompt, prepare_llm_chunks
from llm import query_ollama
import json5

@app.get("/extract_structure")
async def extract_structure(source_id: str, max_chars: int = 30000, min_chars: int = 8000):
    chunks = query_chunks("irrelevant", top_k=1000, source_id=source_id)
    raw_texts = [c["text"] for c in chunks]
    optimized_chunks = prepare_llm_chunks(raw_texts, max_chars=max_chars, min_chars=min_chars)
    results = []

    for chunk in optimized_chunks:
        prompt = build_structure_prompt(chunk)

        print(f"Processing chunk with {len(chunk)} length...")

        llm_response = query_ollama(prompt)

        try:
            data = json5.loads(llm_response)  # Use json5 here instead of json.loads
            # Since your expected format is a JSON array, you may want to check accordingly:
            if isinstance(data, list) and data:  # Non-empty list
                results.extend(data)  # Append all items from this chunk's extraction
        except Exception as e:
            print("Invalid JSON from LLM:", llm_response)
            print("Parsing error:", e)

    return JSONResponse(content={"structures": results})

