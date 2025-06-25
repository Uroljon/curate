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
import re

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
            # Try raw first
            data = json5.loads(llm_response)
        except Exception:
            # Try to extract the JSON array from noisy response
            try:
                match = re.search(r"\[.*\]", llm_response, re.DOTALL)
                if match:
                    json_part = match.group(0)
                    data = json5.loads(json_part)
                else:
                    raise ValueError("No JSON array found in LLM response.")
            except Exception as e:
                print("❌ Parsing failed")
                print("Raw LLM response:", llm_response)
                print("Error:", e)
                data = None

        # If valid structured data found
        if isinstance(data, list) and data:
            results.extend(data)

    return JSONResponse(content={"structures": results})

