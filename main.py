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
    
    # TESTING FOR NOW
    # Basic query (optional placeholder)
    # matches = query_chunks("What is the main conclusion?")

    # for match in matches:
    #     print(match["text"], f"(score: {match['score']})")

    # Optional: return chunks (shortened) in API response for now
    # return JSONResponse(content={
    #     "chunks": [c for c in chunks[:5]],
    #     "matches": matches
    # })

from structure_extractor import build_structure_prompt
from llm import query_ollama
import json

@app.get("/extract_structure")
async def extract_structure(source_id: str):
    chunks = query_chunks("irrelevant", top_k=1000, source_id=source_id)
    results = []

    for chunk in chunks:
        prompt = build_structure_prompt(chunk["text"])
        llm_response = query_ollama(prompt)

        try:
            data = json.loads(llm_response)
            if isinstance(data, dict) and data.get("action_field"):
                results.append(data)
        except json.JSONDecodeError:
            print("Invalid JSON from LLM:", llm_response)

    return JSONResponse(content={"structures": results})
