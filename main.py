# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
from parser import extract_text_with_ocr_fallback
from llm import query_ollama
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
    matches = query_chunks("What is the main conclusion?")

    for match in matches:
        print(match["text"], f"(score: {match['score']})")

    # Optional: return chunks (shortened) in API response for now
    return JSONResponse(content={
        "chunks": [c for c in chunks[:5]],
        "matches": matches
    })


    short_text = extracted_text[:3000]  # Truncate for now

    # Query LLM for summary
    summary_prompt = f"Summarize the following document content:\n\n{short_text}"
    summary = query_ollama(summary_prompt)

    return JSONResponse(content={
        "summary": summary,
        "text": short_text
    })
