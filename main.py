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

    raw_outputs = []

    for chunk in optimized_chunks:
        prompt = build_structure_prompt(chunk)
        print(f"Processing chunk with {len(chunk)} characters...")
        llm_response = query_ollama(prompt)
        # Extract only the JSON array part using regex
        match = re.search(r"{.*}", llm_response, re.DOTALL)
        if match:
            cleaned = match.group(0)
            raw_outputs.append(cleaned)
        else:
            print("⚠️ No valid JSON array found in response:")
            print(llm_response)
        raw_outputs.append(llm_response)

    # Combine all raw outputs
    combined_raw_json = "\n".join(raw_outputs)

    print(combined_raw_json)

    # Ask LLM to fix & deduplicate
    final_prompt = f"""
Below is a series of potentially malformed or partially valid JSON fragments extracted from a German municipality's strategic plan.

Please:
1. Merge all valid JSON fragments into a single **JSON array**.
2. Fix any syntax issues (trailing commas, missing brackets, etc).
3. Merge any duplicate `action_field` entries based on their name.
4. Return only valid JSON — no extra commentary.

Fragments:
{combined_raw_json.strip()}
"""

    fixed_response = query_ollama(final_prompt)

    # 🧹 Clean up extra text around JSON
    match = re.search(r"\{.*\}", fixed_response, re.DOTALL)
    if match:
        cleaned_json = match.group(0)
    else:
        print("⚠️ Could not find valid JSON object in final response:")
        print(fixed_response)
        return JSONResponse(content={"structures": []}, status_code=500)

    try:
        final_data = json5.loads(cleaned_json)
        if isinstance(final_data, list):
            return JSONResponse(content={"structures": final_data})
        else:
            print("⚠️ Unexpected JSON structure (not a list):", final_data)
            return JSONResponse(content={"structures": []}, status_code=500)
    except Exception as e:
        print("❌ Final JSON parsing failed:", e)
        print("Cleaned JSON:", cleaned_json)
        return JSONResponse(content={"structures": []}, status_code=500)