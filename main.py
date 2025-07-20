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

from structure_extractor import build_structure_prompt, prepare_llm_chunks, extract_structures_with_retry
from llm import query_ollama
import json5
import re

@app.get("/extract_structure")
async def extract_structure(source_id: str, max_chars: int = 12000, min_chars: int = 8000):
    chunks = query_chunks("irrelevant", top_k=1000, source_id=source_id)
    raw_texts = [c["text"] for c in chunks]
    optimized_chunks = prepare_llm_chunks(raw_texts, max_chars=max_chars, min_chars=min_chars)

    all_extracted_data = []

    for i, chunk in enumerate(optimized_chunks):
        print(f"🔄 Processing chunk {i+1}/{len(optimized_chunks)} with {len(chunk)} characters...")
        
        # Use new retry-based extraction
        chunk_data = extract_structures_with_retry(chunk)
        
        if chunk_data:
            all_extracted_data.extend(chunk_data)
            print(f"✅ Chunk {i+1} yielded {len(chunk_data)} action fields")
        else:
            print(f"⚠️ Chunk {i+1} yielded no valid data")

    print(f"📊 Total extracted data: {len(all_extracted_data)} action fields from {len(optimized_chunks)} chunks")
    
    if not all_extracted_data:
        print("⚠️ No valid data extracted from any chunks")
        return JSONResponse(content={"structures": []})
    
    # Deduplicate action fields by name (merge projects from same action field)
    deduplicated_data = {}
    
    for item in all_extracted_data:
        action_field = item["action_field"]
        
        if action_field in deduplicated_data:
            # Merge projects from duplicate action fields
            existing_projects = deduplicated_data[action_field]["projects"]
            new_projects = item["projects"]
            
            # Simple deduplication by project title
            existing_titles = {p["title"] for p in existing_projects}
            for project in new_projects:
                if project["title"] not in existing_titles:
                    existing_projects.append(project)
        else:
            deduplicated_data[action_field] = item
    
    # Convert back to list format
    final_structures = list(deduplicated_data.values())
    
    print(f"✅ Final result: {len(final_structures)} unique action fields")
    for structure in final_structures:
        print(f"   📋 {structure['action_field']}: {len(structure['projects'])} projects")
    
    return JSONResponse(content={"structures": final_structures})