# main.py
import os

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from embedder import embed_chunks, query_chunks
from parser import extract_text_with_ocr_fallback
from semantic_chunker import smart_chunk
from config import (
    UPLOAD_FOLDER, 
    CHUNK_MAX_CHARS, 
    CHUNK_MIN_CHARS, 
    SEMANTIC_CHUNK_MAX_CHARS,
    FAST_EXTRACTION_ENABLED,
    FAST_EXTRACTION_MAX_CHUNKS
)

app = FastAPI()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    extracted_text = extract_text_with_ocr_fallback(file_path)
    chunks = smart_chunk(extracted_text, max_chars=SEMANTIC_CHUNK_MAX_CHARS)

    # Store embeddings
    embed_chunks(chunks, source_id=file.filename)

    # send chunk length as proof
    return JSONResponse(content={"chunks": len(chunks)})


import re

import json5

from llm import query_ollama
from structure_extractor import (
    build_structure_prompt,
    extract_structures_with_retry,
    extract_with_accumulation,
    prepare_llm_chunks,
    extract_action_fields_only,
    extract_projects_for_field,
    extract_project_details,
)


@app.get("/extract_structure")
async def extract_structure(
    source_id: str, max_chars: int = CHUNK_MAX_CHARS, min_chars: int = CHUNK_MIN_CHARS
):
    chunks = query_chunks("irrelevant", top_k=1000, source_id=source_id)
    raw_texts = [c["text"] for c in chunks]
    optimized_chunks = prepare_llm_chunks(
        raw_texts, max_chars=max_chars, min_chars=min_chars
    )

    # Multi-stage extraction approach
    print(f"\n🚀 Starting multi-stage extraction with {len(optimized_chunks)} chunks\n")
    
    # Stage 1: Extract action fields
    print("=" * 60)
    print("STAGE 1: DISCOVERING ACTION FIELDS")
    print("=" * 60)
    action_fields = extract_action_fields_only(optimized_chunks)
    
    if not action_fields:
        print("⚠️ No action fields found in Stage 1")
        return JSONResponse(content={"structures": []})
    
    # Stage 2 & 3: Extract projects and details for each action field
    all_extracted_data = []
    
    for field_idx, action_field in enumerate(action_fields):
        print(f"\n{'=' * 60}")
        print(f"STAGE 2: EXTRACTING PROJECTS FOR '{action_field}' ({field_idx + 1}/{len(action_fields)})")
        print("=" * 60)
        
        projects = extract_projects_for_field(optimized_chunks, action_field)
        
        if not projects:
            print(f"   ⚠️ No projects found for {action_field}")
            continue
        
        # Stage 3: Extract details for each project
        action_field_data = {
            "action_field": action_field,
            "projects": []
        }
        
        print(f"\n{'=' * 60}")
        print(f"STAGE 3: EXTRACTING DETAILS FOR {len(projects)} PROJECTS")
        print("=" * 60)
        
        for proj_idx, project_title in enumerate(projects):
            print(f"\n📁 Project {proj_idx + 1}/{len(projects)}: {project_title}")
            
            details = extract_project_details(optimized_chunks, action_field, project_title)
            
            project_data = {"title": project_title}
            if details.measures:
                project_data["measures"] = details.measures
            if details.indicators:
                project_data["indicators"] = details.indicators
                
            action_field_data["projects"].append(project_data)
        
        all_extracted_data.append(action_field_data)

    print(
        f"📊 Final extraction result: {len(all_extracted_data)} unique action fields from {len(optimized_chunks)} chunks"
    )

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

    # Count statistics
    total_projects = sum(len(af["projects"]) for af in final_structures)
    measures_count = sum(
        1 for af in final_structures for p in af["projects"] if p.get("measures")
    )
    indicators_count = sum(
        1 for af in final_structures for p in af["projects"] if p.get("indicators")
    )

    print(
        f"✅ Final result: {len(final_structures)} unique action fields, {total_projects} total projects"
    )
    print(
        f"   📊 {measures_count} projects with measures, {indicators_count} projects with indicators"
    )

    for structure in final_structures:
        print(
            f"   📋 {structure['action_field']}: {len(structure['projects'])} projects"
        )

    return JSONResponse(content={"structures": final_structures})


@app.get("/extract_structure_fast")
async def extract_structure_fast(
    source_id: str, max_chars: int = CHUNK_MAX_CHARS, min_chars: int = CHUNK_MIN_CHARS
):
    """Fast single-pass extraction endpoint for quicker iteration and testing."""
    
    if not FAST_EXTRACTION_ENABLED:
        return JSONResponse(
            status_code=503,
            content={"error": "Fast extraction is disabled in configuration"}
        )
    
    import time
    start_time = time.time()
    
    # Get chunks (same as regular extraction)
    chunks = query_chunks("irrelevant", top_k=1000, source_id=source_id)
    raw_texts = [c["text"] for c in chunks]
    optimized_chunks = prepare_llm_chunks(
        raw_texts, max_chars=max_chars, min_chars=min_chars
    )
    
    # Apply chunk limit if configured
    if FAST_EXTRACTION_MAX_CHUNKS > 0:
        optimized_chunks = optimized_chunks[:FAST_EXTRACTION_MAX_CHUNKS]
        print(f"⚡ Fast mode: Processing only first {len(optimized_chunks)} chunks")
    
    print(f"\n🚀 Starting FAST single-pass extraction with {len(optimized_chunks)} chunks\n")
    
    # Single-pass extraction
    all_extracted_data = []
    
    for i, chunk in enumerate(optimized_chunks):
        print(f"⚡ Processing chunk {i+1}/{len(optimized_chunks)} ({len(chunk)} chars)")
        
        # Use the existing single-pass extraction function
        chunk_data = extract_structures_with_retry(chunk)
        
        if chunk_data:
            all_extracted_data.extend(chunk_data)
            print(f"   ✅ Found {len(chunk_data)} action fields")
        else:
            print(f"   ⚠️ No data found")
    
    # Deduplicate action fields (same logic as multi-stage)
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
    
    # Calculate extraction time
    extraction_time = time.time() - start_time
    
    # Count statistics
    total_projects = sum(len(af["projects"]) for af in final_structures)
    measures_count = sum(
        1 for af in final_structures for p in af["projects"] if p.get("measures")
    )
    indicators_count = sum(
        1 for af in final_structures for p in af["projects"] if p.get("indicators")
    )
    
    print(f"\n⏱️  Fast extraction completed in {extraction_time:.2f} seconds")
    print(
        f"✅ Results: {len(final_structures)} action fields, {total_projects} projects"
    )
    print(
        f"📊 {measures_count} projects with measures, {indicators_count} projects with indicators"
    )
    
    return JSONResponse(content={
        "structures": final_structures,
        "metadata": {
            "extraction_time_seconds": round(extraction_time, 2),
            "chunks_processed": len(optimized_chunks),
            "action_fields_found": len(final_structures),
            "total_projects": total_projects,
            "projects_with_measures": measures_count,
            "projects_with_indicators": indicators_count
        }
    })
