"""API routes for CURATE."""

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from src.core import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    FAST_EXTRACTION_ENABLED,
    FAST_EXTRACTION_MAX_CHUNKS,
    UPLOAD_FOLDER,
)
from src.extraction import extract_structures_with_retry
from src.processing import extract_text_with_ocr_fallback
from src.utils import (
    ChunkQualityMonitor,
    get_extraction_monitor,
    log_api_request,
    log_api_response,
)

from .extraction_helpers import (
    add_source_attributions,
    aggregate_extraction_results,
    deduplicate_extraction_results,
    transform_to_enhanced_structure,
)


def load_pages_from_file(source_id: str) -> list[tuple[str, int]]:
    """Load page-aware text from the saved pages file."""
    pages_filename = os.path.splitext(source_id)[0] + "_pages.txt"
    pages_path = os.path.join(UPLOAD_FOLDER, pages_filename)

    if not os.path.exists(pages_path):
        error_msg = f"Pages file not found: {pages_filename}"
        raise FileNotFoundError(error_msg)

    page_aware_text = []
    with open(pages_path, encoding="utf-8") as f:
        content = f.read()

    # Parse pages using regex
    page_pattern = re.compile(
        r"\[Page (\d+)\]\n(.*?)(?=\n\n\[Page|\Z)",
        re.DOTALL,
    )
    matches = page_pattern.findall(content)

    for page_num_str, page_text in matches:
        try:
            page_num = int(page_num_str)
            page_text = page_text.strip()
            if page_text:  # Only add non-empty pages
                page_aware_text.append((page_text, page_num))
        except ValueError:
            print(f"⚠️ Skipping invalid page number: {page_num_str}")

    return page_aware_text


async def upload_pdf(request: Request, file: UploadFile):
    """Handle PDF upload and processing with page attribution support."""
    start_time = time.time()

    # Log API request
    log_api_request("/upload", "POST", {"filename": file.filename}, request.client.host)

    # Get monitor for this extraction
    monitor = get_extraction_monitor(file.filename)

    try:
        # Stage 1: File upload
        monitor.start_stage("file_upload", filename=file.filename, size=file.size)

        # Generate unique filename to prevent race conditions
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        safe_filename = f"{timestamp}_{unique_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        monitor.end_stage("file_upload", file_size=len(content))

        # Stage 2: Page-aware text extraction
        monitor.start_stage("text_extraction")
        page_aware_text, extraction_metadata = extract_text_with_ocr_fallback(file_path)

        # Save page-aware text as .txt file (for debugging) with safe filename
        txt_filename = os.path.splitext(safe_filename)[0] + "_pages.txt"
        txt_path = os.path.join(UPLOAD_FOLDER, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            for text, page_num in page_aware_text:
                txt_file.write(f"\n\n[Page {page_num}]\n\n")
                txt_file.write(text)

        total_text_length = sum(len(text) for text, _ in page_aware_text)
        monitor.end_stage(
            "text_extraction",
            text_length=total_text_length,
            total_pages=extraction_metadata["total_pages"],
            pages_with_content=extraction_metadata["pages_with_content"],
            ocr_pages=extraction_metadata["ocr_pages"],
            native_pages=extraction_metadata["native_pages"],
            ocr_percentage=extraction_metadata["extraction_method_ratio"][
                "ocr_percentage"
            ],
        )

        # No more semantic chunking - pages are already saved to file for later retrieval

        response_data = {
            "pages_extracted": len(page_aware_text),
            "total_text_length": total_text_length,
            "page_attribution_enabled": True,
            "extraction_metadata": extraction_metadata,
            "source_id": safe_filename,  # Return the safe filename for subsequent API calls
            "original_filename": file.filename,  # Keep original filename for reference
        }

        # Log successful completion
        monitor.finalize({"upload_result": response_data})

        # Log API response
        response_time = time.time() - start_time
        log_api_response("/upload", 200, response_time)

        return JSONResponse(content=response_data)

    except FileNotFoundError as e:
        monitor.log_error("upload", e)
        response_time = time.time() - start_time
        log_api_response("/upload", 404, response_time)
        raise
    except PermissionError as e:
        monitor.log_error("upload", e)
        response_time = time.time() - start_time
        log_api_response("/upload", 403, response_time)
        raise
    except OSError as e:
        monitor.log_error("upload", e)
        response_time = time.time() - start_time
        log_api_response("/upload", 500, response_time)
        raise
    except Exception as e:
        # Final fallback for unexpected errors
        monitor.log_error("upload", e)
        response_time = time.time() - start_time
        log_api_response("/upload", 500, response_time)
        raise


async def extract_structure(
    request: Request,
    source_id: str,
    max_chars: int = CHUNK_MAX_CHARS,
    min_chars: int = CHUNK_MIN_CHARS,
):
    """Structure extraction endpoint with page attribution support."""

    start_time = time.time()

    # Log API request
    log_api_request(
        "/extract_structure",
        "GET",
        {"source_id": source_id, "max_chars": max_chars, "min_chars": min_chars},
        request.client.host,
    )

    if not FAST_EXTRACTION_ENABLED:
        log_api_response("/extract_structure", 503, time.time() - start_time)
        return JSONResponse(
            status_code=503,
            content={"error": "Extraction is disabled in configuration"},
        )

    # Get or create monitor
    monitor = get_extraction_monitor(source_id)

    try:
        # Stage 1: Page retrieval
        monitor.start_stage("page_retrieval", source_id=source_id)
        # Load page-aware text from saved file
        page_aware_text = load_pages_from_file(source_id)
        print(f"📄 Loaded {len(page_aware_text)} pages from file")
        monitor.end_stage("page_retrieval", pages_retrieved=len(page_aware_text))

        # Stage 2: LLM chunk preparation
        monitor.start_stage("llm_chunk_preparation")
        # Use page-aware chunking
        from src.processing.chunker import chunk_for_llm_with_pages

        # Extract document title from source_id (remove timestamp and hash)
        doc_title = "Dokument"  # Default title
        if source_id:
            # Try to extract meaningful name from source_id
            # Format: timestamp_hash_filename.pdf
            parts = source_id.split("_")
            if len(parts) >= 3:
                # Skip timestamp and hash, get the filename part
                doc_name = "_".join(parts[2:]).replace(".pdf", "")
                if doc_name:
                    # Clean up the title - replace hyphens with spaces, proper case
                    doc_title = doc_name.replace("-", " ").replace("_", " ")
                    # Capitalize appropriately for German
                    doc_title = " ".join(
                        word.capitalize() for word in doc_title.split()
                    )

        optimized_chunks_with_pages = chunk_for_llm_with_pages(
            page_aware_text,
            max_chars=max_chars,
            min_chars=min_chars,
            doc_title=doc_title,
            add_overlap=True,  # Enable 15% overlap to prevent information loss at boundaries
        )

        # Extract just the text for compatibility with existing code
        optimized_chunks: list[str] = [
            chunk_text for chunk_text, _ in optimized_chunks_with_pages
        ]

        # Analyze LLM chunk quality
        llm_chunk_metrics = ChunkQualityMonitor.analyze_chunks(optimized_chunks, "llm")
        monitor.end_stage(
            "llm_chunk_preparation",
            original_pages=len(page_aware_text),
            optimized_chunks=len(optimized_chunks),
            chunk_metrics=llm_chunk_metrics,
        )

        # Apply chunk limit if configured
        if FAST_EXTRACTION_MAX_CHUNKS > 0:
            optimized_chunks = optimized_chunks[:FAST_EXTRACTION_MAX_CHUNKS]
            optimized_chunks_with_pages = optimized_chunks_with_pages[
                :FAST_EXTRACTION_MAX_CHUNKS
            ]
            print(f"⚡ Processing only first {len(optimized_chunks)} chunks")

        # Save LLM chunks as separate file
        llm_chunks_filename = f"{source_id}_llm_chunks.txt"
        llm_chunks_path = os.path.join(UPLOAD_FOLDER, llm_chunks_filename)
        with open(llm_chunks_path, "w", encoding="utf-8") as llm_chunks_file:
            for i, chunk_text in enumerate(optimized_chunks, 1):
                llm_chunks_file.write(f"\n\n# ====== LLM CHUNK {i} ======\n\n")
                llm_chunks_file.write(chunk_text)

        # Set up LLM dialog log file path
        llm_dialog_filename = f"{source_id}_llm_dialog.txt"
        llm_dialog_path = os.path.join(UPLOAD_FOLDER, llm_dialog_filename)

        print(f"\n🚀 Starting extraction with {len(optimized_chunks)} chunks\n")

        # Stage 3: LLM extraction
        monitor.start_stage("llm_extraction", total_chunks=len(optimized_chunks))

        # Independent extraction per chunk
        all_chunk_results = []
        chunk_timings = []

        for i, chunk_text in enumerate(optimized_chunks):
            chunk_start = time.time()
            print(
                f"⚡ Processing chunk {i+1}/{len(optimized_chunks)} ({len(chunk_text)} chars)"
            )

            # Extract independently from each chunk
            chunk_data = extract_structures_with_retry(
                chunk_text,
                log_file_path=llm_dialog_path,
                log_context=f"Extraction - Chunk {i+1}/{len(optimized_chunks)} ({len(chunk_text)} chars)",
            )

            chunk_time = time.time() - chunk_start
            chunk_timings.append(chunk_time)

            if chunk_data:
                # Add chunk_id to each extracted action field
                for action_field in chunk_data:
                    action_field["_source_chunk_id"] = i
                all_chunk_results.extend(chunk_data)
                print(
                    f"   ✅ Found {len(chunk_data)} action fields in {chunk_time:.2f}s"
                )
            else:
                print(f"   ⚠️ No data found ({chunk_time:.2f}s)")

        # Aggregate all results using separate LLM call
        print(f"\n🔄 Aggregating {len(all_chunk_results)} extracted action fields...")
        aggregated_fields = aggregate_extraction_results(
            all_chunk_results,
            log_file_path=llm_dialog_path,
            log_context_prefix="Extraction Aggregation",
        )

        # Final deduplication to ensure unique action fields
        final_action_fields = deduplicate_extraction_results(aggregated_fields)

        monitor.end_stage(
            "llm_extraction",
            chunks_processed=len(optimized_chunks),
            total_action_fields=len(final_action_fields),
            avg_chunk_time=(
                sum(chunk_timings) / len(chunk_timings) if chunk_timings else 0
            ),
            total_extraction_time=sum(chunk_timings),
        )

        # Stage 4: Source Attribution (if page metadata is available)
        monitor.start_stage("source_attribution")

        # Initialize attribution_stats with default values
        attribution_stats = {
            "total_projects": 0,
            "projects_with_sources": 0,
            "attribution_success_rate": 0.0,
        }

        # Create page-aware LLM chunks for source attribution
        # Create chunks that map LLM chunk IDs to their page numbers
        llm_page_aware_chunks = []
        for chunk_id, (chunk_text, chunk_pages) in enumerate(
            optimized_chunks_with_pages
        ):
            llm_page_aware_chunks.append(
                {"text": chunk_text, "pages": chunk_pages, "chunk_id": chunk_id}
            )

        print(
            f"\n🔍 Adding source attribution using {len(llm_page_aware_chunks)} LLM page-aware chunks..."
        )
        final_structures = add_source_attributions(
            final_action_fields, llm_page_aware_chunks, page_aware_text
        )

        # Count attribution statistics
        attribution_stats["total_projects"] = sum(
            len(af["projects"]) for af in final_structures
        )
        attribution_stats["projects_with_sources"] = sum(
            1
            for af in final_structures
            for project in af["projects"]
            if project.get("sources")
        )
        attribution_stats["attribution_success_rate"] = (
            round(
                attribution_stats["projects_with_sources"]
                / attribution_stats["total_projects"]
                * 100,
                1,
            )
            if attribution_stats["total_projects"] > 0
            else 0.0
        )

        monitor.end_stage(
            "source_attribution",
            page_aware_chunks=len(llm_page_aware_chunks),
            projects_with_sources=attribution_stats["projects_with_sources"],
            total_projects=attribution_stats["total_projects"],
            attribution_success_rate=attribution_stats["attribution_success_rate"],
        )

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

        print(f"\n⏱️  Extraction completed in {extraction_time:.2f} seconds")
        print(
            f"✅ Results: {len(final_structures)} action fields, {total_projects} projects"
        )
        print(
            f"📊 {measures_count} projects with measures, {indicators_count} projects with indicators"
        )

        # Save intermediate extraction JSON for two-layer pipeline
        intermediate_filename = f"{source_id}_intermediate_extraction.json"
        intermediate_path = os.path.join(UPLOAD_FOLDER, intermediate_filename)

        # Prepare intermediate data structure (the "flawed nested JSON")
        intermediate_data = {
            "structures": final_structures,
            "metadata": {
                "extraction_time_seconds": round(extraction_time, 2),
                "chunks_processed": len(optimized_chunks),
                "action_fields_found": len(final_structures),
                "total_projects": total_projects,
                "projects_with_measures": measures_count,
                "projects_with_indicators": indicators_count,
                "page_attribution_enabled": True,
                "projects_with_sources": attribution_stats["projects_with_sources"],
                "attribution_success_rate": round(
                    attribution_stats["attribution_success_rate"], 1
                ),
            },
        }

        # Save intermediate JSON to file
        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump(intermediate_data, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved intermediate extraction to: {intermediate_filename}")

        # Prepare response (same as before but with file path)
        response_data = {
            "structures": final_structures,
            "intermediate_file": intermediate_filename,  # Add file path for pipeline
            "metadata": {
                "extraction_time_seconds": round(extraction_time, 2),
                "chunks_processed": len(optimized_chunks),
                "action_fields_found": len(final_structures),
                "total_projects": total_projects,
                "projects_with_measures": measures_count,
                "projects_with_indicators": indicators_count,
                "page_attribution_enabled": True,
                "projects_with_sources": attribution_stats["projects_with_sources"],
                "attribution_success_rate": round(
                    attribution_stats["attribution_success_rate"], 1
                ),
            },
        }

        # Finalize monitoring
        monitor.finalize(response_data)

        # Log API response
        log_api_response("/extract_structure", 200, extraction_time)

        # Ensure response is JSON serializable
        from fastapi.encoders import jsonable_encoder

        return JSONResponse(content=jsonable_encoder(response_data))

    except FileNotFoundError as e:
        monitor.log_error("extraction", e)
        response_time = time.time() - start_time
        log_api_response("/extract_structure", 404, response_time)
        raise
    except ValueError as e:
        monitor.log_error("extraction", e)
        response_time = time.time() - start_time
        log_api_response("/extract_structure", 400, response_time)
        raise
    except Exception as e:
        # Final fallback for unexpected errors
        monitor.log_error("extraction", e)
        response_time = time.time() - start_time
        log_api_response("/extract_structure", 500, response_time)
        raise


async def enhance_structure(
    request: Request,
    source_id: str,
):
    """
    Enhancement endpoint that transforms intermediate extraction to relational 4-bucket structure.

    This implements the "Transformer LLM" from the two-layer pipeline strategy.
    Takes the intermediate_extraction.json and converts it to the clean relational structure
    with explicit connections and confidence scores.

    Args:
        request: FastAPI request object
        source_id: Source identifier to locate the intermediate extraction file

    Returns:
        Enhanced JSON structure with 4 separate entity buckets and connections
    """
    start_time = time.time()

    # Log API request
    log_api_request(
        "/enhance_structure",
        "GET",
        {"source_id": source_id},
        request.client.host,
    )

    # Get or create monitor for this enhancement
    monitor = get_extraction_monitor(source_id)

    try:
        # Stage 1: Load intermediate extraction file
        monitor.start_stage("file_loading", source_id=source_id)

        intermediate_filename = f"{source_id}_intermediate_extraction.json"
        intermediate_path = os.path.join(UPLOAD_FOLDER, intermediate_filename)

        if not os.path.exists(intermediate_path):
            error_msg = (
                f"Intermediate extraction file not found: {intermediate_filename}"
            )
            raise FileNotFoundError(error_msg)

        with open(intermediate_path, encoding="utf-8") as f:
            intermediate_data = json.load(f)

        # Validate intermediate data structure
        structures = intermediate_data.get("structures", [])
        if not structures:
            msg = "No structures found in intermediate extraction file"
            raise ValueError(msg)

        monitor.end_stage(
            "file_loading",
            structures_loaded=len(structures),
            intermediate_file_size=os.path.getsize(intermediate_path),
        )

        print(f"📄 Loaded intermediate file: {intermediate_filename}")
        print(f"   - {len(structures)} action fields to transform")

        # Stage 2: LLM transformation to enhanced structure
        monitor.start_stage("llm_transformation")

        # Set up LLM dialog log file path
        enhance_dialog_filename = f"{source_id}_enhance_dialog.txt"
        enhance_dialog_path = os.path.join(UPLOAD_FOLDER, enhance_dialog_filename)

        print("\n🔄 Starting structure enhancement transformation...\n")

        # Call the transformer LLM
        enhanced_result = transform_to_enhanced_structure(
            intermediate_data,
            log_file_path=enhance_dialog_path,
            log_context="Structure Enhancement - Two-Layer Pipeline",
        )

        if not enhanced_result:
            msg = "LLM transformation failed - no enhanced structure returned"
            raise ValueError(msg)

        # Stage 2.5: Apply entity resolution and consistency validation
        monitor.start_stage("entity_resolution_and_validation")

        print("\n🔧 Applying entity resolution and consistency validation...")

        # Convert enhanced structure to intermediate format for processing
        structures_for_processing = []

        # Group entities by action field for processing
        proj_lookup = {p.id: p for p in enhanced_result.projects}

        for action_field in enhanced_result.action_fields:
            # Find projects connected to this action field
            connected_projects = []
            for connection in action_field.connections:
                if connection.target_id in proj_lookup:
                    project = proj_lookup[connection.target_id]

                    # Find measures and indicators for this project
                    project_measures = []
                    project_indicators = []

                    for proj_conn in project.connections:
                        # Find connected measures
                        for measure in enhanced_result.measures:
                            if measure.id == proj_conn.target_id:
                                measure_title = measure.content.get("title", "")
                                if measure_title:
                                    project_measures.append(measure_title)
                                break
                        # Find connected indicators
                        for indicator in enhanced_result.indicators:
                            if indicator.id == proj_conn.target_id:
                                indicator_name = indicator.content.get("name", "")
                                if indicator_name:
                                    project_indicators.append(indicator_name)
                                break

                    project_title = project.content.get("title", "")
                    if project_title:
                        connected_projects.append(
                            {
                                "title": project_title,
                                "measures": project_measures,
                                "indicators": project_indicators,
                            }
                        )

            action_field_name = action_field.content.get("name", "")
            if action_field_name:
                structures_for_processing.append(
                    {"action_field": action_field_name, "projects": connected_projects}
                )

        # Apply entity resolution and consistency validation
        from src.api.extraction_helpers import (
            apply_consistency_validation,
            apply_entity_resolution_with_monitoring,
            rebuild_enhanced_structure_from_resolved,
        )

        resolved_structures = apply_entity_resolution_with_monitoring(
            structures_for_processing
        )
        validated_structures = apply_consistency_validation(resolved_structures)

        # Convert back to enhanced structure format with unique ID validation
        enhanced_result = rebuild_enhanced_structure_from_resolved(
            validated_structures, enhanced_result
        )

        monitor.end_stage(
            "entity_resolution_and_validation",
            resolved_entities=len(validated_structures),
        )

        print("✅ Entity resolution and validation completed")

        # Calculate transformation statistics
        total_entities = (
            len(enhanced_result.action_fields)
            + len(enhanced_result.projects)
            + len(enhanced_result.measures)
            + len(enhanced_result.indicators)
        )

        # Count total connections
        total_connections = (
            sum(len(af.connections) for af in enhanced_result.action_fields)
            + sum(len(p.connections) for p in enhanced_result.projects)
            + sum(len(m.connections) for m in enhanced_result.measures)
            + sum(len(i.connections) for i in enhanced_result.indicators)
        )

        monitor.end_stage(
            "llm_transformation",
            total_entities=total_entities,
            action_fields=len(enhanced_result.action_fields),
            projects=len(enhanced_result.projects),
            measures=len(enhanced_result.measures),
            indicators=len(enhanced_result.indicators),
            total_connections=total_connections,
        )

        # Stage 3: Save enhanced structure to file
        monitor.start_stage("file_saving")

        enhanced_filename = f"{source_id}_enhanced_structure.json"
        enhanced_path = os.path.join(UPLOAD_FOLDER, enhanced_filename)

        enhanced_data = enhanced_result.model_dump()
        with open(enhanced_path, "w", encoding="utf-8") as f:
            json.dump(enhanced_data, f, ensure_ascii=False, indent=2)

        monitor.end_stage(
            "file_saving", enhanced_file_size=os.path.getsize(enhanced_path)
        )

        # Calculate total processing time
        processing_time = time.time() - start_time

        print(f"\n⏱️  Enhancement completed in {processing_time:.2f} seconds")
        print(
            f"✅ Results: {total_entities} entities with {total_connections} connections"
        )
        print(f"💾 Saved enhanced structure to: {enhanced_filename}")

        # Prepare response data
        response_data = enhanced_data.copy()
        response_data.update(
            {
                "enhanced_file": enhanced_filename,
                "processing_time_seconds": round(processing_time, 2),
                "transformation_stats": {
                    "total_entities": total_entities,
                    "total_connections": total_connections,
                    "action_fields_count": len(enhanced_result.action_fields),
                    "projects_count": len(enhanced_result.projects),
                    "measures_count": len(enhanced_result.measures),
                    "indicators_count": len(enhanced_result.indicators),
                },
            }
        )

        # Finalize monitoring
        monitor.finalize(response_data)

        # Log API response
        log_api_response("/enhance_structure", 200, processing_time)

        # Return enhanced structure
        return JSONResponse(content=jsonable_encoder(response_data))

    except FileNotFoundError as e:
        monitor.log_error("enhancement", e)
        response_time = time.time() - start_time
        log_api_response("/enhance_structure", 404, response_time)
        raise
    except ValueError as e:
        monitor.log_error("enhancement", e)
        response_time = time.time() - start_time
        log_api_response("/enhance_structure", 400, response_time)
        raise
    except Exception as e:
        # Final fallback for unexpected errors
        monitor.log_error("enhancement", e)
        response_time = time.time() - start_time
        log_api_response("/enhance_structure", 500, response_time)
        raise


async def extract_enhanced(
    request: Request,
    source_id: str,
):
    """
    Consolidated extraction endpoint that goes directly from PDF text to enhanced 4-bucket structure.

    This endpoint implements the streamlined approach:
    1. Reads {source_id}_pages.txt directly
    2. Chunks with smaller windows and minimal overlap
    3. Uses simplified prompts focused on descriptions
    4. Returns EnrichedReviewJSON structure directly

    Benefits:
    - Single API call instead of two-step process
    - Better prompts focused on descriptions from the start
    - No intermediate files
    - Faster processing with single LLM round-trip

    Args:
        request: FastAPI request object
        source_id: Source identifier to locate the page-aware text file

    Returns:
        Enhanced JSON structure with 4 separate entity buckets and connections
    """
    start_time = time.time()

    # Log API request
    log_api_request(
        "/extract_enhanced",
        "GET",
        {"source_id": source_id},
        request.client.host,
    )

    # Get or create monitor for this extraction
    monitor = get_extraction_monitor(source_id)

    try:
        # Stage 1: Load page-aware text file
        monitor.start_stage("file_loading", source_id=source_id)

        # Use existing helper function to load pages
        page_aware_text = load_pages_from_file(source_id)

        if not page_aware_text:
            pages_filename = os.path.splitext(source_id)[0] + "_pages.txt"
            error_msg = f"No valid page content found in {pages_filename}"
            raise ValueError(error_msg)

        print(f"📄 Loaded {len(page_aware_text)} pages from page-aware text file")
        monitor.end_stage("file_loading")

        # Stage 2: Enhanced direct extraction
        monitor.start_stage("enhanced_extraction", source_id=source_id)

        # Import the extraction function
        from src.api.extraction_helpers import extract_direct_to_enhanced

        # Create log file path
        log_file_path = os.path.join(UPLOAD_FOLDER, f"{source_id}_enhanced_extraction.jsonl")

        # Perform direct enhanced extraction
        extraction_result = extract_direct_to_enhanced(
            page_aware_text=page_aware_text,
            source_id=source_id,
            log_file_path=log_file_path,
        )

        if not extraction_result:
            error_msg = "Enhanced extraction failed - no results returned"
            raise RuntimeError(error_msg)

        monitor.end_stage("enhanced_extraction")

        # Stage 3: Save enhanced structure to file
        monitor.start_stage("file_saving", source_id=source_id)

        # Save the enhanced structure JSON file for visualization tool (same format as /enhance_structure)
        enhanced_filename = f"{source_id}_enhanced_structure.json"
        enhanced_path = os.path.join(UPLOAD_FOLDER, enhanced_filename)
        
        # Save the same format as /enhance_structure endpoint for compatibility
        enhanced_data = extraction_result["extraction_result"]
        with open(enhanced_path, "w", encoding="utf-8") as f:
            json.dump(enhanced_data, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved enhanced structure to: {enhanced_filename}")
        monitor.end_stage("file_saving", enhanced_file_size=os.path.getsize(enhanced_path))

        # Stage 4: Prepare response
        monitor.start_stage("response_preparation", source_id=source_id)

        response_data = {
            "source_id": source_id,
            "extraction_result": extraction_result["extraction_result"],
            "enhanced_file": enhanced_filename,  # Add filename for reference
            "metadata": extraction_result["metadata"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_stages": ["file_loading", "enhanced_extraction", "file_saving", "response_preparation"]
        }

        monitor.end_stage("response_preparation")

        # Log successful response
        response_time = time.time() - start_time
        log_api_response("/extract_enhanced", 200, response_time)

        print(f"✅ Enhanced extraction completed in {response_time:.1f}s")

        return JSONResponse(
            content=jsonable_encoder(response_data),
            status_code=200,
            headers={"Content-Type": "application/json; charset=utf-8"}
        )

    except FileNotFoundError as e:
        # File not found errors
        monitor.log_error("file_loading", e)
        response_time = time.time() - start_time
        log_api_response("/extract_enhanced", 404, response_time)

        return JSONResponse(
            content={
                "error": "File not found",
                "detail": str(e),
                "source_id": source_id,
                "suggestion": "Make sure to upload the PDF first using /upload endpoint"
            },
            status_code=404
        )

    except ValueError as e:
        # Invalid data errors
        monitor.log_error("file_loading", e)
        response_time = time.time() - start_time
        log_api_response("/extract_enhanced", 400, response_time)

        return JSONResponse(
            content={
                "error": "Invalid data",
                "detail": str(e),
                "source_id": source_id
            },
            status_code=400
        )

    except RuntimeError as e:
        # Extraction errors
        monitor.log_error("enhanced_extraction", e)
        response_time = time.time() - start_time
        log_api_response("/extract_enhanced", 500, response_time)

        return JSONResponse(
            content={
                "error": "Extraction failed",
                "detail": str(e),
                "source_id": source_id
            },
            status_code=500
        )

    except Exception as e:
        # Final fallback for unexpected errors
        monitor.log_error("enhanced_extraction", e)
        response_time = time.time() - start_time
        log_api_response("/extract_enhanced", 500, response_time)

        return JSONResponse(
            content={
                "error": "Internal server error",
                "detail": str(e),
                "source_id": source_id
            },
            status_code=500
        )


async def extract_enhanced_operations(
    request: Request,
    source_id: str,
):
    """
    Operations-based extraction endpoint that builds state incrementally.

    This endpoint implements the operations-based approach:
    1. Reads {source_id}_pages.txt directly
    2. For each chunk, LLM returns operations (CREATE, UPDATE, MERGE, CONNECT)
    3. Operations are applied deterministically to build final state
    4. No copy degradation or context bloat issues

    Benefits:
    - LLM makes intelligent decisions without managing full state
    - No risk of copy degradation across chunks
    - Smaller LLM responses (operations only)
    - Clear audit trail of all changes
    - Deterministic state building

    Args:
        request: FastAPI request object
        source_id: Source identifier to locate the page-aware text file

    Returns:
        Enhanced JSON structure built incrementally via operations
    """
    start_time = time.time()

    # Log API request
    log_api_request(
        "/extract_enhanced_operations",
        "GET",
        {"source_id": source_id},
        request.client.host,
    )

    # Get or create monitor for this extraction
    monitor = get_extraction_monitor(source_id)

    try:
        # Stage 1: Load page-aware text file
        monitor.start_stage("file_loading", source_id=source_id)

        # Use existing helper function to load pages
        page_aware_text = load_pages_from_file(source_id)

        if not page_aware_text:
            pages_filename = os.path.splitext(source_id)[0] + "_pages.txt"
            error_msg = f"No valid page content found in {pages_filename}"
            raise ValueError(error_msg)

        print(f"📄 Loaded {len(page_aware_text)} pages from page-aware text file")
        
        monitor.end_stage("file_loading", success=True)

        # Stage 2: Operations-based extraction
        monitor.start_stage("operations_extraction", source_id=source_id)

        # Build log file path (consistent with extract_enhanced)
        log_file_path = os.path.join(UPLOAD_FOLDER, f"{source_id}_operations_extraction.jsonl")

        # Import and run operations-based extraction
        from src.api.extraction_helpers import extract_direct_to_enhanced_with_operations

        extraction_result = extract_direct_to_enhanced_with_operations(
            page_aware_text=page_aware_text,
            source_id=source_id,
            log_file_path=log_file_path,
        )

        if not extraction_result:
            monitor.end_stage("operations_extraction", success=False)
            response_time = time.time() - start_time
            log_api_response("/extract_enhanced_operations", 404, response_time)

            return JSONResponse(
                content={
                    "error": "No content extracted",
                    "detail": "Operations-based extraction returned no results",
                    "source_id": source_id
                },
                status_code=404
            )

        monitor.end_stage("operations_extraction", success=True)

        # Save result to file
        try:
            upload_dir = DATA_DIR / "uploads"
            upload_dir.mkdir(exist_ok=True)
            
            result_filename = f"{source_id}_operations_result.json"
            result_path = upload_dir / result_filename
            
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_result, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Operations result saved to {result_filename}")
            
        except Exception as save_error:
            print(f"⚠️ Failed to save operations result: {save_error}")
            # Continue despite save failure

        response_time = time.time() - start_time
        log_api_response("/extract_enhanced_operations", 200, response_time)

        return JSONResponse(content=extraction_result)

    except ValueError as e:
        monitor.log_error("operations_extraction", e)
        response_time = time.time() - start_time
        log_api_response("/extract_enhanced_operations", 404, response_time)

        return JSONResponse(
            content={
                "error": "File not found",
                "detail": str(e),
                "source_id": source_id
            },
            status_code=404
        )

    except json.JSONDecodeError as e:
        monitor.log_error("operations_extraction", e)
        response_time = time.time() - start_time
        log_api_response("/extract_enhanced_operations", 400, response_time)

        return JSONResponse(
            content={
                "error": "Invalid JSON in extraction",
                "detail": str(e),
                "source_id": source_id
            },
            status_code=400
        )

    except Exception as e:
        # Log the exception for debugging
        import traceback
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"❌ Operations extraction error: {json.dumps(error_details, indent=2)}")

        # Final fallback for unexpected errors
        monitor.log_error("operations_extraction", e)
        response_time = time.time() - start_time
        log_api_response("/extract_enhanced_operations", 500, response_time)

        return JSONResponse(
            content={
                "error": "Internal server error",
                "detail": str(e),
                "source_id": source_id
            },
            status_code=500
        )
