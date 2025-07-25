"""API routes for CURATE."""

import json
import os
import time
from typing import Any

from fastapi import Request, UploadFile
from fastapi.responses import JSONResponse

from src.core import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    FAST_EXTRACTION_ENABLED,
    FAST_EXTRACTION_MAX_CHUNKS,
    SEMANTIC_CHUNK_TARGET_CHARS,
    UPLOAD_FOLDER,
)
from src.extraction import extract_structures_with_retry, extract_with_accumulation
from src.processing import (
    chunk_for_embedding_enhanced,
    chunk_for_llm,
    embed_chunks,
    extract_text_with_ocr_fallback,
    get_all_chunks_for_document,
)
from src.utils import (
    ChunkQualityMonitor,
    get_extraction_monitor,
    log_api_request,
    log_api_response,
)

from .extraction_helpers import (
    ExtractionChangeTracker,
    aggregate_extraction_results,
    deduplicate_extraction_results,
    extract_all_action_fields,
    extract_projects_and_details,
    merge_extraction_results,
    prepare_chunks_for_extraction,
    print_extraction_summary,
    process_chunks_for_fast_extraction,
)


async def upload_pdf(request: Request, file: UploadFile):
    """Handle PDF upload and processing."""
    start_time = time.time()

    # Log API request
    log_api_request("/upload", "POST", {"filename": file.filename}, request.client.host)

    # Get monitor for this extraction
    monitor = get_extraction_monitor(file.filename)

    try:
        # Stage 1: File upload
        monitor.start_stage("file_upload", filename=file.filename, size=file.size)
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        monitor.end_stage("file_upload", file_size=len(content))

        # Stage 2: Text extraction
        monitor.start_stage("text_extraction")
        extracted_text, extraction_metadata = extract_text_with_ocr_fallback(file_path)
        monitor.end_stage(
            "text_extraction",
            text_length=len(extracted_text),
            total_pages=extraction_metadata["total_pages"],
            ocr_pages=extraction_metadata["ocr_pages"],
            native_pages=extraction_metadata["native_pages"],
            ocr_percentage=extraction_metadata["extraction_method_ratio"][
                "ocr_percentage"
            ],
        )

        # Stage 3: Semantic chunking
        monitor.start_stage("semantic_chunking")
        chunks = chunk_for_embedding_enhanced(
            extracted_text,
            max_chars=SEMANTIC_CHUNK_TARGET_CHARS,
            use_structure_aware=True,
            pdf_path=file_path,
        )

        # Analyze chunk quality
        chunk_metrics = ChunkQualityMonitor.analyze_chunks(chunks, "semantic")
        monitor.end_stage(
            "semantic_chunking", chunk_count=len(chunks), chunk_metrics=chunk_metrics
        )

        # Stage 4: Embedding generation
        monitor.start_stage("embedding_generation")
        embed_chunks(chunks, source_id=file.filename)
        monitor.end_stage("embedding_generation", chunks_embedded=len(chunks))

        # Prepare response
        response_data = {
            "chunks": len(chunks),
            "text_length": len(extracted_text),
            "chunk_quality": {
                "avg_size": chunk_metrics["size_stats"]["avg"],
                "with_structure": chunk_metrics["structural_stats"][
                    "chunks_with_structure"
                ],
            },
        }

        # Log successful completion
        monitor.finalize({"upload_result": response_data})

        # Log API response
        response_time = time.time() - start_time
        log_api_response("/upload", 200, response_time)

        return JSONResponse(content=response_data)

    except Exception as e:
        monitor.log_error("upload", e)
        response_time = time.time() - start_time
        log_api_response("/upload", 500, response_time)
        raise


async def extract_structure(
    source_id: str, max_chars: int = CHUNK_MAX_CHARS, min_chars: int = CHUNK_MIN_CHARS
):
    """Multi-stage structure extraction."""
    # Prepare chunks for extraction
    optimized_chunks = prepare_chunks_for_extraction(source_id, max_chars, min_chars)

    print(f"\n🚀 Starting multi-stage extraction with {len(optimized_chunks)} chunks\n")

    # Stage 1: Extract action fields
    action_fields = extract_all_action_fields(optimized_chunks)

    if not action_fields:
        return JSONResponse(content={"structures": []})

    # Stage 2 & 3: Extract projects and details
    all_extracted_data = extract_projects_and_details(optimized_chunks, action_fields)

    if not all_extracted_data:
        print("⚠️ No valid data extracted from any chunks")
        return JSONResponse(content={"structures": []})

    # Deduplicate results
    final_structures = deduplicate_extraction_results(all_extracted_data)

    # Print summary
    print_extraction_summary(final_structures, len(optimized_chunks))

    return JSONResponse(content={"structures": final_structures})


async def extract_structure_fast(
    request: Request,
    source_id: str,
    max_chars: int = CHUNK_MAX_CHARS,
    min_chars: int = CHUNK_MIN_CHARS,
):
    """Fast single-pass extraction endpoint for quicker iteration and testing."""

    start_time = time.time()

    # Log API request
    log_api_request(
        "/extract_structure_fast",
        "GET",
        {"source_id": source_id, "max_chars": max_chars, "min_chars": min_chars},
        request.client.host,
    )

    if not FAST_EXTRACTION_ENABLED:
        log_api_response("/extract_structure_fast", 503, time.time() - start_time)
        return JSONResponse(
            status_code=503,
            content={"error": "Fast extraction is disabled in configuration"},
        )

    # Get or create monitor
    monitor = get_extraction_monitor(source_id)

    try:
        # Stage 1: Chunk retrieval
        monitor.start_stage("chunk_retrieval", source_id=source_id)
        # Get all chunks for fast but comprehensive extraction
        chunks = get_all_chunks_for_document(source_id)
        raw_texts = [c["text"] for c in chunks]
        monitor.end_stage("chunk_retrieval", chunks_retrieved=len(raw_texts))

        # Stage 2: LLM chunk preparation
        monitor.start_stage("llm_chunk_preparation")
        # Use simple size-based chunking
        optimized_chunks = chunk_for_llm(
            raw_texts, max_chars=max_chars, min_chars=min_chars
        )

        # Analyze LLM chunk quality
        llm_chunk_metrics = ChunkQualityMonitor.analyze_chunks(optimized_chunks, "llm")
        monitor.end_stage(
            "llm_chunk_preparation",
            original_chunks=len(raw_texts),
            optimized_chunks=len(optimized_chunks),
            chunk_metrics=llm_chunk_metrics,
        )

        # Apply chunk limit if configured
        if FAST_EXTRACTION_MAX_CHUNKS > 0:
            optimized_chunks = optimized_chunks[:FAST_EXTRACTION_MAX_CHUNKS]
            print(f"⚡ Fast mode: Processing only first {len(optimized_chunks)} chunks")

        print(
            f"\n🚀 Starting FAST single-pass extraction with {len(optimized_chunks)} chunks\n"
        )

        # Stage 3: LLM extraction
        monitor.start_stage("llm_extraction", total_chunks=len(optimized_chunks))

        # Independent extraction per chunk
        all_chunk_results = []
        chunk_timings = []

        for i, chunk in enumerate(optimized_chunks):
            chunk_start = time.time()
            print(
                f"⚡ Processing chunk {i+1}/{len(optimized_chunks)} ({len(chunk)} chars)"
            )

            # Extract independently from each chunk
            chunk_data = extract_structures_with_retry(chunk)

            chunk_time = time.time() - chunk_start
            chunk_timings.append(chunk_time)

            if chunk_data:
                all_chunk_results.extend(chunk_data)
                print(
                    f"   ✅ Found {len(chunk_data)} action fields in {chunk_time:.2f}s"
                )
            else:
                print(f"   ⚠️ No data found ({chunk_time:.2f}s)")

        # Aggregate all results using separate LLM call
        print(f"\n🔄 Aggregating {len(all_chunk_results)} extracted action fields...")
        aggregated_fields = aggregate_extraction_results(all_chunk_results)

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

        # Use the aggregated results
        final_structures = final_action_fields

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

        # Prepare response
        response_data = {
            "structures": final_structures,
            "metadata": {
                "extraction_time_seconds": round(extraction_time, 2),
                "chunks_processed": len(optimized_chunks),
                "action_fields_found": len(final_structures),
                "total_projects": total_projects,
                "projects_with_measures": measures_count,
                "projects_with_indicators": indicators_count,
            },
        }

        # Finalize monitoring
        monitor.finalize(response_data)

        # Log API response
        log_api_response("/extract_structure_fast", 200, extraction_time)

        return JSONResponse(content=response_data)

    except Exception as e:
        monitor.log_error("extraction", e)
        response_time = time.time() - start_time
        log_api_response("/extract_structure_fast", 500, response_time)
        raise
