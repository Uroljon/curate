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
