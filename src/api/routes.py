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
    SEMANTIC_CHUNK_TARGET_CHARS,
    UPLOAD_FOLDER,
)
from src.extraction import extract_structures_with_retry, extract_with_accumulation
from src.processing import (
    chunk_for_embedding_enhanced,
    chunk_for_embedding_with_pages,
    chunk_for_llm,
    embed_chunks,
    embed_chunks_with_pages,
    extract_text_legacy,
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
    add_source_attributions,
    aggregate_extraction_results,
    deduplicate_extraction_results,
    extract_all_action_fields,
    extract_projects_and_details,
    merge_extraction_results,
    prepare_chunks_for_extraction,
    print_extraction_summary,
    process_chunks_for_fast_extraction,
)


async def upload_pdf_with_pages(request: Request, file: UploadFile):
    """Handle PDF upload and processing with page attribution support."""
    start_time = time.time()

    # Log API request
    log_api_request(
        "/upload_with_pages", "POST", {"filename": file.filename}, request.client.host
    )

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
                txt_file.write(f"\n\n# ====== PAGE {page_num} ======\n\n")
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

        # Stage 3: Page-aware semantic chunking
        monitor.start_stage("semantic_chunking")
        chunks_with_pages = chunk_for_embedding_with_pages(
            page_aware_text,
            max_chars=SEMANTIC_CHUNK_TARGET_CHARS,
        )

        # Analyze chunk quality (using text only for compatibility)
        chunk_texts = [chunk["text"] for chunk in chunks_with_pages]
        chunk_metrics = ChunkQualityMonitor.analyze_chunks(chunk_texts, "semantic")

        # Save chunked text with page info using safe filename
        chunks_filename = os.path.splitext(safe_filename)[0] + "_chunks_with_pages.txt"
        chunks_path = os.path.join(UPLOAD_FOLDER, chunks_filename)
        with open(chunks_path, "w", encoding="utf-8") as chunks_file:
            for i, chunk in enumerate(chunks_with_pages, 1):
                chunks_file.write(
                    f"\n\n# ====== CHUNK {i} (Pages: {chunk['pages']}) ======\n\n"
                )
                chunks_file.write(chunk["text"])

        monitor.end_stage(
            "semantic_chunking",
            chunk_count=len(chunks_with_pages),
            chunk_metrics=chunk_metrics,
            page_aware=True,
        )

        # Stage 4: Page-aware embedding generation
        monitor.start_stage("embedding_generation")
        embed_chunks_with_pages(chunks_with_pages, source_id=safe_filename)
        monitor.end_stage(
            "embedding_generation", chunks_embedded=len(chunks_with_pages)
        )

        # Prepare response
        page_stats: dict[int, int] = {}
        # Initialize all pages found in chunks to 0
        all_pages_in_chunks = set()
        for chunk in chunks_with_pages:
            all_pages_in_chunks.update(chunk["pages"])
        for page_num in all_pages_in_chunks:
            page_stats[page_num] = 0

        # Count how many chunks each page belongs to
        for page_num in page_stats:
            count = 0
            for chunk in chunks_with_pages:
                if page_num in chunk["pages"]:
                    count += 1
            page_stats[page_num] = count

        response_data = {
            "chunks": len(chunks_with_pages),
            "total_text_length": total_text_length,
            "pages_processed": len(page_aware_text),
            "page_attribution_enabled": True,
            "chunk_quality": {
                "avg_size": chunk_metrics["size_stats"]["avg"],
                "with_structure": chunk_metrics["structural_stats"].get(
                    "chunks_with_structure", 0
                ),
            },
            "page_coverage": {
                "total_pages": len(page_stats),
                "chunks_per_page": dict(sorted(page_stats.items())),
            },
            "source_id": safe_filename,  # Return the safe filename for subsequent API calls
            "original_filename": file.filename,  # Keep original filename for reference
        }

        # Log successful completion
        monitor.finalize({"upload_result": response_data})

        # Log API response
        response_time = time.time() - start_time
        log_api_response("/upload_with_pages", 200, response_time)

        return JSONResponse(content=response_data)

    except FileNotFoundError as e:
        monitor.log_error("upload", e)
        response_time = time.time() - start_time
        log_api_response("/upload_with_pages", 404, response_time)
        raise
    except PermissionError as e:
        monitor.log_error("upload", e)
        response_time = time.time() - start_time
        log_api_response("/upload_with_pages", 403, response_time)
        raise
    except OSError as e:
        monitor.log_error("upload", e)
        response_time = time.time() - start_time
        log_api_response("/upload_with_pages", 500, response_time)
        raise
    except Exception as e:
        # Final fallback for unexpected errors
        monitor.log_error("upload", e)
        response_time = time.time() - start_time
        log_api_response("/upload_with_pages", 500, response_time)
        raise


async def upload_pdf(request: Request, file: UploadFile):
    """Handle PDF upload and processing (legacy version for backward compatibility)."""
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

        # Stage 2: Text extraction
        monitor.start_stage("text_extraction")
        extracted_text, extraction_metadata = extract_text_legacy(file_path)

        # Save extracted text as .txt file
        txt_filename = os.path.splitext(file.filename)[0] + ".txt"
        txt_path = os.path.join(UPLOAD_FOLDER, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(extracted_text)

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

        # Save chunked text as separate file
        chunks_filename = os.path.splitext(file.filename)[0] + "_chunks.txt"
        chunks_path = os.path.join(UPLOAD_FOLDER, chunks_filename)
        with open(chunks_path, "w", encoding="utf-8") as chunks_file:
            for i, chunk in enumerate(chunks, 1):
                chunks_file.write(f"\n\n# ====== CHUNK {i} ======\n\n")
                chunks_file.write(chunk)

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
                "with_structure": chunk_metrics["structural_stats"].get(
                    "chunks_with_structure", 0
                ),
            },
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
    source_id: str, max_chars: int = CHUNK_MAX_CHARS, min_chars: int = CHUNK_MIN_CHARS
):
    """Multi-stage structure extraction."""
    # Set up LLM dialog log file path
    llm_dialog_filename = f"{source_id}_llm_dialog.txt"
    llm_dialog_path = os.path.join(UPLOAD_FOLDER, llm_dialog_filename)

    # Prepare chunks for extraction
    optimized_chunks = prepare_chunks_for_extraction(source_id, max_chars, min_chars)

    print(f"\n🚀 Starting multi-stage extraction with {len(optimized_chunks)} chunks\n")

    # Stage 1: Extract action fields
    action_fields = extract_all_action_fields(
        optimized_chunks,
        log_file_path=llm_dialog_path,
        log_context_prefix="Regular Extraction",
    )

    if not action_fields:
        return JSONResponse(content={"structures": []})

    # Stage 2 & 3: Extract projects and details
    all_extracted_data = extract_projects_and_details(
        optimized_chunks,
        action_fields,
        log_file_path=llm_dialog_path,
        log_context_prefix="Regular Extraction",
    )

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

        # Check if we have page-aware chunks and convert to page_aware_text format
        page_aware_chunks = [chunk for chunk in chunks if chunk.get("pages")]
        if page_aware_chunks:
            print(f"🔍 Using page-aware chunking with {len(page_aware_chunks)} chunks")
            # Convert chunks to page_aware_text format for new chunker
            page_aware_text: list[tuple[str, int]] = []
            for chunk in page_aware_chunks:
                chunk_text = chunk["text"]
                chunk_pages = chunk["pages"]

                if not chunk_pages:
                    # No page info, use page 1
                    page_aware_text.append((chunk_text, 1))
                elif len(chunk_pages) == 1:
                    # Single page chunk
                    page_aware_text.append((chunk_text, chunk_pages[0]))
                else:
                    # Multi-page chunk - distribute text proportionally
                    # This is still an approximation but better than using only first page
                    lines = chunk_text.split("\n")
                    lines_per_page = max(1, len(lines) // len(chunk_pages))

                    for i, page_num in enumerate(chunk_pages):
                        start_line = i * lines_per_page
                        if i == len(chunk_pages) - 1:
                            # Last page gets remaining lines
                            page_text = "\n".join(lines[start_line:])
                        else:
                            end_line = start_line + lines_per_page
                            page_text = "\n".join(lines[start_line:end_line])

                        if page_text.strip():  # Only add non-empty text
                            page_aware_text.append((page_text, page_num))
        else:
            # Fallback: create page_aware_text with page 1 for all chunks
            print("⚠️ No page information available, using fallback chunking")
            page_aware_text = [(text, 1) for text in raw_texts]

        monitor.end_stage("chunk_retrieval", chunks_retrieved=len(raw_texts))

        # Stage 2: LLM chunk preparation
        monitor.start_stage("llm_chunk_preparation")
        # Use page-aware chunking
        from src.processing.chunker import chunk_for_llm_with_pages

        optimized_chunks_with_pages = chunk_for_llm_with_pages(
            page_aware_text, max_chars=max_chars, min_chars=min_chars
        )

        # Extract just the text for compatibility with existing code
        optimized_chunks: list[str] = [
            chunk_text for chunk_text, _ in optimized_chunks_with_pages
        ]

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
            optimized_chunks_with_pages = optimized_chunks_with_pages[
                :FAST_EXTRACTION_MAX_CHUNKS
            ]
            print(f"⚡ Fast mode: Processing only first {len(optimized_chunks)} chunks")

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

        print(
            f"\n🚀 Starting FAST single-pass extraction with {len(optimized_chunks)} chunks\n"
        )

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
                log_context=f"Fast Extraction - Chunk {i+1}/{len(optimized_chunks)} ({len(chunk_text)} chars)",
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
            log_context_prefix="Fast Extraction Aggregation",
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
        if page_aware_chunks:
            # Create chunks that map LLM chunk IDs to their page numbers
            llm_page_aware_chunks = []
            for chunk_id, (chunk_text, chunk_pages) in enumerate(
                optimized_chunks_with_pages
            ):
                llm_page_aware_chunks.append(
                    {"text": chunk_text, "pages": chunk_pages, "chunk_id": chunk_id}
                )

            # Try to load original page-aware text for precise attribution
            original_page_text = []
            pages_file = os.path.join(
                UPLOAD_FOLDER, f"{os.path.splitext(source_id)[0]}_pages.txt"
            )
            if os.path.exists(pages_file):
                print(f"📄 Loading original page text from {pages_file}")
                try:
                    with open(pages_file, encoding="utf-8") as f:
                        content = f.read()

                    # Use regex for more robust parsing
                    page_pattern = re.compile(
                        r"# ====== PAGE (\d+) ======\n(.*?)(?=\n\n# ====== PAGE|\Z)",
                        re.DOTALL,
                    )
                    matches = page_pattern.findall(content)

                    for page_num_str, page_text in matches:
                        try:
                            page_num = int(page_num_str)
                            original_page_text.append((page_text.strip(), page_num))
                        except ValueError:
                            print(f"   ⚠️ Skipping invalid page number: {page_num_str}")

                    print(
                        f"   ✅ Loaded {len(original_page_text)} pages of original text"
                    )
                except ValueError as e:
                    print(f"   ⚠️ Error parsing page numbers: {e}")
                    # Continue without page attribution rather than failing
                except OSError as e:
                    print(f"   ⚠️ Error reading page text file: {e}")
                    # Continue without page attribution rather than failing
                except Exception as e:
                    print(
                        f"   ⚠️ Unexpected error parsing page text: {type(e).__name__}: {e}"
                    )
                    # Continue without page attribution rather than failing
            else:
                print(f"   ⚠️ Original page text file not found: {pages_file}")

            print(
                f"\n🔍 Adding source attribution using {len(llm_page_aware_chunks)} LLM page-aware chunks..."
            )
            final_structures = add_source_attributions(
                final_action_fields, llm_page_aware_chunks, original_page_text
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
                page_aware_chunks=len(page_aware_chunks),
                projects_with_sources=attribution_stats["projects_with_sources"],
                total_projects=attribution_stats["total_projects"],
                attribution_success_rate=attribution_stats["attribution_success_rate"],
            )
        else:
            print("INFO: No page attribution data available - using legacy chunks")
            final_structures = final_action_fields

            # Calculate stats even without attribution for accurate reporting
            attribution_stats["total_projects"] = sum(
                len(af["projects"]) for af in final_structures
            )
            attribution_stats["projects_with_sources"] = (
                0  # No sources without page-aware chunks
            )
            attribution_stats["attribution_success_rate"] = (
                0.0  # No attribution possible
            )

            monitor.end_stage(
                "source_attribution", page_aware_chunks=0, attribution_enabled=False
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
        log_api_response("/extract_structure_fast", 200, extraction_time)

        # Ensure response is JSON serializable
        from fastapi.encoders import jsonable_encoder

        return JSONResponse(content=jsonable_encoder(response_data))

    except FileNotFoundError as e:
        monitor.log_error("extraction", e)
        response_time = time.time() - start_time
        log_api_response("/extract_structure_fast", 404, response_time)
        raise
    except ValueError as e:
        monitor.log_error("extraction", e)
        response_time = time.time() - start_time
        log_api_response("/extract_structure_fast", 400, response_time)
        raise
    except Exception as e:
        # Final fallback for unexpected errors
        monitor.log_error("extraction", e)
        response_time = time.time() - start_time
        log_api_response("/extract_structure_fast", 500, response_time)
        raise
