"""API routes for CURATE."""

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone

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
from src.core.config import PROJECT_ROOT
from src.processing import extract_text_with_ocr_fallback
from src.utils import (
    get_extraction_monitor,
    log_api_request,
    log_api_response,
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
        log_file_path = os.path.join(
            UPLOAD_FOLDER, f"{source_id}_enhanced_extraction.jsonl"
        )

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
        monitor.end_stage(
            "file_saving", enhanced_file_size=os.path.getsize(enhanced_path)
        )

        # Stage 4: Prepare response
        monitor.start_stage("response_preparation", source_id=source_id)

        response_data = {
            "source_id": source_id,
            "extraction_result": extraction_result["extraction_result"],
            "enhanced_file": enhanced_filename,  # Add filename for reference
            "metadata": extraction_result["metadata"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_stages": [
                "file_loading",
                "enhanced_extraction",
                "file_saving",
                "response_preparation",
            ],
        }

        monitor.end_stage("response_preparation")

        # Log successful response
        response_time = time.time() - start_time
        log_api_response("/extract_enhanced", 200, response_time)

        print(f"✅ Enhanced extraction completed in {response_time:.1f}s")

        return JSONResponse(
            content=jsonable_encoder(response_data),
            status_code=200,
            headers={"Content-Type": "application/json; charset=utf-8"},
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
                "suggestion": "Make sure to upload the PDF first using /upload endpoint",
            },
            status_code=404,
        )

    except ValueError as e:
        # Invalid data errors
        monitor.log_error("file_loading", e)
        response_time = time.time() - start_time
        log_api_response("/extract_enhanced", 400, response_time)

        return JSONResponse(
            content={"error": "Invalid data", "detail": str(e), "source_id": source_id},
            status_code=400,
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
                "source_id": source_id,
            },
            status_code=500,
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
                "source_id": source_id,
            },
            status_code=500,
        )


async def extract_enhanced_operations(
    request: Request,
    source_id: str,
):
    """
    Operations-based extraction endpoint that builds state incrementally.

    This endpoint implements the operations-based approach:
    1. Reads {source_id}_pages.txt directly
    2. For each chunk, LLM returns operations (CREATE, UPDATE, CONNECT)
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
        log_file_path = os.path.join(
            UPLOAD_FOLDER, f"{source_id}_operations_extraction.jsonl"
        )

        # Import and run operations-based extraction
        from src.api.extraction_helpers import (
            extract_direct_to_enhanced_with_operations,
        )

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
                    "source_id": source_id,
                },
                status_code=404,
            )

        monitor.end_stage("operations_extraction", success=True)

        # Save result to file
        try:
            upload_dir = PROJECT_ROOT / "data" / "uploads"
            upload_dir.mkdir(exist_ok=True)

            result_filename = f"{source_id}_operations_result.json"
            result_path = upload_dir / result_filename

            import aiofiles
            async with aiofiles.open(result_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(extraction_result, indent=2, ensure_ascii=False))

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
                "source_id": source_id,
            },
            status_code=404,
        )

    except json.JSONDecodeError as e:
        monitor.log_error("operations_extraction", e)
        response_time = time.time() - start_time
        log_api_response("/extract_enhanced_operations", 400, response_time)

        return JSONResponse(
            content={
                "error": "Invalid JSON in extraction",
                "detail": str(e),
                "source_id": source_id,
            },
            status_code=400,
        )

    except Exception as e:
        # Log the exception for debugging
        import traceback

        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
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
                "source_id": source_id,
            },
            status_code=500,
        )
