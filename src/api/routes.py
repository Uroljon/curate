"""API routes for CURATE."""

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone

import aiofiles
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
from src.api.extraction_helpers import (
    create_error_response,
    create_success_response,
    load_and_validate_pages,
    monitor_stage,
    save_json_file,
)



async def upload_pdf(request: Request, file: UploadFile):
    """Handle PDF upload and processing with page attribution support."""
    start_time = time.time()

    # Log API request
    log_api_request("/upload", "POST", {"filename": file.filename}, request.client.host)

    # Get monitor for this extraction
    monitor = get_extraction_monitor(file.filename)

    try:
        # Stage 1: File upload
        with monitor_stage(monitor, "file_upload", filename=file.filename, size=file.size):
            # Generate unique filename to prevent race conditions
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            safe_filename = f"{timestamp}_{unique_id}_{file.filename}"
            file_path = os.path.join(UPLOAD_FOLDER, safe_filename)

            content = await file.read()
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(content)

        # Stage 2: Page-aware text extraction
        with monitor_stage(monitor, "text_extraction"):
            page_aware_text, extraction_metadata = extract_text_with_ocr_fallback(file_path)

            # Save page-aware text as .txt file (for debugging) with safe filename
            txt_filename = os.path.splitext(safe_filename)[0] + "_pages.txt"
            txt_path = os.path.join(UPLOAD_FOLDER, txt_filename)
            async with aiofiles.open(txt_path, "w", encoding="utf-8") as txt_file:
                for text, page_num in page_aware_text:
                    await txt_file.write(f"\n\n[Page {page_num}]\n\n")
                    await txt_file.write(text)

            total_text_length = sum(len(text) for text, _ in page_aware_text)

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

    except (FileNotFoundError, PermissionError, OSError, Exception) as e:
        # Log error and re-raise for FastAPI to handle
        monitor.log_error("upload", e)
        response_time = time.time() - start_time
        
        # Determine status code based on exception type
        if isinstance(e, FileNotFoundError):
            status_code = 404
        elif isinstance(e, PermissionError):
            status_code = 403
        else:
            status_code = 500
            
        log_api_response("/upload", status_code, response_time)
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
        page_aware_text = await load_and_validate_pages(source_id, monitor)

        # Stage 2: Enhanced direct extraction
        with monitor_stage(monitor, "enhanced_extraction", source_id=source_id):
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

        # Stage 3: Save enhanced structure to file
        with monitor_stage(monitor, "file_saving", source_id=source_id):
            # Save the enhanced structure JSON file for visualization tool
            enhanced_filename = f"{source_id}_enhanced_structure.json"
            enhanced_data = extraction_result["extraction_result"]
            enhanced_path = await save_json_file(enhanced_data, enhanced_filename, UPLOAD_FOLDER)
            print(f"💾 Saved enhanced structure to: {enhanced_filename}")

        # Stage 4: Prepare response
        with monitor_stage(monitor, "response_preparation", source_id=source_id):
            additional_fields = {
                "enhanced_file": enhanced_filename,  # Add filename for reference
                "metadata": extraction_result["metadata"],
                "processing_stages": [
                    "file_loading",
                    "enhanced_extraction", 
                    "file_saving",
                    "response_preparation",
                ],
            }

        print(f"✅ Enhanced extraction completed in {time.time() - start_time:.1f}s")
        return create_success_response(
            "/extract_enhanced",
            source_id,
            extraction_result["extraction_result"],
            start_time,
            additional_fields,
        )

    except FileNotFoundError as e:
        return create_error_response(
            "/extract_enhanced", "File not found", 404, source_id, e, start_time, monitor, "file_loading",
            "Make sure to upload the PDF first using /upload endpoint"
        )
    except ValueError as e:
        return create_error_response(
            "/extract_enhanced", "Invalid data", 400, source_id, e, start_time, monitor, "file_loading"
        )
    except RuntimeError as e:
        return create_error_response(
            "/extract_enhanced", "Extraction failed", 500, source_id, e, start_time, monitor, "enhanced_extraction"
        )
    except Exception as e:
        return create_error_response(
            "/extract_enhanced", "Internal server error", 500, source_id, e, start_time, monitor, "enhanced_extraction"
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
        page_aware_text = await load_and_validate_pages(source_id, monitor)

        # Stage 2: Operations-based extraction
        with monitor_stage(monitor, "operations_extraction", source_id=source_id):
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
                raise ValueError("Operations-based extraction returned no results")

        # Save result to file
        try:
            upload_dir = PROJECT_ROOT / "data" / "uploads"
            upload_dir.mkdir(exist_ok=True)
            result_filename = f"{source_id}_operations_result.json"
            await save_json_file(extraction_result, result_filename, str(upload_dir))
            print(f"💾 Operations result saved to {result_filename}")
        except Exception as save_error:
            print(f"⚠️ Failed to save operations result: {save_error}")
            # Continue despite save failure

        return create_success_response(
            "/extract_enhanced_operations",
            source_id,
            extraction_result,
            start_time,
        )

    except json.JSONDecodeError as e:
        return create_error_response(
            "/extract_enhanced_operations", "Invalid JSON in extraction", 400, source_id, e, start_time, monitor, "operations_extraction"
        )
    except ValueError as e:
        return create_error_response(
            "/extract_enhanced_operations", "File not found", 404, source_id, e, start_time, monitor, "operations_extraction"
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

        return create_error_response(
            "/extract_enhanced_operations", "Internal server error", 500, source_id, e, start_time, monitor, "operations_extraction"
        )
