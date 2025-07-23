#!/usr/bin/env python3
"""
Test the improved semantic-aware chunking with the Regensburg PDF.
Verifies if we achieve >70% indicator extraction rate.
"""

import json
import time
from datetime import datetime, timezone

import requests


def test_regensburg_extraction():
    """Test extraction on Regensburg PDF with semantic-aware chunking."""
    base_url = "http://127.0.0.1:8000"
    pdf_path = "uploads/regensburg.pdf"

    print("=" * 80)
    print("Testing Regensburg PDF with Semantic-Aware Chunking")
    print("=" * 80)
    print(f"Timestamp: {datetime.now(timezone.utc)}")
    print(f"PDF: {pdf_path}")
    print()

    # Step 1: Upload the PDF
    print("Step 1: Uploading PDF...")
    with open(pdf_path, "rb") as f:
        files = {"file": ("regensburg.pdf", f, "application/pdf")}
        response = requests.post(f"{base_url}/upload", files=files)

    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return

    upload_result = response.json()
    # The source_id is the filename (from the main.py code)
    source_id = "regensburg.pdf"

    print(f"✅ Upload successful: {source_id}")
    print(f"   Text length: {upload_result['text_length']:,} chars")
    print(f"   Chunks created: {upload_result['chunks']}")
    print()

    # Step 2: Extract structure using fast endpoint
    print("Step 2: Extracting structure with semantic-aware chunking...")
    start_time = time.time()

    response = requests.get(
        f"{base_url}/extract_structure_fast", params={"source_id": source_id}
    )

    if response.status_code != 200:
        print(f"❌ Extraction failed: {response.text}")
        return

    extraction_time = time.time() - start_time
    result = response.json()

    print(f"✅ Extraction completed in {extraction_time:.1f} seconds")
    print()

    # Step 3: Analyze results
    print("Step 3: Analyzing Results")
    print("-" * 60)

    # Count entities
    action_fields = result.get("action_fields", [])
    total_projects = sum(len(af.get("projects", [])) for af in action_fields)
    total_measures = sum(
        sum(len(p.get("measures", [])) for p in af.get("projects", []))
        for af in action_fields
    )
    total_indicators = sum(
        sum(len(p.get("indicators", [])) for p in af.get("projects", []))
        for af in action_fields
    )

    print(f"Action Fields: {len(action_fields)}")
    print(f"Projects: {total_projects}")
    print(f"Measures: {total_measures}")
    print(f"Indicators: {total_indicators}")
    print()

    # Show action fields
    print("Action Fields Found:")
    for af in action_fields:
        print(f"  - {af['action_field']}")
    print()

    # Show extraction metadata
    metadata = result.get("metadata", {})
    print("Extraction Metadata:")
    print(f"  Extraction time: {metadata.get('extraction_time', 'N/A'):.1f}s")
    print(f"  Processing mode: {metadata.get('processing_mode', 'N/A')}")
    print(f"  Total chunks: {metadata.get('total_chunks', 'N/A')}")
    print(f"  Chunks processed: {metadata.get('chunks_processed', 'N/A')}")
    print(
        f"  Chunk consolidation: {metadata.get('original_chunks', 'N/A')} → {metadata.get('llm_chunks', 'N/A')} LLM chunks"
    )
    print()

    # Analyze chunk quality if available
    chunk_quality = metadata.get("chunk_quality", {})
    if chunk_quality:
        print("Chunk Quality Analysis:")
        topic_coherence = chunk_quality.get("topic_coherence", {})
        if topic_coherence:
            print(
                f"  Single topic chunks: {topic_coherence.get('chunks_with_single_topic', 'N/A')}"
            )
            print(
                f"  Multiple topic chunks: {topic_coherence.get('chunks_with_multiple_topics', 'N/A')}"
            )
            print(
                f"  No topic chunks: {topic_coherence.get('chunks_with_no_topic', 'N/A')}"
            )
            print(f"  Mixed sections: {chunk_quality.get('mixed_sections', 'N/A')}")
    print()

    # Calculate indicator extraction rate
    # Expected indicators based on previous analysis: ~25-30
    expected_indicators = 28  # Conservative estimate
    extraction_rate = (total_indicators / expected_indicators) * 100

    print("Indicator Extraction Analysis:")
    print(f"  Indicators found: {total_indicators}")
    print(f"  Expected indicators: ~{expected_indicators}")
    print(f"  Extraction rate: {extraction_rate:.1f}%")
    print("  Target: >70%")
    print(f"  Status: {'✅ PASSED' if extraction_rate > 70 else '❌ FAILED'}")
    print()

    # Sample some indicators
    print("Sample Indicators Found:")
    sample_count = 0
    for af in action_fields:
        for project in af.get("projects", []):
            for indicator in project.get("indicators", []):
                if sample_count < 10:
                    print(f"  - {indicator}")
                    sample_count += 1

    # Save detailed results
    output_file = f"test_results/regensburg_semantic_test_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pdf": pdf_path,
                "extraction_time": extraction_time,
                "statistics": {
                    "action_fields": len(action_fields),
                    "projects": total_projects,
                    "measures": total_measures,
                    "indicators": total_indicators,
                    "extraction_rate": extraction_rate,
                },
                "full_result": result,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nDetailed results saved to: {output_file}")

    return extraction_rate > 70


if __name__ == "__main__":
    # Make sure the API is running
    try:
        response = requests.get("http://127.0.0.1:8000/docs")
        if response.status_code != 200:
            print(
                "❌ API is not accessible. Please start with: uvicorn main:app --reload"
            )
            exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ API is not running. Please start with: uvicorn main:app --reload")
        exit(1)

    # Run the test
    success = test_regensburg_extraction()
    exit(0 if success else 1)
