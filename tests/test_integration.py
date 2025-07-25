#!/usr/bin/env python3
"""Integration test for the full PDF extraction pipeline."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time

from src.processing.chunker import chunk_for_embedding_enhanced, chunk_for_llm
from src.processing.embedder import embed_chunks, query_chunks
from src.processing.parser import extract_text_with_ocr_fallback


def test_real_pdf_flow(pdf_path: str):
    """Test the complete flow with a real PDF."""

    print(f"🧪 Integration Test with Real PDF: {pdf_path}")
    print("=" * 70)

    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return False

    test_source_id = f"test_{int(time.time())}"

    try:
        # Step 1: Extract text from PDF
        print("\n1️⃣ Extracting text from PDF...")
        start = time.time()
        extracted_text, extraction_metadata = extract_text_with_ocr_fallback(pdf_path)
        extraction_time = time.time() - start

        print(
            f"   ✅ Extracted {len(extracted_text):,} characters in {extraction_time:.2f}s"
        )
        print(
            f"   📄 Total pages: {extraction_metadata['total_pages']} ({extraction_metadata['native_pages']} native, {extraction_metadata['ocr_pages']} OCR)"
        )
        print(f"   📄 First 200 chars: {extracted_text[:200]}...")

        if len(extracted_text) < 100:
            print("   ⚠️  Warning: Very little text extracted!")

        # Step 2: Create semantic chunks
        print("\n2️⃣ Creating semantic chunks...")
        start = time.time()
        chunks = chunk_for_embedding_enhanced(extracted_text, max_chars=5000)
        chunking_time = time.time() - start

        print(f"   ✅ Created {len(chunks)} chunks in {chunking_time:.2f}s")

        # Analyze chunk quality
        chunk_sizes = [len(c) for c in chunks]
        print(
            f"   📊 Chunk sizes: min={min(chunk_sizes)}, max={max(chunk_sizes)}, avg={sum(chunk_sizes)/len(chunk_sizes):.0f}"
        )

        # Check for heading detection
        from src.utils.text import is_heading

        total_headings = 0
        chunks_with_headings = 0
        for chunk in chunks:
            chunk_headings = 0
            for line in chunk.split("\n"):
                if is_heading(line.strip()):
                    chunk_headings += 1
                    total_headings += 1
            if chunk_headings > 0:
                chunks_with_headings += 1

        print(
            f"   📋 Heading detection: {total_headings} headings in {chunks_with_headings}/{len(chunks)} chunks"
        )

        # Step 3: Create and store embeddings
        print("\n3️⃣ Creating embeddings...")
        start = time.time()
        embed_chunks(chunks, source_id=test_source_id)
        embedding_time = time.time() - start

        print(f"   ✅ Created embeddings in {embedding_time:.2f}s")

        # Step 4: Test retrieval
        print("\n4️⃣ Testing retrieval...")
        test_queries = [
            "Handlungsfeld",
            "Klimaschutz",
            "Mobilität",
            "Indikatoren",
            "2030",
        ]

        for query in test_queries:
            results = query_chunks(query, top_k=3, source_id=test_source_id)
            print(f"   🔍 Query '{query}': Found {len(results)} results")
            if results and results[0]["text"]:
                preview = results[0]["text"][:100].replace("\n", " ")
                print(f"      Best match: {preview}...")

        # Step 5: Compare chunking methods
        print("\n5️⃣ Comparing LLM chunking methods...")

        # Old method
        old_chunks = chunk_for_llm(chunks, max_chars=20000, min_chars=15000)
        print(f"   📊 Old method: {len(old_chunks)} chunks")

        # New semantic method
        # New semantic method uses same function now
        new_chunks = chunk_for_llm(chunks, max_chars=20000, min_chars=15000)
        print(f"   📊 New method: {len(new_chunks)} chunks")

        # Compare quality
        from src.processing.chunker import analyze_chunk_quality

        old_quality = analyze_chunk_quality(old_chunks)
        new_quality = analyze_chunk_quality(new_chunks)

        print("\n   📈 Quality comparison:")
        print(
            f"      Chunks with headings: {old_quality['heading_stats']['chunks_with_headings']} → {new_quality['heading_stats']['chunks_with_headings']}"
        )
        print(
            f"      Mixed sections: {old_quality.get('mixed_sections', 0)} → {new_quality.get('mixed_sections', 0)}"
        )

        # Step 6: Check for potential issues
        print("\n6️⃣ Checking for potential issues...")

        issues = []

        # Check for empty chunks
        empty_chunks = sum(1 for c in chunks if len(c.strip()) < 10)
        if empty_chunks > 0:
            issues.append(f"Found {empty_chunks} nearly empty chunks")

        # Check for huge chunks
        huge_chunks = sum(1 for c in chunks if len(c) > 10000)
        if huge_chunks > 0:
            issues.append(f"Found {huge_chunks} chunks over 10K chars")

        # Check for poor heading distribution
        if chunks_with_headings < len(chunks) * 0.3:
            issues.append(
                f"Only {chunks_with_headings}/{len(chunks)} chunks have headings"
            )

        if issues:
            print("   ⚠️  Issues found:")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print("   ✅ No major issues detected")

        # Cleanup
        print("\n7️⃣ Cleaning up test data...")
        from src.processing.embedder import collection

        existing = collection.get(where={"source": test_source_id})
        if existing and existing["ids"]:
            collection.delete(ids=existing["ids"])
            print("   ✅ Test data cleaned up")

        # Summary
        print("\n" + "=" * 70)
        print("📊 INTEGRATION TEST SUMMARY:")
        print(
            f"   Total processing time: {extraction_time + chunking_time + embedding_time:.2f}s"
        )
        print(f"   Text extracted: {len(extracted_text):,} chars")
        print(f"   Chunks created: {len(chunks)}")
        print(f"   Headings found: {total_headings}")
        print(f"   Issues found: {len(issues)}")

        return len(issues) == 0

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases that might break the pipeline."""

    print("\n\n🧪 Testing Edge Cases")
    print("=" * 70)

    from src.processing.chunker import chunk_for_embedding_enhanced, chunk_for_llm

    edge_cases = [
        ("Empty text", ""),
        ("Only whitespace", "   \n\n   \t\t   "),
        ("Single word", "Klimaschutz"),
        ("Only numbers", "2030 55% 100 500 18"),
        (
            "Mixed languages",
            "Klimaschutz means climate protection und Mobilität means mobility",
        ),
        ("Huge heading", "A" * 150),
        ("Many short lines", "\n".join(["Line " + str(i) for i in range(100)])),
    ]

    all_passed = True

    for name, text in edge_cases:
        try:
            print(f"\n   Testing: {name}")
            chunks = chunk_for_embedding_enhanced(text, max_chars=5000)
            print(f"   ✅ Chunking: {len(chunks)} chunks")

            if chunks:  # Only test LLM chunking if we have chunks
                llm_chunks = chunk_for_llm(chunks)
                print(f"   ✅ LLM prep: {len(llm_chunks)} chunks")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Integration test for PDF extraction pipeline"
    )
    parser.add_argument("pdf_path", nargs="?", help="Path to PDF file to test")
    parser.add_argument("--edge-cases", action="store_true", help="Run edge case tests")

    args = parser.parse_args()

    all_passed = True

    if args.edge_cases or not args.pdf_path:
        edge_passed = test_edge_cases()
        all_passed = all_passed and edge_passed

    if args.pdf_path:
        pdf_passed = test_real_pdf_flow(args.pdf_path)
        all_passed = all_passed and pdf_passed
    elif not args.edge_cases:
        print("Usage: python test_integration.py [pdf_path] [--edge-cases]")
        print("\nExamples:")
        print("  python test_integration.py uploads/regensburg.pdf")
        print("  python test_integration.py --edge-cases")
        sys.exit(1)

    sys.exit(0 if all_passed else 1)
