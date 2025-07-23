#!/usr/bin/env python3
"""Debug script to see how chunks are being merged."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extraction import prepare_llm_chunks
from src.processing import query_chunks


def show_chunk_boundaries(source_id: str):
    # Get stored semantic chunks
    chunks = query_chunks("irrelevant", top_k=1000, source_id=source_id)
    raw_texts = [c["text"] for c in chunks]

    print(f"📊 Chunk Boundary Analysis for: {source_id}")
    print("=" * 60)

    # Show first few semantic chunks
    print("\n🔍 SEMANTIC CHUNKS (from smart_chunk):")
    for i, chunk in enumerate(raw_texts[:5]):
        lines = chunk.strip().split("\n")
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        print(f"  First line: {lines[0][:80]}...")
        if len(lines) > 1:
            print(f"  Last line: {lines[-1][:80]}...")

        # Check for heading markers
        if any(
            marker in chunk
            for marker in ["Handlungsfeld:", "Maßnahmen:", "Projekte:", "Indikatoren:"]
        ):
            print("  ⚠️  Contains section markers!")

    # Show how they get merged
    optimized = prepare_llm_chunks(raw_texts, max_chars=20000, min_chars=15000)

    print("\n\n🔀 LLM CHUNKS (after prepare_llm_chunks):")
    for i, chunk in enumerate(optimized[:3]):
        lines = chunk.strip().split("\n")
        print(f"\nLLM Chunk {i+1} ({len(chunk)} chars):")
        print(f"  First line: {lines[0][:80]}...")
        if len(lines) > 1:
            print(f"  Last line: {lines[-1][:80]}...")

        # Count how many section markers are in this chunk
        markers = ["Handlungsfeld:", "Maßnahmen:", "Projekte:", "Indikatoren:"]
        marker_count = sum(chunk.count(m) for m in markers)
        if marker_count > 1:
            print(
                f"  ⚠️  Contains {marker_count} section markers - mixing different sections!"
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_chunk_boundaries.py <source_id>")
        sys.exit(1)

    show_chunk_boundaries(sys.argv[1])
