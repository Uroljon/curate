#!/usr/bin/env python3
"""Test semantic-aware LLM chunking vs the old method."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processing import query_chunks, prepare_llm_chunks, prepare_semantic_llm_chunks, analyze_chunk_quality


def compare_chunking_methods(source_id: str):
    """Compare old vs new chunking methods."""
    # Get stored semantic chunks
    chunks = query_chunks("irrelevant", top_k=1000, source_id=source_id)
    raw_texts = [c["text"] for c in chunks]
    
    print(f"📊 Chunking Method Comparison for: {source_id}")
    print("="*70)
    print(f"Starting with {len(raw_texts)} semantic chunks from database")
    
    # Old method
    print("\n🔴 OLD METHOD (prepare_llm_chunks):")
    old_chunks = prepare_llm_chunks(raw_texts, max_chars=20000, min_chars=15000)
    old_quality = analyze_chunk_quality(old_chunks)
    
    print(f"  Chunks created: {old_quality['total_chunks']}")
    print(f"  Average size: {old_quality['avg_size']:,.0f} chars")
    print(f"  Mixed sections: {old_quality['mixed_sections']} chunks")
    print(f"  With headings: {old_quality['chunks_with_headings']} chunks")
    
    # New method
    print("\n🟢 NEW METHOD (prepare_semantic_llm_chunks):")
    new_chunks = prepare_semantic_llm_chunks(raw_texts, max_chars=20000, min_chars=15000)
    new_quality = analyze_chunk_quality(new_chunks)
    
    print(f"  Chunks created: {new_quality['total_chunks']}")
    print(f"  Average size: {new_quality['avg_size']:,.0f} chars")
    print(f"  Mixed sections: {new_quality['mixed_sections']} chunks")
    print(f"  With headings: {new_quality['chunks_with_headings']} chunks")
    
    # Detailed comparison
    print("\n📈 QUALITY COMPARISON:")
    print(f"  Mixed sections reduced by: {old_quality['mixed_sections'] - new_quality['mixed_sections']}")
    print(f"  Heading preservation: {new_quality['chunks_with_headings']}/{new_quality['total_chunks']} "
          f"vs {old_quality['chunks_with_headings']}/{old_quality['total_chunks']}")
    
    # Show first few chunks from each method
    print("\n🔍 CHUNK BOUNDARIES COMPARISON:")
    
    for i in range(min(3, len(old_chunks), len(new_chunks))):
        print(f"\n--- Chunk {i+1} ---")
        
        # Old method
        old_lines = old_chunks[i].strip().split('\n')
        print(f"OLD ({len(old_chunks[i])} chars):")
        print(f"  First: {old_lines[0][:70]}...")
        if len(old_lines) > 1:
            print(f"  Last:  {old_lines[-1][:70]}...")
        
        # New method
        if i < len(new_chunks):
            new_lines = new_chunks[i].strip().split('\n')
            print(f"NEW ({len(new_chunks[i])} chars):")
            print(f"  First: {new_lines[0][:70]}...")
            if len(new_lines) > 1:
                print(f"  Last:  {new_lines[-1][:70]}...")
    
    return old_quality, new_quality


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_semantic_chunking.py <source_id>")
        sys.exit(1)
    
    compare_chunking_methods(sys.argv[1])