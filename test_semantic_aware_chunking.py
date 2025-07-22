"""
Test script for semantic-aware LLM chunking.
Tests that topics are kept together and not mixed across chunks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from semantic_chunker import extract_chunk_topic
from semantic_llm_chunker import prepare_semantic_llm_chunks_v2, analyze_chunk_quality


def test_topic_extraction():
    """Test that topic extraction works correctly."""
    
    test_chunks = [
        "Handlungsfeld: Klimaschutz\n\nDie Stadt setzt sich ehrgeizige Ziele...",
        "Projekte:\n- CO2-Neutralität bis 2035\n- Ausbau erneuerbarer Energien",
        "Die Emissionen sollen um 55% bis 2030 reduziert werden.",
        "\n\nHandlungsfeld: Mobilität\n\nDer Verkehrssektor muss transformiert werden.",
        "Maßnahmen:\n- Ausbau der Radwege\n- Förderung des ÖPNV"
    ]
    
    print("Testing topic extraction:")
    print("-" * 60)
    
    for i, chunk in enumerate(test_chunks):
        metadata = extract_chunk_topic(chunk)
        print(f"\nChunk {i+1}:")
        print(f"  Topic: {metadata['topic']}")
        print(f"  Subtopic: {metadata['subtopic']}")
        print(f"  Confidence: {metadata['confidence']:.2f}")
        print(f"  Preview: {chunk[:50].replace(chr(10), ' ')}...")


def test_semantic_aware_merging():
    """Test that semantic-aware merging keeps topics together."""
    
    # Create test chunks that simulate a real document structure
    test_chunks = [
        # Klimaschutz section (should stay together)
        "Handlungsfeld: Klimaschutz\n\nDie Stadt Regensburg setzt sich ehrgeizige Klimaziele.",
        "Die CO2-Emissionen sollen bis 2030 um 55% reduziert werden.",
        "Projekte:\n- Energetische Sanierung öffentlicher Gebäude\n- Ausbau Photovoltaik",
        "Indikatoren:\n- CO2-Reduktion 55% bis 2030\n- 100% Ökostrom bis 2035",
        
        # Mobilität section (should stay together, separate from Klimaschutz)
        "\n\nHandlungsfeld: Mobilität\n\nDer Verkehrssektor muss grundlegend transformiert werden.",
        "Der Modal Split soll sich zugunsten des Umweltverbunds verschieben.",
        "Maßnahmen:\n- Ausbau Radwegenetz auf 500km\n- Einführung Stadtbahn",
        "Indikatoren:\n- Modal Split 70% Umweltverbund bis 2040\n- 30% weniger PKW-Verkehr",
        
        # Stadtentwicklung section
        "\n\nHandlungsfeld: Stadtentwicklung\n\nNachhaltige Quartiersentwicklung ist zentral.",
        "Neue Quartiere sollen klimaneutral entwickelt werden."
    ]
    
    print("\n\nTesting semantic-aware merging:")
    print("=" * 60)
    print(f"Input: {len(test_chunks)} chunks")
    
    # Test with small max_chars to force some splitting
    merged_chunks = prepare_semantic_llm_chunks_v2(
        test_chunks, 
        max_chars=500,  # Small to test splitting behavior
        min_chars=200
    )
    
    print(f"Output: {len(merged_chunks)} merged chunks")
    
    # Analyze the results
    quality_analysis = analyze_chunk_quality(merged_chunks, "test")
    
    print("\nTopic Coherence Analysis:")
    print(f"  Chunks with single topic: {quality_analysis['topic_coherence']['chunks_with_single_topic']}")
    print(f"  Chunks with multiple topics: {quality_analysis['topic_coherence']['chunks_with_multiple_topics']}")
    print(f"  Chunks with no topic: {quality_analysis['topic_coherence']['chunks_with_no_topic']}")
    print(f"  Topics found: {quality_analysis['topic_coherence']['topics_found']}")
    
    print("\nMerged chunks:")
    print("-" * 60)
    for i, chunk in enumerate(merged_chunks):
        topic_info = extract_chunk_topic(chunk)
        print(f"\nChunk {i+1} (size: {len(chunk)} chars, topic: {topic_info['topic']}):")
        # Show first 150 chars
        preview = chunk[:150].replace('\n', ' | ')
        print(f"  {preview}...")
        
        # Check for mixed topics
        if "Handlungsfeld:" in chunk:
            count = chunk.count("Handlungsfeld:")
            if count > 1:
                print(f"  ⚠️ WARNING: Contains {count} different Handlungsfelder!")


def test_edge_cases():
    """Test edge cases like very large topics."""
    
    print("\n\nTesting edge cases:")
    print("=" * 60)
    
    # Test 1: Single topic that's too large
    large_topic = ["Handlungsfeld: Klimaschutz\n\n" + "X" * 100 for _ in range(10)]
    merged = prepare_semantic_llm_chunks_v2(large_topic, max_chars=500, min_chars=200)
    print(f"\nTest 1 - Large single topic: {len(large_topic)} chunks -> {len(merged)} chunks")
    
    # Test 2: Many small chunks from same topic
    small_chunks = ["Klimaschutz " + str(i) for i in range(20)]
    merged = prepare_semantic_llm_chunks_v2(small_chunks, max_chars=500, min_chars=200)
    print(f"Test 2 - Many small chunks: {len(small_chunks)} chunks -> {len(merged)} chunks")
    
    # Test 3: Chunks with no clear topic
    no_topic_chunks = ["Random text " + str(i) for i in range(5)]
    merged = prepare_semantic_llm_chunks_v2(no_topic_chunks, max_chars=500, min_chars=200)
    print(f"Test 3 - No topic chunks: {len(no_topic_chunks)} chunks -> {len(merged)} chunks")


if __name__ == "__main__":
    test_topic_extraction()
    test_semantic_aware_merging()
    test_edge_cases()
    
    print("\n\n✅ All tests completed!")