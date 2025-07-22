"""
Test the fixes for semantic-aware chunking issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from semantic_chunker import extract_chunk_topic
from semantic_llm_chunker import prepare_semantic_llm_chunks_v2, has_topic_boundary, analyze_chunk_quality


def test_regex_fix():
    """Test that regex patterns don't capture extra text."""
    print("Testing Regex Pattern Fix:")
    print("-" * 60)
    
    test_cases = [
        "Handlungsfeld: Mobilität starts here with more text",
        "Handlungsfeld: Klimaschutz, sowie weitere Themen",
        "Handlungsfeld: Stadtentwicklung. Dies umfasst...",
        "Handlungsfeld: Digitalisierung\nNeue Zeile hier"
    ]
    
    for test in test_cases:
        result = extract_chunk_topic(test)
        print(f"Input: '{test[:50]}...'")
        print(f"  Extracted topic: '{result['topic']}'")
        print(f"  Confidence: {result['confidence']}")
        print()


def test_boundary_detection():
    """Test that boundary detection prevents topic mixing."""
    print("\nTesting Boundary Detection:")
    print("-" * 60)
    
    chunk1 = "Ende des Klimaschutz Abschnitts mit einigen Indikatoren."
    chunk2_with_boundary = "Handlungsfeld: Mobilität\nNeuer Abschnitt beginnt hier"
    chunk2_no_boundary = "Fortsetzung des vorherigen Themas ohne neue Markierung"
    
    print(f"Chunk 1 topic: {extract_chunk_topic(chunk1)['topic']}")
    print(f"\nBoundary with chunk2a? {has_topic_boundary(chunk1, chunk2_with_boundary)}")
    print(f"Boundary with chunk2b? {has_topic_boundary(chunk1, chunk2_no_boundary)}")


def test_real_world_scenario():
    """Test with realistic document chunks."""
    print("\n\nTesting Real-World Scenario:")
    print("=" * 60)
    
    # Simulate actual problematic case
    chunks = [
        # Klimaschutz section
        "Die CO2-Emissionen sollen bis 2030 um 55% reduziert werden. Dies erfordert massive Investitionen.",
        "Weitere Maßnahmen umfassen den Ausbau erneuerbarer Energien und die energetische Sanierung.",
        "Die Kosten werden auf 120 Millionen Euro geschätzt.",  # This might merge with next
        
        # New topic starts
        "Handlungsfeld: Mobilität\n\nDer Verkehrssektor muss grundlegend transformiert werden.",
        "Modal Split Ziel: 70% Umweltverbund bis 2040.",
        
        # Another topic
        "\n\nHandlungsfeld: Stadtentwicklung\n\nNachhaltige Quartiersentwicklung.",
        "Neue Wohngebiete sollen klimaneutral sein."
    ]
    
    # Test merging
    result = prepare_semantic_llm_chunks_v2(chunks, max_chars=400, min_chars=150)
    
    print(f"Input: {len(chunks)} chunks")
    print(f"Output: {len(result)} merged chunks\n")
    
    # Analyze for mixed topics
    quality = analyze_chunk_quality(result, "real_world_test")
    print(f"Topic Coherence:")
    print(f"  Single topic chunks: {quality['topic_coherence']['chunks_with_single_topic']}")
    print(f"  Multiple topic chunks: {quality['topic_coherence']['chunks_with_multiple_topics']}")
    print(f"  No topic chunks: {quality['topic_coherence']['chunks_with_no_topic']}")
    
    # Check each chunk
    print("\nChunk Analysis:")
    for i, chunk in enumerate(result):
        topic_info = extract_chunk_topic(chunk)
        print(f"\nChunk {i+1} (size: {len(chunk)}, topic: {topic_info['topic']}, conf: {topic_info['confidence']:.2f}):")
        
        # Count topic markers
        klimaschutz_count = chunk.count('Klimaschutz')
        mobility_count = chunk.count('Mobilität')
        stadtentw_count = chunk.count('Stadtentwicklung')
        
        topics_present = []
        if klimaschutz_count > 0:
            topics_present.append(f"Klimaschutz({klimaschutz_count})")
        if mobility_count > 0:
            topics_present.append(f"Mobilität({mobility_count})")
        if stadtentw_count > 0:
            topics_present.append(f"Stadtentwicklung({stadtentw_count})")
            
        print(f"  Topics present: {', '.join(topics_present) if topics_present else 'None (content only)'}")
        
        # Show preview
        preview = chunk[:150].replace('\n', ' | ')
        print(f"  Preview: {preview}...")


def test_keyword_fallback():
    """Test that keyword fallback is less aggressive."""
    print("\n\nTesting Keyword Fallback:")
    print("=" * 60)
    
    # Short chunk - should not get topic assigned
    short_chunk = "Dies umfasst CO2 Reduktion und Emission."  # < 500 chars
    result_short = extract_chunk_topic(short_chunk)
    print(f"Short chunk ({len(short_chunk)} chars): topic='{result_short['topic']}', conf={result_short['confidence']}")
    
    # Long chunk with few keywords - should not get topic
    long_sparse = "A" * 600 + " CO2 emission " + "B" * 400  # Low keyword density
    result_sparse = extract_chunk_topic(long_sparse)
    print(f"Long sparse chunk: topic='{result_sparse['topic']}', conf={result_sparse['confidence']}")
    
    # Long chunk with many keywords - should get topic
    long_dense = "Die Stadt fokussiert auf CO2 Reduktion. " * 10 + "Emission und Klimawandel sind zentral. " * 5
    result_dense = extract_chunk_topic(long_dense)
    print(f"Long dense chunk: topic='{result_dense['topic']}', conf={result_dense['confidence']}")


if __name__ == "__main__":
    test_regex_fix()
    test_boundary_detection()
    test_real_world_scenario()
    test_keyword_fallback()
    
    print("\n\n✅ All tests completed!")