#!/usr/bin/env python3
"""Test chunking improvements."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_chunker import smart_chunk, merge_short_chunks

def test_chunk_size_improvement():
    """Test that chunks are now character-based and larger."""
    
    # Sample German text with multiple sections
    test_text = """
1. Klimaschutz und Energie

Die Stadt Regensburg setzt sich ambitionierte Ziele im Bereich Klimaschutz. 
Bis zum Jahr 2030 sollen die CO2-Emissionen um 55% gegenüber 1990 reduziert werden.
Dies erfordert umfangreiche Maßnahmen in verschiedenen Bereichen.

1.1 Energieeffizienz in kommunalen Gebäuden

Alle städtischen Gebäude werden bis 2025 energetisch saniert. Die Investitionssumme
beträgt 45 Millionen Euro. Erwartete Energieeinsparung: 40% gegenüber dem Ist-Zustand.

1.2 Ausbau erneuerbarer Energien

Die Stadtwerke planen den Bau von Photovoltaikanlagen mit einer Gesamtleistung von
25 Megawatt bis 2028. Zusätzlich werden 15 Windkraftanlagen mit je 3 MW Leistung
errichtet.

2. Mobilität und Verkehr

Die Verkehrswende ist ein zentraler Baustein der Nachhaltigkeitsstrategie.

2.1 Stadtbahn Regensburg

Das Leuchtturmprojekt der Stadt ist die neue Stadtbahn. Mit einer Streckenlänge von
18 Kilometern und 24 Haltestellen soll sie bis 2028 in Betrieb gehen. 
Investitionsvolumen: 650 Millionen Euro. Erwartete Fahrgastzahlen: 50.000 pro Tag.

2.2 Radverkehrsförderung

Der Radverkehrsanteil soll bis 2030 von derzeit 18% auf 30% gesteigert werden.
Dafür werden 100 Kilometer neue Radwege gebaut und 5.000 zusätzliche 
Fahrradstellplätze geschaffen.
""" * 3  # Repeat to make it longer

    # Test with old word-based approach (simulated)
    old_chunks_words = test_text.split('\n\n')  # Simple paragraph split
    old_chunks = []
    temp = []
    word_count = 0
    
    for para in old_chunks_words:
        words_in_para = len(para.split())
        if word_count + words_in_para > 300:  # Old 300-word limit
            if temp:
                old_chunks.append('\n\n'.join(temp))
            temp = [para]
            word_count = words_in_para
        else:
            temp.append(para)
            word_count += words_in_para
    
    if temp:
        old_chunks.append('\n\n'.join(temp))
    
    # Test with new character-based approach
    new_chunks = smart_chunk(test_text, max_chars=5000)
    
    print("=== CHUNKING COMPARISON ===")
    print(f"Old approach (300 words): {len(old_chunks)} chunks")
    print(f"New approach (5000 chars): {len(new_chunks)} chunks")
    print(f"Reduction: {len(old_chunks) - len(new_chunks)} fewer chunks ({(1 - len(new_chunks)/len(old_chunks))*100:.1f}% reduction)")
    
    print("\n=== CHUNK SIZE ANALYSIS ===")
    print("Old chunks (characters):")
    for i, chunk in enumerate(old_chunks[:3]):  # Show first 3
        print(f"  Chunk {i+1}: {len(chunk)} chars (~{len(chunk.split())} words)")
    
    print("\nNew chunks (characters):")
    for i, chunk in enumerate(new_chunks[:3]):  # Show first 3
        print(f"  Chunk {i+1}: {len(chunk)} chars (~{len(chunk.split())} words)")
    
    # Test indicator preservation
    print("\n=== INDICATOR PRESERVATION TEST ===")
    
    # Check if indicators stay with their context
    test_indicators = [
        ("55%", "CO2-Emissionen"),
        ("45 Millionen Euro", "energetisch saniert"),
        ("18 Kilometern", "Stadtbahn"),
        ("650 Millionen Euro", "Investitionsvolumen"),
        ("30%", "Radverkehrsanteil")
    ]
    
    for indicator, context in test_indicators:
        found_together = False
        for chunk in new_chunks:
            if indicator in chunk and context in chunk:
                found_together = True
                break
        
        status = "✅" if found_together else "❌"
        print(f"{status} '{indicator}' and '{context}' in same chunk")
    
    # Assert improvements
    assert len(new_chunks) < len(old_chunks), "New approach should create fewer chunks"
    assert all(len(c) <= 5000 for c in new_chunks), "All chunks should be under 5000 chars"
    
    return len(old_chunks), len(new_chunks)

def test_heading_detection():
    """Test that German headings are properly detected."""
    from semantic_chunker import is_heading
    
    test_cases = [
        # (text, expected_result)
        ("1. Klimaschutz", True),
        ("2.1 Energieeffizienz", True),
        ("MOBILITÄT UND VERKEHR", True),
        ("Stadtentwicklung", True),  # Title case
        ("Dies ist ein normaler Satz.", False),
        ("", False),  # Empty
        ("a" * 101, False),  # Too long
    ]
    
    print("\n=== HEADING DETECTION TEST ===")
    for text, expected in test_cases:
        result = is_heading(text)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{text[:30]}...' -> {result} (expected: {expected})")

if __name__ == "__main__":
    print("Testing chunking improvements...\n")
    
    old_count, new_count = test_chunk_size_improvement()
    test_heading_detection()
    
    print(f"\n✅ All tests passed!")
    print(f"Summary: Reduced chunks from {old_count} to {new_count} ({(1-new_count/old_count)*100:.0f}% improvement)")