"""
Test indicator-aware chunking functionality.
"""

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processing.chunker import (
    chunk_for_embedding_enhanced,
    contains_indicator_context,
    find_safe_split_point,
    split_large_chunk,
)


def test_indicator_preservation_in_chunking():
    """Test that indicators are preserved with their context during chunking."""

    # Text where indicator might get split at boundary
    text = """Die Stadtverwaltung hat einen Plan vorgelegt. Die Emissionen sollen
um 55% bis 2030 reduziert werden. Investitionen von 120 Millionen Euro sind geplant."""

    # Force small chunks to test splitting
    chunks = split_large_chunk(text, max_chars=100)

    # Check that indicators are not split
    full_text = " ".join(chunks)
    assert "55% bis 2030" in full_text

    # Verify each indicator is in exactly one chunk
    indicator_found = False
    for chunk in chunks:
        if "55% bis 2030" in chunk:
            assert not indicator_found, "Indicator found in multiple chunks"
            indicator_found = True

    assert indicator_found, "Indicator not found in any chunk"


def test_chunk_for_embedding_enhanced_with_indicators():
    """Test chunk_for_embedding_enhanced preserves indicators in real document."""

    document = """Handlungsfeld: Klimaschutz

Die Stadt setzt sich das Ziel, die CO2-Emissionen um 55% bis 2030 zu reduzieren.

Maßnahmen:
- Ausbau der Photovoltaik auf 25 MW bis 2025
- Energetische Sanierung mit 45 Millionen Euro Budget
- 500 neue Ladepunkte für Elektrofahrzeuge

Handlungsfeld: Mobilität

Der Modal Split soll 30% PKW und 70% Umweltverbund erreichen.
Investitionen: 250 Millionen Euro bis 2028."""

    chunks = chunk_for_embedding_enhanced(document, max_chars=300)

    # All indicators should be preserved
    indicators = [
        "55% bis 2030",
        "25 MW bis 2025",
        "45 Millionen Euro",
        "500 neue Ladepunkte",
        "30% PKW",
        "250 Millionen Euro bis 2028",
    ]

    for indicator in indicators:
        found = any(indicator in chunk for chunk in chunks)
        assert found, f"Indicator '{indicator}' not found in chunks"


def test_indicator_at_chunk_boundary():
    """Test indicator detection at chunk boundaries."""

    # Create text where indicator would be at exact boundary
    text = "A" * 95 + " Die Reduktion beträgt 55% bis 2030 im Vergleich zu 1990."

    # This would normally split right before "55%"
    chunks = split_large_chunk(text, max_chars=100)

    # Verify indicator is kept together
    indicator_chunks = [c for c in chunks if "55% bis 2030" in c]
    assert len(indicator_chunks) == 1, "Indicator was split across chunks"


if __name__ == "__main__":
    test_indicator_preservation_in_chunking()
    print("✅ Test 1 passed: Indicator preservation")

    test_chunk_for_embedding_enhanced_with_indicators()
    print("✅ Test 2 passed: Chunk for embedding enhanced with indicators")

    test_indicator_at_chunk_boundary()
    print("✅ Test 3 passed: Indicator at boundary")

    print("\n✅ All indicator chunking tests passed!")
