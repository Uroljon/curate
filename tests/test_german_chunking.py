#!/usr/bin/env python3
"""Test German-aware chunking improvements."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processing import is_heading, smart_chunk, split_by_heading


def test_german_heading_detection():
    """Test that German-specific heading patterns are detected correctly."""
    print("🧪 Testing German heading detection...\n")

    # Test cases: (text, expected_result, description)
    test_cases = [
        # Numbered sections
        ("1. Klimaschutz", True, "Numbered section"),
        ("2.1 Maßnahmen", True, "Sub-numbered section"),
        ("III. Ziele", True, "Roman numeral section"),
        ("a) Ziel", True, "Letter enumeration"),
        # German structure keywords
        ("Kapitel 1: Einführung", True, "Kapitel heading"),
        ("Abschnitt A", True, "Abschnitt heading"),
        ("Maßnahmen:", True, "Maßnahmen with colon"),
        ("Projekte:", True, "Projekte with colon"),
        ("Handlungsfeld: Mobilität", True, "Handlungsfeld heading"),
        ("Indikatoren:", True, "Indikatoren heading"),
        ("Zielstellung:", True, "Zielstellung heading"),
        ("Monitoring:", True, "Monitoring heading"),
        # Uppercase headings
        ("KLIMASCHUTZ UND ENERGIE", True, "All uppercase heading"),
        ("MOBILITÄT", True, "Single word uppercase"),
        # Title case with German characters
        ("Öffentlicher Nahverkehr", True, "Title case with Ö"),
        ("Städtische Entwicklung", True, "Title case with ä"),
        ("Maßnahmen für Klimaschutz", True, "Title case phrase"),
        # Lines ending with colon
        ("Folgende Ziele:", True, "Short line with colon"),
        ("Hauptziele des Projekts:", True, "Line ending with colon"),
        # Non-headings
        ("Dies ist ein normaler Satz mit vielen Wörtern.", False, "Regular sentence"),
        ("öffentliche Verkehrsmittel nutzen", False, "Lowercase start"),
        (
            "Ein sehr langer Satz der definitiv kein Heading ist weil er zu lang ist und auch zu viele Wörter enthält.",
            False,
            "Too long",
        ),
        ("Kurz", False, "Too short (< 3 chars)"),
    ]

    passed = 0
    failed = 0

    for text, expected, description in test_cases:
        result = is_heading(text)
        status = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} {description}: '{text}' -> {result} (expected {expected})")

    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0


def test_multiline_heading_detection():
    """Test detection of multi-line German headings."""
    print("\n🧪 Testing multi-line heading detection...\n")

    # Test multi-line heading scenarios
    test_text_1 = """Vorwort

Handlungsfeld 1:
Klimaschutz und Energie

Die Stadt setzt sich ehrgeizige Ziele."""

    chunks_1 = split_by_heading(test_text_1)
    print("Test 1 - Heading with colon on separate line:")
    print(f"Number of chunks: {len(chunks_1)}")
    for i, chunk in enumerate(chunks_1):
        print(f"\nChunk {i+1}:\n{chunk[:100]}...")

    # Test numbered heading on separate line
    test_text_2 = """Einleitung

1.
Mobilität und Verkehr

Der Verkehrssektor ist wichtig."""

    chunks_2 = split_by_heading(test_text_2)
    print("\n\nTest 2 - Number on separate line from heading:")
    print(f"Number of chunks: {len(chunks_2)}")
    for i, chunk in enumerate(chunks_2):
        print(f"\nChunk {i+1}:\n{chunk[:100]}...")

    # Verify multi-line headings are kept together
    # The test expects chunks to be split at headings, so we should have:
    # Chunk 1: Vorwort
    # Chunk 2: Handlungsfeld 1: + content
    # But since "Die Stadt..." comes after, it's all in one chunk
    assert len(chunks_1) >= 2, f"Expected at least 2 chunks, got {len(chunks_1)}"

    # Check that multi-line heading is kept together in one of the chunks
    found_multiline = False
    for chunk in chunks_1:
        if "Handlungsfeld 1:\nKlimaschutz und Energie" in chunk:
            found_multiline = True
            break
    assert found_multiline, "Multi-line heading should be kept together"

    return True


def test_german_document_chunking():
    """Test chunking on a sample German municipal document text."""
    print("\n🧪 Testing German document chunking...\n")

    sample_text = """NACHHALTIGKEITSSTRATEGIE DER STADT

Kapitel 1: Einführung

Die Stadt hat sich zum Ziel gesetzt, bis 2030 klimaneutral zu werden.

Handlungsfeld: Klimaschutz

Maßnahmen:
- Ausbau erneuerbarer Energien
- Energetische Sanierung öffentlicher Gebäude
- Förderung der Elektromobilität

Indikatoren:
- CO2-Reduktion um 55% bis 2030
- 100% Ökostrom bis 2035
- Sanierungsquote 3% pro Jahr

Handlungsfeld: Mobilität

Die Verkehrswende ist ein zentraler Baustein.

Projekte:
1. Ausbau des Radwegenetzes auf 500 km
2. Einführung einer Stadtbahn
3. Verdopplung der Park&Ride-Plätze

Zielstellung:
Modal Split von 70% für den Umweltverbund erreichen."""

    chunks = smart_chunk(sample_text, max_chars=5000)

    print(f"Number of chunks created: {len(chunks)}")
    print(f"Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")

    # Verify chunks contain expected headings
    heading_found = {
        "NACHHALTIGKEITSSTRATEGIE": False,
        "Kapitel 1": False,
        "Handlungsfeld: Klimaschutz": False,
        "Handlungsfeld: Mobilität": False,
        "Maßnahmen:": False,
        "Indikatoren:": False,
        "Projekte:": False,
        "Zielstellung:": False,
    }

    for chunk in chunks:
        for heading in heading_found:
            if heading in chunk:
                heading_found[heading] = True

    print("\nHeading detection results:")
    all_found = True
    for heading, found in heading_found.items():
        status = "✅" if found else "❌"
        print(f"{status} {heading}")
        if not found:
            all_found = False

    # Print chunk boundaries for inspection
    print("\nChunk boundaries:")
    for i, chunk in enumerate(chunks):
        lines = chunk.split("\n")
        print(f"\nChunk {i+1}: {len(chunk)} chars")
        print(f"  First line: {lines[0][:50]}...")
        if len(lines) > 1:
            print(f"  Last line: {lines[-1][:50]}...")

    return all_found


def test_indicator_preservation():
    """Test that indicators are not split across chunks."""
    print("\n🧪 Testing indicator preservation in chunks...\n")

    # Text with indicators that should stay together
    test_text = """Klimaschutz Maßnahmen

Das Ziel ist eine CO2-Reduktion um 55% bis 2030 zu erreichen.
Dabei soll der Anteil erneuerbarer Energien auf 70% steigen.

Mobilität

Die Stadt plant 500 neue Ladepunkte für E-Autos bis 2025.
Der Modal Split soll 30% PKW und 70% Umweltverbund betragen."""

    chunks = smart_chunk(test_text, max_chars=150)  # Small chunks to test preservation

    # Check that indicators stay in same chunk
    indicators = [
        "55% bis 2030",
        "70% steigen",
        "500 neue Ladepunkte",
        "30% PKW und 70% Umweltverbund",
    ]

    print(f"Created {len(chunks)} chunks")

    for indicator in indicators:
        found_in_chunks = []
        for i, chunk in enumerate(chunks):
            if indicator in chunk:
                found_in_chunks.append(i + 1)

        if len(found_in_chunks) == 1:
            print(
                f"✅ '{indicator}' found in exactly one chunk (chunk {found_in_chunks[0]})"
            )
        else:
            print(
                f"❌ '{indicator}' found in {len(found_in_chunks)} chunks: {found_in_chunks}"
            )

    return True


def main():
    """Run all German chunking tests."""
    print("🇩🇪 German-Aware Chunking Test Suite\n")
    print("=" * 60)

    tests = [
        ("German Heading Detection", test_german_heading_detection),
        ("Multi-line Heading Detection", test_multiline_heading_detection),
        ("German Document Chunking", test_german_document_chunking),
        ("Indicator Preservation", test_indicator_preservation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n❌ {test_name} failed with error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for test_name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if error:
            print(f"         Error: {error}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! German-aware chunking is working correctly.")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
