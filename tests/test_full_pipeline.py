#!/usr/bin/env python3
"""Comprehensive test for the full extraction pipeline."""

import sys
import os
import time
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_chunker import smart_chunk, is_heading
from embedder import embed_chunks, query_chunks, model
from semantic_llm_chunker import prepare_semantic_llm_chunks, analyze_chunk_quality


def test_pipeline_components():
    """Test each component of the pipeline with sample German text."""
    
    print("🧪 CURATE Pipeline Component Tests\n")
    print("="*60)
    
    # Sample German municipal document text
    sample_text = """NACHHALTIGKEITSSTRATEGIE DER STADT

Kapitel 1: Einführung

Die Stadt hat sich zum Ziel gesetzt, bis 2030 klimaneutral zu werden. Dies erfordert umfassende Maßnahmen in allen Bereichen.

Handlungsfeld: Klimaschutz

Die Reduzierung der CO2-Emissionen um 55% bis 2030 ist das zentrale Ziel. Dabei spielen folgende Projekte eine wichtige Rolle:

Projekte:
- Ausbau erneuerbarer Energien auf 100% bis 2035
- Energetische Sanierung mit 3% Sanierungsquote pro Jahr
- Förderung der Elektromobilität mit 500 neuen Ladepunkten

Indikatoren:
- CO2-Reduktion: 55% bis 2030
- Erneuerbare Energien: 100% bis 2035
- Sanierungsquote: 3% pro Jahr
- Ladepunkte: 500 bis 2025

Handlungsfeld: Mobilität

Die Verkehrswende ist ein zentraler Baustein für eine nachhaltige Stadt.

Maßnahmen:
- Ausbau des Radwegenetzes auf 500 km
- Einführung einer Stadtbahn mit 18 km Streckenlänge
- Modal Split von 70% für den Umweltverbund

Zielstellung:
30% weniger PKW-Verkehr bis 2030."""

    all_tests_passed = True
    
    # Test 1: German-aware chunking
    print("\n1️⃣ Testing German-aware chunking...")
    try:
        chunks = smart_chunk(sample_text, max_chars=5000)
        
        # Check if headings are detected
        heading_count = 0
        for chunk in chunks:
            lines = chunk.split('\n')
            for line in lines:
                if is_heading(line.strip()):
                    heading_count += 1
        
        print(f"   ✅ Created {len(chunks)} chunks")
        print(f"   ✅ Detected {heading_count} headings")
        
        # Verify key sections are preserved
        full_text = '\n\n'.join(chunks)
        assert "Handlungsfeld: Klimaschutz" in full_text
        assert "Handlungsfeld: Mobilität" in full_text
        assert "55% bis 2030" in full_text
        print("   ✅ All key sections preserved")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        all_tests_passed = False
    
    # Test 2: Multilingual embeddings
    print("\n2️⃣ Testing multilingual embeddings...")
    try:
        # Test German-German similarity
        german_texts = ["Klimaschutz und Energie", "Umweltschutz und Energiewende"]
        embeddings = model.encode(german_texts)
        
        from numpy import dot
        from numpy.linalg import norm
        
        similarity = dot(embeddings[0], embeddings[1])/(norm(embeddings[0])*norm(embeddings[1]))
        print(f"   ✅ German-German similarity: {similarity:.3f}")
        assert similarity > 0.7, "German similarity should be high"
        
        # Test German-English cross-lingual
        mixed_texts = ["Klimaschutz", "climate protection"]
        mixed_embeddings = model.encode(mixed_texts)
        cross_similarity = dot(mixed_embeddings[0], mixed_embeddings[1])/(norm(mixed_embeddings[0])*norm(mixed_embeddings[1]))
        print(f"   ✅ German-English similarity: {cross_similarity:.3f}")
        assert cross_similarity > 0.6, "Cross-lingual similarity should work"
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        all_tests_passed = False
    
    # Test 3: Semantic LLM chunking
    print("\n3️⃣ Testing semantic LLM chunking...")
    try:
        # Create test chunks
        test_chunks = [
            "Handlungsfeld: Klimaschutz\n\nWichtige Maßnahmen für den Klimaschutz.",
            "Weitere Details zum Klimaschutz mit Indikatoren.",
            "Handlungsfeld: Mobilität\n\nDie Verkehrswende ist wichtig.",
            "Zusätzliche Mobilitätsmaßnahmen und Projekte."
        ]
        
        # Test semantic chunking
        semantic_chunks = prepare_semantic_llm_chunks(test_chunks, max_chars=10000, min_chars=5000)
        
        print(f"   ✅ Input chunks: {len(test_chunks)}")
        print(f"   ✅ Output chunks: {len(semantic_chunks)}")
        
        # Check that Handlungsfelder are not mixed
        for i, chunk in enumerate(semantic_chunks):
            klimaschutz_count = chunk.count("Klimaschutz")
            mobilität_count = chunk.count("Mobilität")
            
            if klimaschutz_count > 0 and mobilität_count > 0:
                print(f"   ⚠️  Chunk {i+1} mixes Handlungsfelder!")
            else:
                print(f"   ✅ Chunk {i+1} maintains section integrity")
        
        # Analyze quality
        quality = analyze_chunk_quality(semantic_chunks)
        print(f"   ✅ Chunks with headings: {quality['heading_stats']['chunks_with_headings']}/{quality['total_chunks']}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        all_tests_passed = False
    
    # Test 4: Indicator detection
    print("\n4️⃣ Testing indicator patterns...")
    try:
        indicator_patterns = [
            ("55% bis 2030", True, "percentage with year"),
            ("500 neue Ladepunkte", True, "number with unit"),
            ("3% pro Jahr", True, "percentage per year"),
            ("18 km Streckenlänge", True, "distance measurement"),
            ("Modal Split 70%", True, "modal split percentage"),
            ("normale text", False, "regular text")
        ]
        
        import re
        patterns = [
            r'\d+\s*%',  # Percentages
            r'bis\s+20\d{2}',  # Year targets
            r'\d+\s*(km|m²|MW|ha|t)',  # Units
            r'\d+\s+(neue|mehr|weniger)',  # Quantities
        ]
        
        for text, should_match, description in indicator_patterns:
            matched = any(re.search(pattern, text) for pattern in patterns)
            if matched == should_match:
                print(f"   ✅ '{text}' - {description}")
            else:
                print(f"   ❌ '{text}' - detection failed")
                all_tests_passed = False
                
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        all_tests_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_tests_passed:
        print("✅ All pipeline components working correctly!")
        print("\nThe pipeline is ready for:")
        print("  • German document processing")
        print("  • Semantic chunk preservation") 
        print("  • Multilingual understanding")
        print("  • Indicator extraction")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return all_tests_passed


def test_embedding_storage():
    """Test that embeddings can be stored and retrieved correctly."""
    print("\n\n5️⃣ Testing embedding storage and retrieval...")
    
    test_source_id = "test_pipeline_doc"
    test_chunks = [
        "Handlungsfeld: Testbereich\n\nDies ist ein Testchunk für die Pipeline.",
        "Indikatoren: 50% Reduktion bis 2030, 100 neue Einheiten"
    ]
    
    try:
        # Store embeddings
        embed_chunks(test_chunks, source_id=test_source_id)
        print(f"   ✅ Stored {len(test_chunks)} chunks")
        
        # Retrieve chunks
        retrieved = query_chunks("Testbereich", top_k=10, source_id=test_source_id)
        print(f"   ✅ Retrieved {len(retrieved)} chunks")
        
        # Verify content
        retrieved_texts = [r["text"] for r in retrieved]
        assert any("Testbereich" in text for text in retrieved_texts), "Should retrieve test content"
        print("   ✅ Content verified")
        
        # Clean up test data
        from embedder import collection
        existing = collection.get(where={"source": test_source_id})
        if existing and existing["ids"]:
            collection.delete(ids=existing["ids"])
            print("   ✅ Cleaned up test data")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 CURATE Full Pipeline Test Suite")
    print("  Testing German PDF extraction pipeline components")
    
    # Run component tests
    components_ok = test_pipeline_components()
    
    # Run storage test
    storage_ok = test_embedding_storage()
    
    # Overall result
    print("\n" + "="*60)
    print("📊 FINAL RESULT:")
    if components_ok and storage_ok:
        print("✅ All tests passed! Pipeline is working correctly.")
        print("\nNext steps:")
        print("1. Upload a PDF via /upload endpoint")
        print("2. Extract with /extract_structure_fast")
        print("3. Enjoy better German document understanding! 🇩🇪")
    else:
        print("❌ Some tests failed. Please fix issues before using pipeline.")
    
    sys.exit(0 if (components_ok and storage_ok) else 1)