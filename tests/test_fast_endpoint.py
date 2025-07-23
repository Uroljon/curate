#!/usr/bin/env python3
"""Quick test for the fast extraction endpoint."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_fast_endpoint_config():
    """Test that fast endpoint configuration is correct."""
    from src.core import FAST_EXTRACTION_ENABLED, FAST_EXTRACTION_MAX_CHUNKS

    print("Testing fast extraction configuration...")
    print(f"✅ FAST_EXTRACTION_ENABLED: {FAST_EXTRACTION_ENABLED}")
    print(f"✅ FAST_EXTRACTION_MAX_CHUNKS: {FAST_EXTRACTION_MAX_CHUNKS}")

    assert FAST_EXTRACTION_ENABLED == True, "Fast extraction should be enabled"
    assert FAST_EXTRACTION_MAX_CHUNKS > 0, "Should have a chunk limit for speed"

    print("\n✅ Configuration test passed!")


def test_imports():
    """Test that all required imports work."""
    print("\nTesting imports...")

    try:
        from main import extract_structure_fast

        print("✅ Fast endpoint function imported successfully")

        from src.extraction import extract_structures_with_retry

        print("✅ Single-pass extraction function available")

        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Fast Extraction Setup\n")

    test_fast_endpoint_config()

    if test_imports():
        print("\n✅ All tests passed! Fast extraction is ready to use.")
        print("\n📝 Usage:")
        print("   1. Start the server: uvicorn main:app --reload")
        print("   2. Upload a PDF via /upload endpoint")
        print("   3. Compare extraction methods:")
        print("      - Original: GET /extract_structure?source_id=document.pdf")
        print("      - Fast:     GET /extract_structure_fast?source_id=document.pdf")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
