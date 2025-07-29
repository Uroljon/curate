#!/usr/bin/env python3
"""
Test script for verifying PyMuPDF table extraction functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.parser import extract_text_with_ocr_fallback


def test_table_extraction(pdf_path: str):
    """Test table extraction on a PDF file."""
    print(f"\nTesting table extraction on: {pdf_path}\n")

    try:
        # Extract text with table detection
        page_aware_text, metadata = extract_text_with_ocr_fallback(pdf_path)

        # Print extraction statistics
        print("Extraction Statistics:")
        print(f"  Total pages: {metadata['total_pages']}")
        print(f"  Pages with content: {metadata['pages_with_content']}")
        print(f"  OCR pages: {metadata['ocr_pages']}")

        # Print table statistics
        table_stats = metadata.get("table_extraction", {})
        print("\nTable Detection Statistics:")
        print(f"  Total tables found: {table_stats.get('total_tables', 0)}")
        print(f"  Pages with tables: {table_stats.get('pages_with_tables', [])}")
        print(f"  Table conversion errors: {table_stats.get('table_errors', 0)}")

        # Show sample of extracted content with tables
        if table_stats.get("total_tables", 0) > 0:
            print("\nSample pages with tables:")
            for text, page_num in page_aware_text:
                if page_num in table_stats.get("pages_with_tables", []):
                    # Look for table markers in the text
                    if "<!-- Table" in text:
                        print(f"\n--- Page {page_num} ---")
                        # Extract just the table portion for display
                        start = text.find("<!-- Table")
                        end = text.find("<!-- End of table -->", start) + len(
                            "<!-- End of table -->"
                        )
                        if start != -1 and end > start:
                            table_content = text[start:end]
                            print(
                                table_content[:500] + "..."
                                if len(table_content) > 500
                                else table_content
                            )
                        break

        print("\n✅ Table extraction test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error during table extraction: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if Path(pdf_path).exists():
            test_table_extraction(pdf_path)
        else:
            print(f"Error: PDF file not found: {pdf_path}")
    else:
        print("Usage: python test_table_extraction.py <path_to_pdf>")
        print("\nThis script tests the table extraction functionality.")
        print("It will show statistics about tables found and sample table content.")
