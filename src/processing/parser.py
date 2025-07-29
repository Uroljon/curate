import tempfile

import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from spellchecker import SpellChecker

from src.core import (
    MIN_CHARS_FOR_VALID_PAGE,
    SPELL_CHECK_LANGUAGES,
    SPELL_CHECK_THRESHOLD,
    SUPPORTED_LANGUAGES,
    SYMBOL_FILTER_THRESHOLD,
)
from src.utils.text import (
    clean_ocr_text,
    clean_text,
    identify_headers_footers,
    remove_structural_noise,
)

# Initialize spell checkers based on config
spell_checkers = {
    lang: SpellChecker(language=spell_lang)
    for lang, spell_lang in SPELL_CHECK_LANGUAGES.items()
}


def extract_text_with_ocr_fallback(pdf_path: str) -> tuple[list[tuple[str, int]], dict]:
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    page_texts = [""] * total_pages
    scanned_pages = []
    table_stats: dict[str, int | list[int]] = {
        "total_tables": 0,
        "pages_with_tables": [],
        "table_errors": 0,
    }

    for i, page in enumerate(doc):
        # Extract regular text
        text = page.get_text().strip()

        # Extract tables
        table_markdown = ""
        try:
            table_finder = page.find_tables()
            tables = table_finder.tables  # Get the actual list of tables
            if tables:
                pages_list = table_stats["pages_with_tables"]
                if isinstance(pages_list, list):
                    pages_list.append(i + 1)  # 1-based page number
                total_tables = table_stats["total_tables"]
                if isinstance(total_tables, int):
                    table_stats["total_tables"] = total_tables + len(tables)

                # Convert each table to markdown
                for table_idx, table in enumerate(tables):
                    try:
                        # Extract raw table data
                        table_data = table.extract()

                        if table_data and len(table_data) > 0:
                            # Try to get better headers
                            headers = None
                            try:
                                # Check if PyMuPDF detected headers
                                if hasattr(table, "header") and hasattr(
                                    table.header, "names"
                                ):
                                    header_names = table.header.names
                                    # Check if headers are generic (Col1, Col2, etc.)
                                    if header_names and not all(
                                        name.startswith("Col") for name in header_names
                                    ):
                                        headers = header_names
                            except:
                                pass

                            # If no good headers detected, try using first row
                            if not headers and len(table_data) > 1:
                                first_row = table_data[0]
                                # Check if first row looks like headers (not numbers)
                                if all(
                                    cell
                                    and not str(cell)
                                    .replace(".", "")
                                    .replace(",", "")
                                    .isdigit()
                                    for cell in first_row[:3]
                                    if cell
                                ):
                                    headers = first_row
                                    table_data = table_data[
                                        1:
                                    ]  # Remove header row from data

                            # Build markdown table
                            table_markdown += (
                                f"\n\n<!-- Table {table_idx + 1} on page {i + 1} -->\n"
                            )

                            # Create markdown table
                            if headers:
                                # Build header row
                                header_row = (
                                    "|"
                                    + "|".join(str(h) if h else "" for h in headers)
                                    + "|"
                                )
                                separator_row = (
                                    "|" + "|".join("---" for _ in headers) + "|"
                                )
                                table_markdown += (
                                    header_row + "\n" + separator_row + "\n"
                                )

                                # Build data rows
                                for row in table_data:
                                    # Ensure row has same number of columns as headers
                                    while len(row) < len(headers):
                                        row.append("")
                                    row_str = (
                                        "|"
                                        + "|".join(
                                            str(cell) if cell else ""
                                            for cell in row[: len(headers)]
                                        )
                                        + "|"
                                    )
                                    table_markdown += row_str + "\n"
                            else:
                                # Fall back to default markdown if no headers
                                table_md = table.to_markdown()
                                if table_md:
                                    table_markdown += table_md
                                else:
                                    # Last resort: create simple table from raw data
                                    for row in table_data:
                                        row_str = (
                                            "|"
                                            + "|".join(
                                                str(cell) if cell else ""
                                                for cell in row
                                            )
                                            + "|"
                                        )
                                        table_markdown += row_str + "\n"

                            table_markdown += "<!-- End of table -->\n"
                    except Exception as e:
                        print(
                            f"Warning: Could not convert table {table_idx + 1} on page {i + 1}: {e}"
                        )
                        table_errors = table_stats["table_errors"]
                        if isinstance(table_errors, int):
                            table_stats["table_errors"] = table_errors + 1
        except Exception as e:
            print(f"Warning: Could not detect tables on page {i + 1}: {e}")
            # Continue with regular text extraction even if table detection fails

        # Combine text and tables
        combined_text = text
        if table_markdown:
            # Insert tables at the end of the page text
            combined_text = text + "\n" + table_markdown if text else table_markdown

        if len(combined_text) > MIN_CHARS_FOR_VALID_PAGE:
            page_texts[i] = combined_text
        else:
            scanned_pages.append(i)
    doc.close()

    if scanned_pages:
        # print(f"OCR needed for pages: {scanned_pages}")
        with tempfile.TemporaryDirectory() as tempdir:
            for i in scanned_pages:
                images = convert_from_path(
                    pdf_path, first_page=i + 1, last_page=i + 1, output_folder=tempdir
                )
                if images:
                    ocr_raw = pytesseract.image_to_string(images[0])
                    ocr_cleaned = clean_ocr_text(
                        ocr_raw,
                        SUPPORTED_LANGUAGES,
                        spell_checkers,
                        SYMBOL_FILTER_THRESHOLD,
                        SPELL_CHECK_THRESHOLD,
                    )
                    page_texts[i] = ocr_cleaned

    # First pass: basic cleaning on all pages
    cleaned_page_texts = []
    for text in page_texts:
        if text.strip():
            cleaned_page_texts.append(clean_text(text))
        else:
            cleaned_page_texts.append("")

    # Identify headers and footers across all pages
    headers, footers = identify_headers_footers(cleaned_page_texts)

    # Second pass: remove structural noise and create page-aware text list
    page_aware_text = []
    for i, text in enumerate(cleaned_page_texts):
        if text.strip():  # Only include pages with actual content
            # Remove identified headers/footers
            if headers or footers:
                text = remove_structural_noise(text, headers, footers)

            # Only add if there's still content after cleaning
            if text.strip():
                page_aware_text.append((text, i + 1))  # 1-based page numbers

    # Prepare metadata
    metadata = {
        "total_pages": total_pages,
        "ocr_pages": len(scanned_pages),
        "native_pages": total_pages - len(scanned_pages),
        "pages_with_content": len(page_aware_text),
        "ocr_page_numbers": [
            p + 1 for p in scanned_pages
        ],  # 1-based for human readability
        "headers_detected": len(headers),
        "footers_detected": len(footers),
        "extraction_method_ratio": {
            "ocr_percentage": (
                (len(scanned_pages) / total_pages * 100) if total_pages > 0 else 0
            ),
            "native_percentage": (
                ((total_pages - len(scanned_pages)) / total_pages * 100)
                if total_pages > 0
                else 0
            ),
        },
        "table_extraction": table_stats,
    }

    return page_aware_text, metadata


def extract_text_legacy(pdf_path: str) -> tuple[str, dict]:
    """
    Legacy function that returns combined text for backward compatibility.

    This function maintains the old interface while using the new page-aware parser.
    """
    page_aware_text, metadata = extract_text_with_ocr_fallback(pdf_path)

    # Combine all page texts into a single string
    combined_text = "\n\n".join(text for text, _ in page_aware_text)

    return combined_text, metadata
