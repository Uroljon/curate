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

    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if len(text) > MIN_CHARS_FOR_VALID_PAGE:
            page_texts[i] = text
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
