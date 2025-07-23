import re
import tempfile

import fitz  # PyMuPDF
import pytesseract
from langdetect import detect
from pdf2image import convert_from_path
from spellchecker import SpellChecker

from src.core import (
    MIN_CHARS_FOR_VALID_PAGE,
    SPELL_CHECK_THRESHOLD,
    SYMBOL_FILTER_THRESHOLD,
    SUPPORTED_LANGUAGES,
    SPELL_CHECK_LANGUAGES,
)

# Initialize spell checkers based on config
spell_checkers = {
    lang: SpellChecker(language=spell_lang) 
    for lang, spell_lang in SPELL_CHECK_LANGUAGES.items()
}


def clean_ocr_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip lines that are mostly symbols or garbage
        if len(re.findall(r"[A-Za-z]", line)) < len(line) * SYMBOL_FILTER_THRESHOLD:
            continue

        try:
            lang = detect(line)
            if lang not in SUPPORTED_LANGUAGES:
                continue
        except:
            continue

        # Filter out lines with mostly misspellings
        words = re.findall(r"\b\w+\b", line)
        if words and lang in spell_checkers:
            misspelled = spell_checkers[lang].unknown(words)
            if len(misspelled) > len(words) * SPELL_CHECK_THRESHOLD:
                continue

        cleaned.append(line)

    return "\n".join(cleaned)


def clean_text(text: str) -> str:
    # Remove page numbering lines
    text = re.sub(
        r"^(Seite|Page)\s+\d+(\s+(von|of)\s+\d+)?\s*$", "", text, flags=re.MULTILINE
    )

    # Merge hyphenated words split across lines
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Merge lines that were broken mid-sentence
    text = re.sub(r"(?<!\n)\n(?![\n0-9•*-])", " ", text)

    # Normalize bullets and whitespace
    text = re.sub(r"^[•*-]\s*", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Remove residual empty lines
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(lines).strip()


def extract_text_with_ocr_fallback(pdf_path: str) -> str:
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
                    ocr_cleaned = clean_ocr_text(ocr_raw)
                    # page_texts[i] = f"[OCR Page {i + 1}]\n{ocr_cleaned}"  # You can remove this tag later
                    page_texts[i] = ocr_cleaned

    combined_text = "\n\n".join(page_texts)
    return clean_text(combined_text)
