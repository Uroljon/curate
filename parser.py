# parser.py

import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
import tempfile
import os

def extract_text_with_ocr_fallback(pdf_path):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    page_texts = [""] * total_pages

    scanned_pages = []

    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if len(text) > 10:
            page_texts[i] = text
        else:
            scanned_pages.append(i)
    doc.close()

    if scanned_pages:
        print(f"OCR needed for pages: {scanned_pages}")

        with tempfile.TemporaryDirectory() as tempdir:
            # Convert only the scanned pages
            for i in scanned_pages:
                images = convert_from_path(pdf_path, first_page=i + 1, last_page=i + 1, output_folder=tempdir)
                if images:
                    ocr_text = pytesseract.image_to_string(images[0])
                    page_texts[i] = f"[OCR Page {i + 1}]\n" + ocr_text

    return "\n\n".join(page_texts)
