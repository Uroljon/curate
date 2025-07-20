import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
import tempfile
import re
from langdetect import detect
from spellchecker import SpellChecker

spell_de = SpellChecker(language="de")
spell_en = SpellChecker(language="en")


def clean_ocr_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip lines that are mostly symbols or garbage
        if len(re.findall(r"[A-Za-z]", line)) < len(line) * 0.3:
            continue

        try:
            lang = detect(line)
            if lang not in ("en", "de"):
                continue
        except:
            continue

        # Filter out lines with mostly misspellings
        words = re.findall(r"\b\w+\b", line)
        if words:
            misspelled = (
                spell_de.unknown(words) if lang == "de" else spell_en.unknown(words)
            )
            if len(misspelled) > len(words) * 0.6:
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
        if len(text) > 10:
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
