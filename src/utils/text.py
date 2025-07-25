"""
Text processing utilities for CURATE.

This module consolidates common text processing functions
used throughout the codebase.
"""

import re
from typing import Any

from langdetect import LangDetectException, detect
from spellchecker import SpellChecker

from src.core.constants import (
    GERMAN_SECTION_KEYWORDS,
    MAX_HEADING_LENGTH,
    MIN_LINE_LENGTH,
    SECTION_PATTERNS,
    TITLE_CASE_RATIO,
)
from src.core.errors import LanguageDetectionError


def clean_ocr_text(
    text: str,
    supported_languages: list[str],
    spell_checkers: dict[str, SpellChecker],
    symbol_filter_threshold: float,
    spell_check_threshold: float,
) -> str:
    """
    Clean and filter OCR text by removing noise and invalid content.

    Args:
        text: Raw OCR text to clean
        supported_languages: List of supported language codes
        spell_checkers: Dictionary mapping language codes to SpellChecker instances
        symbol_filter_threshold: Minimum ratio of letters in a line to keep it
        spell_check_threshold: Maximum ratio of misspelled words to keep a line

    Returns:
        Cleaned text with invalid lines removed
    """
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip lines that are mostly symbols or garbage
        letter_count = len(re.findall(r"[A-Za-z]", line))
        if letter_count < len(line) * symbol_filter_threshold:
            continue

        # Try to detect language
        try:
            lang = detect(line)
            if lang not in supported_languages:
                continue
        except LangDetectException:
            # Skip lines where language cannot be detected
            continue

        # Filter out lines with mostly misspellings
        words = re.findall(r"\b\w+\b", line)
        if words and lang in spell_checkers:
            misspelled = spell_checkers[lang].unknown(words)
            if len(misspelled) > len(words) * spell_check_threshold:
                continue

        cleaned.append(line)

    return "\n".join(cleaned)


def clean_text(text: str) -> str:
    """
    Clean extracted text by normalizing formatting and structure.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned and normalized text
    """
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


def is_heading(line: str) -> bool:
    """
    Detect if a line is likely a heading.

    Criteria:
    - Numbered sections (1.1, 2.3.4, etc.)
    - Title case lines
    - All uppercase lines
    - Lines starting with specific keywords
    - Lines ending with a colon

    Args:
        line: Text line to check

    Returns:
        True if line appears to be a heading
    """
    line = line.strip()
    if not line or len(line) < MIN_LINE_LENGTH:
        return False

    # Check against section patterns
    for _pattern_name, pattern in SECTION_PATTERNS.items():
        if pattern.match(line):
            return True

    # German section keywords
    if any(line.startswith(keyword) for keyword in GERMAN_SECTION_KEYWORDS):
        return True

    # Check if it's all uppercase (but not too short)
    if len(line) > MIN_LINE_LENGTH and line.isupper():
        return True

    # Check if it's title case (most words capitalized)
    words = line.split()
    if len(words) >= 2:
        capitalized = sum(1 for word in words if word and word[0].isupper())
        if capitalized / len(words) > TITLE_CASE_RATIO:
            return True

    # Check if line ends with colon (often indicates a heading)
    if line.endswith(":") and len(line) < MAX_HEADING_LENGTH:
        return True

    return False


def normalize_german_text(text: str) -> str:
    """
    Normalize German text for consistent processing.

    Args:
        text: German text to normalize

    Returns:
        Normalized text
    """
    # Normalize German quotes
    text = re.sub(r"[„" "]", '"', text)
    text = re.sub(r"[‚'']", "'", text)  # noqa: RUF001

    # Normalize German dashes
    text = re.sub(r"[–—]", "-", text)  # noqa: RUF001

    # Fix common OCR errors in German
    replacements = {
        "ß": "ß",  # Ensure correct eszett
        "ae": "ä",  # Common OCR substitution
        "oe": "ö",
        "ue": "ü",
        "Ae": "Ä",
        "Oe": "Ö",
        "Ue": "Ü",
    }

    # Apply replacements only in appropriate contexts
    for old, new in replacements.items():
        # Only replace if surrounded by word boundaries to avoid false positives
        text = re.sub(rf"\b{old}\b", new, text, flags=re.IGNORECASE)

    return text


def extract_numbers_from_text(text: str) -> list[dict[str, Any]]:
    """
    Extract all numbers and quantitative information from text.

    Args:
        text: Text to extract numbers from

    Returns:
        List of dictionaries containing number, unit, and context
    """
    numbers = []

    # Pattern for numbers with units
    pattern = r"(\d+(?:[,\.]\d+)?)\s*([A-Za-zäöüÄÖÜß€%]+(?:\s+\w+)?)"

    for match in re.finditer(pattern, text):
        number = match.group(1).replace(",", ".")
        unit = match.group(2).strip()

        # Get context (surrounding text)
        start = max(0, match.start() - 30)
        end = min(len(text), match.end() + 30)
        context = text[start:end].strip()

        numbers.append(
            {
                "value": float(number),
                "raw": match.group(0),
                "unit": unit,
                "context": context,
                "position": match.start(),
            }
        )

    return numbers


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences, handling German abbreviations.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    # Common German abbreviations that don't end sentences
    abbreviations = {
        "bzw",
        "ca",
        "d.h",
        "evtl",
        "ggf",
        "i.d.R",
        "inkl",
        "max",
        "min",
        "o.ä",
        "s.o",
        "u.a",
        "u.ä",
        "usw",
        "vgl",
        "z.B",
        "z.T",
        "zzgl",
        "Dr",
        "Prof",
        "Nr",
        "Str",
        "Mio",
        "Mrd",
        "Tsd",
    }

    # Replace abbreviations temporarily
    temp_text = text
    replacements = {}
    for i, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR{i}__"
        temp_text = re.sub(rf"\b{abbr}\.", placeholder, temp_text, flags=re.IGNORECASE)
        replacements[placeholder] = f"{abbr}."

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ])", temp_text)

    # Restore abbreviations
    for i, sentence in enumerate(sentences):
        for placeholder, original in replacements.items():
            sentences[i] = sentence.replace(placeholder, original)

    return [s.strip() for s in sentences if s.strip()]


def detect_language_safe(text: str, default: str = "de") -> str:
    """
    Safely detect language with fallback.

    Args:
        text: Text to analyze
        default: Default language if detection fails

    Returns:
        Detected language code
    """
    try:
        # Take a sample for faster detection
        sample = text[:1000] if len(text) > 1000 else text
        result = detect(sample)
        return str(result)  # Ensure string type
    except (LangDetectException, Exception):
        return default


def remove_duplicate_lines(text: str, threshold: float = 0.9) -> str:
    """
    Remove duplicate or near-duplicate lines.

    Args:
        text: Text to process
        threshold: Similarity threshold (0-1) for considering lines as duplicates

    Returns:
        Text with duplicates removed
    """
    lines = text.splitlines()
    unique_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            unique_lines.append("")
            continue

        # Check if this line is similar to any existing line
        is_duplicate = False
        for existing in unique_lines[-10:]:  # Only check recent lines
            if not existing:
                continue

            # Simple similarity check based on common characters
            common = len(set(line) & set(existing))
            total = len(set(line) | set(existing))
            if total > 0 and common / total > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_lines.append(line)

    return "\n".join(unique_lines)
