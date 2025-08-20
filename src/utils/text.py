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
    # Normalize special bullet characters (including Û)
    text = re.sub(r"^(\s*)[•*\-⯀Û]\s*", r"\1• ", text, flags=re.MULTILINE)

    # Remove page numbering lines (more comprehensive patterns)
    text = re.sub(
        r"^(Seite|Page|S\.|Blatt)\s*\d+(\s+(von|of|\/)\s*\d+)?\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Remove standalone numbers that are likely page numbers
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)

    # Remove footnote references with URLs
    text = re.sub(r"^\d+\)\s*https?://.*$", "", text, flags=re.MULTILINE)

    # Merge hyphenated words split across lines (German-aware)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Fix split German place names
    text = re.sub(r"(Beratz|Ober|Unter|Bad|Neu|Alt|Klein|Groß)-\s+(\w+)", r"\1\2", text)

    # Preserve numbered lists by adding bullets
    text = re.sub(r"^(\d{1,2})\s+([A-Z])", r"• \1 \2", text, flags=re.MULTILINE)

    # Merge lines that were broken mid-sentence (improved logic)
    # Don't merge if next line starts with uppercase (likely new sentence)
    text = re.sub(r"(?<![.!?])\n(?![A-ZÄÖÜ•*\d-])", " ", text)

    # Normalize whitespace and clean up
    text = re.sub(r"\n{3,}", "\n\n", text)  # Multiple newlines to double
    text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces to single
    text = re.sub(r"^\s+", "", text, flags=re.MULTILINE)  # Leading whitespace

    # Clean up broken sentences at line boundaries
    lines = text.splitlines()
    cleaned_lines: list[str] = []

    for i, line in enumerate(lines):
        line = line.strip()

        # Skip empty lines
        if not line:
            if (
                i > 0 and cleaned_lines and cleaned_lines[-1]
            ):  # Preserve paragraph breaks
                cleaned_lines.append("")
            continue

        # Fix common OCR issues in German text
        line = line.replace("ß", "ß")  # Normalize eszett
        line = line.replace(",,", "„")  # Fix German quotes
        line = line.replace("''", '"')  # Fix closing quotes

        cleaned_lines.append(line)

    # Join and do final cleanup
    text = "\n".join(cleaned_lines)

    # Remove multiple empty lines again (final pass)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


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


def identify_headers_footers(
    page_texts: list[str], frequency_threshold: float = 0.3
) -> tuple[list[str], list[str]]:
    """
    Identify repeating headers and footers across pages.

    This function analyzes text from multiple pages to find lines that appear
    frequently, which are likely headers or footers.

    Args:
        page_texts: List of text content from each page
        frequency_threshold: Minimum fraction of pages a line must appear on
                           to be considered a header/footer (default 0.25)

    Returns:
        Tuple of (headers, footers) where each is a list of patterns to remove
    """
    if not page_texts or len(page_texts) < 3:
        return [], []

    # Count line occurrences across pages
    from collections import Counter

    # Extract first and last few lines from each page
    first_lines = []
    last_lines = []

    for page_text in page_texts:
        if not page_text.strip():
            continue

        lines = page_text.strip().splitlines()
        if not lines:
            continue

        # Get first 3 lines and last 3 lines
        first_lines.extend(lines[:3])
        last_lines.extend(lines[-3:])

    # Normalize lines for comparison (remove page numbers, whitespace)
    def normalize_for_comparison(line: str) -> str:
        # Remove page numbers
        normalized = re.sub(r"\b\d{1,4}\b", "", line)
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        return normalized.strip()

    # Count normalized line frequencies
    header_counts: Counter[str] = Counter()
    footer_counts: Counter[str] = Counter()

    for line in first_lines:
        normalized = normalize_for_comparison(line)
        if len(normalized) > 10:  # Ignore very short lines
            header_counts[normalized] += 1

    for line in last_lines:
        normalized = normalize_for_comparison(line)
        if len(normalized) > 10:
            footer_counts[normalized] += 1

    # Find patterns that appear frequently
    min_occurrences = int(len(page_texts) * frequency_threshold)

    headers = [
        pattern for pattern, count in header_counts.items() if count >= min_occurrences
    ]

    footers = [
        pattern for pattern, count in footer_counts.items() if count >= min_occurrences
    ]

    return headers, footers


def remove_structural_noise(
    text: str, headers: list[str], footers: list[str], min_content_length: int = 100
) -> str:
    """
    Remove identified headers, footers, and other structural noise from text.

    Args:
        text: Text to clean
        headers: List of header patterns to remove
        footers: List of footer patterns to remove
        min_content_length: Minimum length of remaining content to proceed with removal

    Returns:
        Cleaned text with structural noise removed
    """
    # First, check if we would have enough content left after cleaning
    lines = text.splitlines()
    content_lines = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Check if this line would be removed
        normalized_line = re.sub(r"\b\d{1,4}\b", "", line_stripped)
        normalized_line = " ".join(normalized_line.split()).strip()

        # Would this line be kept?
        is_header = any(header in normalized_line for header in headers)
        is_footer = any(footer in normalized_line for footer in footers)
        is_page_num = bool(
            re.match(r"^(Seite\s+)?\d+(\s+(von|of)\s+\d+)?$", line_stripped)
        )

        if not (is_header or is_footer or is_page_num):
            content_lines.append(line_stripped)

    # Check if we have enough content remaining
    remaining_content = " ".join(content_lines)
    if len(remaining_content) < min_content_length:
        # Don't remove headers/footers if it would leave too little content
        return text

    # Proceed with normal cleaning
    cleaned_lines: list[str] = []

    for _i, line in enumerate(lines):
        line_stripped = line.strip()

        # Skip empty lines
        if not line_stripped:
            cleaned_lines.append("")
            continue

        # Check if line matches any header/footer pattern
        normalized_line = re.sub(r"\b\d{1,4}\b", "", line_stripped)
        normalized_line = " ".join(normalized_line.split()).strip()

        # Skip if matches header pattern
        if any(header in normalized_line for header in headers):
            continue

        # Skip if matches footer pattern
        if any(footer in normalized_line for footer in footers):
            continue

        # Remove standalone page numbers
        if re.match(r"^(Seite\s+)?\d+(\s+(von|of)\s+\d+)?$", line_stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)
