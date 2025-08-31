"""
Constants and patterns used throughout CURATE.

This module consolidates all constant values, patterns, and
language-specific terms used across the codebase.
"""

import re

# German section keywords for heading detection
GERMAN_SECTION_KEYWORDS = [
    "Kapitel",
    "Abschnitt",
    "Teil",
    "Artikel",
    "Paragraph",
    "Handlungsfeld",
    "Maßnahme",
    "Projekt",
    "Ziel",
    "Indikator",
    "Anhang",
    "Anlage",
    "Übersicht",
    "Zusammenfassung",
]


# Text processing thresholds
MIN_LINE_LENGTH = 3  # Minimum characters for a valid line
MAX_HEADING_LENGTH = 100  # Maximum length for heading detection
TITLE_CASE_RATIO = 0.7  # Ratio of capitalized words for title case detection


# Section numbering patterns
SECTION_PATTERNS = {
    "numbered": re.compile(r"^\d+(\.\d+)*\.?\s+"),  # 1.1, 2.3.4, etc.
    "roman": re.compile(r"^[IVXLCDM]+\.\s+"),  # Roman numerals
    "letter": re.compile(r"^[a-z]\)\s+", re.IGNORECASE),  # a), b), etc.
}
