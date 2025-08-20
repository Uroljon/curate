"""
Constants and patterns used throughout CURATE.

This module consolidates all constant values, patterns, and
language-specific terms used across the codebase.
"""

import re

# Indicator patterns for German municipal documents
INDICATOR_PATTERNS = {
    "percentage": r"\d+(?:[,\.]\d+)?\s*%",
    "currency": r"\d+(?:[,\.]\d+)?\s*(?:Millionen|Mio\.?|Tsd\.?|Mrd\.?)?\s*(?:Euro|EUR|€)",
    "measurement": r"\d+(?:[,\.]\d+)?\s*(?:km|m²|qm|MW|GW|kW|ha|t|kg|m)",
    "time_target": r"(?:bis|ab|von|nach|vor|zum|zur)\s+(?:20\d{2}|Jahr\s+20\d{2})",
    "quantity": r"\d+(?:[,\.]\d+)?\s*(?:neue|mehr|weniger|zusätzliche)",
    "rate": r"\d+(?:[,\.]\d+)?\s*%?\s*(?:pro|je|per)\s+(?:Jahr|Monat|Tag|Einwohner|km)",
    "compound": r"\d+(?:[,\.]\d+)?\s*(?:von|aus)\s+\d+",  # "5 von 10", "3 aus 20"
}

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
