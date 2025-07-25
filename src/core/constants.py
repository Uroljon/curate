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

# Common German action field prefixes
ACTION_FIELD_PATTERNS = [
    r"im Handlungsfeld[:\s]+([^\.]+)",
    r"Handlungsfeld[:\s]+([^\.]+)",
    r"für das Handlungsfeld[:\s]+([^\.]+)",
    r"zum Handlungsfeld[:\s]+([^\.]+)",
]

# English terms that should not appear in German extraction
ENGLISH_FILTER_TERMS = [
    "Development",
    "Enhancement",
    "Support",
    "Promotion",
    "Implementation",
    "Management",
    "Strategy",
    "Initiative",
]

# Patterns for detecting quantitative measures that should be indicators
QUANTITATIVE_PATTERNS = [
    r"\d+\s*(?:km|m²|€|Ladepunkte|Standorte|Wohneinheiten|Hektar|ha|MW|kW)",  # Numbers with units
    r"\d+\s*%",  # Percentages
    r"(?:bis|ab|seit)\s+\d{4}",  # Year references
    r"\d+\s+\w+(?:,\s*\d+\s+\w+)+",  # Lists of numbered items
    r"(?:Verdopplung|Halbierung|Steigerung um|Reduktion um)",  # Comparative terms
    r"\d+(?:\.\d+)?",  # Any number (as fallback)
]

# File type constants
SUPPORTED_DOCUMENT_TYPES = [".pdf"]
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".gif", ".webp"]

# Text processing thresholds
MIN_LINE_LENGTH = 3  # Minimum characters for a valid line
MAX_HEADING_LENGTH = 100  # Maximum length for heading detection
TITLE_CASE_RATIO = 0.7  # Ratio of capitalized words for title case detection

# OCR quality thresholds
MIN_OCR_CONFIDENCE = 60  # Minimum confidence score for OCR
MAX_SYMBOL_RATIO = 0.3  # Maximum ratio of symbols in OCR text
MIN_LETTER_RATIO = 0.7  # Minimum ratio of letters for valid text

# Prompt separators and markers
PROMPT_SEPARATOR = "\n\n"
JSON_CODE_BLOCK_START = "```json"
JSON_CODE_BLOCK_END = "```"

# Response format instructions (German)
GERMAN_RESPONSE_INSTRUCTIONS = """WICHTIG: Indikatoren sind IMMER quantitative Angaben:
- Prozentangaben: "55% Reduktion", "um 30% steigern"
- Zeitangaben: "bis 2030", "ab 2025", "jährlich"
- Mengenangaben: "500 Ladepunkte", "18 km", "1000 Wohneinheiten"
- Vergleiche: "Verdopplung", "Halbierung", "30% weniger"
- Aufzählungen mit Zahlen: "24 Frauenzellstraße, 25 Sallern, 26 Stadtamhof"

REGEL: Enthält der Text eine Zahl, ein Datum oder Prozent? → INDIKATOR. Sonst → MAẞNAHME."""

# Section numbering patterns
SECTION_PATTERNS = {
    "numbered": re.compile(r"^\d+(\.\d+)*\.?\s+"),  # 1.1, 2.3.4, etc.
    "roman": re.compile(r"^[IVXLCDM]+\.\s+"),  # Roman numerals
    "letter": re.compile(r"^[a-z]\)\s+", re.IGNORECASE),  # a), b), etc.
}

# Chunk boundary markers
CHUNK_BOUNDARY_MARKERS = {
    "paragraph": "\n\n",
    "sentence": ". ",
    "line": "\n",
}

# Default timeout values (seconds)
DEFAULT_OCR_TIMEOUT = 30
DEFAULT_EXTRACTION_TIMEOUT = 180
DEFAULT_CHUNK_TIMEOUT = 10
