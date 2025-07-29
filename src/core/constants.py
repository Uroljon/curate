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

# Advanced German municipal patterns for enhanced classification
ADVANCED_MEASURE_PATTERNS = {
    "initiative_verbs": [
        "initiieren",
        "anstoßen",
        "etablieren",
        "implementieren",
        "realisieren",
        "umsetzen",
        "durchführen",
        "entwickeln",
        "einrichten",
        "schaffen",
        "aufbauen",
        "errichten",
        "bereitstellen",
        "gewährleisten",
        "sicherstellen",
        "ermöglichen",
    ],
    "administrative_actions": [
        "bereitstellung von",
        "gewährleistung von",
        "sicherstellung von",
        "ermöglichung von",
        "schaffung von",
        "einrichtung von",
        "aufbau von",
        "errichtung von",
        "durchführung von",
        "realisierung von",
        "umsetzung von",
        "implementierung von",
    ],
    "planning_instruments": [
        "konzept entwickeln",
        "strategie erarbeiten",
        "leitfaden erstellen",
        "rahmen schaffen",
        "grundlage legen",
        "masterplan erstellen",
        "handlungskonzept",
        "aktionsplan",
        "arbeitsgruppe bilden",
    ],
}

ADVANCED_INDICATOR_PATTERNS = {
    "quantitative_targets": [
        r"\d+(?:[,\.]\d+)?\s*%",  # Percentages: 30%, 15,5%
        r"\d+(?:[,\.]\d+)?\s*(?:Euro|EUR|€)",  # Currency: 2,5 Mio Euro
        r"\d+(?:[,\.]\d+)?\s*(?:km|m²|ha|MW|GW)",  # Measurements
        r"bis\s+(?:20)?\d{2,4}",  # Time targets: bis 2030
        r"ab\s+(?:20)?\d{2,4}",  # From year: ab 2025
        r"um\s+\d+(?:[,\.]\d+)?\s*%",  # Percentage changes: um 30%
        r"\d+(?:[,\.]\d+)?\s*(?:von|aus)\s+\d+",  # Ratios: 5 von 10
        r"\d+(?:[,\.]\d+)?\s*(?:pro|je|per)\s+\w+",  # Rates: 15 pro Jahr
    ],
    "measurement_verbs": [
        "erreichen",
        "reduzieren",
        "steigern",
        "erhöhen",
        "senken",
        "verbessern um",
        "verringern um",
        "zunehmen um",
        "ansteigen auf",
        "sinken auf",
        "verdoppeln",
        "halbieren",
        "vervielfachen",
    ],
    "success_metrics": [
        "zielwert",
        "kennzahl",
        "indikator",
        "messgröße",
        "erfolgsparameter",
        "leistungskennzahl",
        "benchmark",
        "richtwert",
        "sollwert",
        "grenzwert",
    ],
}

# German municipal confidence indicators
CONFIDENCE_BOOSTERS = {
    "very_high": [
        "eindeutig",
        "klar definiert",
        "messbar",
        "quantifiziert",
        "konkret",
        "spezifisch",
        "explizit",
        "präzise",
    ],
    "high": [
        "typischerweise",
        "in der regel",
        "üblicherweise",
        "gewöhnlich",
        "standardmäßig",
        "regulär",
    ],
    "medium": [
        "wahrscheinlich",
        "vermutlich",
        "tendenziell",
        "voraussichtlich",
        "mutmaßlich",
        "möglicherweise",
    ],
    "low": [
        "eventuell",
        "unter umständen",
        "gegebenenfalls",
        "potentiell",
        "denkbar",
        "fraglich",
    ],
}

# German administrative language patterns
ADMINISTRATIVE_LANGUAGE_PATTERNS = {
    "formal_structures": [
        r"gemäß\s+\S+",  # "gemäß §12"
        r"im\s+rahmen\s+(?:der|des)\s+\S+",  # "im Rahmen des Projekts"
        r"auf\s+grundlage\s+(?:der|des)\s+\S+",  # "auf Grundlage des Konzepts"
        r"unter\s+berücksichtigung\s+(?:der|des)\s+\S+",
        r"in\s+abstimmung\s+mit\s+\S+",
        r"in\s+zusammenarbeit\s+mit\s+\S+",
    ],
    "decision_language": [
        "beschlossen",
        "festgelegt",
        "bestimmt",
        "angeordnet",
        "verfügt",
        "entschieden",
        "beschließen",
        "festlegen",
    ],
    "implementation_language": [
        "zur umsetzung",
        "zur realisierung",
        "zur durchführung",
        "zur implementierung",
        "zur einführung",
        "zur etablierung",
    ],
}

# Enhanced action field recognition patterns
ENHANCED_ACTION_FIELD_PATTERNS = {
    "standard_fields": {
        "mobilität": [
            "verkehr",
            "mobilität",
            "öpnv",
            "radweg",
            "fußgänger",
            "elektromobilität",
        ],
        "klimaschutz": [
            "klima",
            "energie",
            "emission",
            "erneuerbar",
            "co2",
            "nachhaltigkeit",
        ],
        "stadtentwicklung": [
            "stadtplanung",
            "quartier",
            "wohnen",
            "bauland",
            "städtebau",
        ],
        "digitalisierung": [
            "digital",
            "smart city",
            "e-government",
            "online",
            "vernetzung",
        ],
        "bildung": [
            "schule",
            "bildung",
            "kita",
            "kindergarten",
            "ausbildung",
            "weiterbildung",
        ],
        "gesundheit": ["gesundheit", "medizin", "pflege", "prävention", "therapie"],
        "kultur": [
            "kultur",
            "kunst",
            "museum",
            "theater",
            "bibliothek",
            "veranstaltung",
        ],
        "wirtschaft": ["wirtschaft", "gewerbe", "handel", "tourismus", "arbeitsplätze"],
    },
    "compound_recognition": [
        r'(?:handlungsfeld|bereich|themenfeld)\s+["\u201E\u201C]([^"\u201E\u201C]+)["\u201E\u201C]',
        r'(?:im|zum|für)\s+(?:handlungsfeld|bereich)\s+["\u201E\u201C]?([^"\u201E\u201C\n\.]+)["\u201E\u201C]?',
        r'schwerpunkt\s+["\u201E\u201C]?([^"\u201E\u201C\n\.]+)["\u201E\u201C]?',
    ],
}

# Text quality indicators for German municipal documents
TEXT_QUALITY_INDICATORS = {
    "high_quality": [
        r"\d+(?:\.\d+)*\.\s+[A-ZÄÖÜ]",  # Numbered sections: "2.1. Ziele"
        r"handlungsfeld\s+\d+",  # "Handlungsfeld 1"
        r"maßnahme\s+\d+(?:\.\d+)?",  # "Maßnahme 3.2"
        r'(?:projekt|initiative)\s+["\u201E\u201C]([^"\u201E\u201C]+)["\u201E\u201C]',  # Quoted project names
    ],
    "administrative_terms": [
        "stadtverwaltung",
        "gemeinderat",
        "bürgermeister",
        "amt für",
        "fachdienst",
        "stadtwerke",
        "öffentlicher träger",
        "kommunal",
    ],
    "document_structure": [
        "inhaltsverzeichnis",
        "zusammenfassung",
        "executive summary",
        "zielsetzung",
        "ausgangssituation",
        "handlungsempfehlung",
    ],
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
