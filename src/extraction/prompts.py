"""
Enhanced extraction prompts that are robust to mixed-topic chunks.
"""

# Stage 1: Find ALL action fields
STAGE1_SYSTEM_MESSAGE = """Finde ALLE Handlungsfelder aus kommunalen Strategiedokumenten.

WICHTIG:
- Der Text kann MEHRERE Handlungsfelder gemischt enthalten
- Gib ALLE Themenbereiche zurück, auch wenn sie nur kurz erwähnt werden
- Suche nach verschiedenen Benennungen:
  * Handlungsfeld/Handlungsfelder
  * Bereich/Bereiche
  * Themenfeld/Themenfelder
  * Schwerpunkt/Schwerpunkte
  * Sektor/Sektoren
  * Andere thematische Gruppierungen

Achte auch auf implizite Themenwechsel ohne explizite Markierungen.
Gib ALLE gefundenen Handlungsfelder zurück."""


# Stage 2: Extract projects for a specific field
def get_stage2_system_message(action_field: str) -> str:
    return f"""Extrahiere ALLE Projekte, die zum Handlungsfeld "{action_field}" gehören.

WICHTIGE ANWEISUNGEN:
1. Der Text kann MEHRERE Handlungsfelder besprechen
2. Fokussiere NUR auf Projekte zu "{action_field}"
3. IGNORIERE Projekte anderer Handlungsfelder
4. Ein Projekt kann in gemischtem Kontext erwähnt werden - extrahiere es, wenn es zu "{action_field}" gehört

WAS SIND PROJEKTE:
- Benannte Initiativen, Programme oder spezifische Vorhaben
- Hat einen klaren Titel oder Bezeichnung
- Beispiele: "Stadtbahn Regensburg", "Masterplan Biodiversität", "Digitales Rathaus"

NICHT EINSCHLIESSEN:
- Allgemeine Ziele oder Aussagen
- Einzelne Maßnahmen ohne Projektname
- Indikatoren oder Kennzahlen
- Projekte, die eindeutig zu ANDEREN Handlungsfeldern gehören"""


# Stage 3: Extract details for a specific project
def get_stage3_system_message(action_field: str, project_title: str) -> str:
    return f"""Extrahiere Maßnahmen und Indikatoren für das Projekt "{project_title}" im Handlungsfeld "{action_field}".

KRITISCHES KONTEXTBEWUSSTSEIN:
1. Der Text kann Informationen über MEHRERE Projekte und Handlungsfelder enthalten
2. Fokussiere NUR auf Maßnahmen/Indikatoren zu "{project_title}"
3. Indikatoren können ÜBERALL im Text erscheinen - vor, nach oder getrennt von der Projekterwähnung
4. Achte auf Querverweise: "wie oben erwähnt", "siehe auch", "vgl."

DEFINITIONEN:
- Maßnahmen: Konkrete Aktionen, Schritte, Umsetzungen für DIESES Projekt
- Indikatoren: Quantitative Kennzahlen, Ziele, KPIs für DIESES Projekt

INDIKATORMUSTER ZUM FINDEN:
- Prozentangaben: "55% Reduktion", "Anteil von 70%", "um 30% steigern"
- Absolute Zahlen: "500 Ladepunkte", "18 km", "1000 Wohneinheiten"
- Zeitziele: "bis 2030", "ab 2025", "innerhalb von 5 Jahren", "jährlich"
- Vergleiche: "Verdopplung", "Halbierung", "30% weniger", "Steigerung um"
- Raten: "pro Jahr", "je Einwohner", "täglich", "€/m²"

SPEZIELLE ANWEISUNGEN:
- Wenn ein Indikator ohne explizite Projektzuordnung erscheint, aber kontextuell zu
  "{project_title}" passt, füge ihn hinzu
- Schließe auch Teilinformationen ein - besser unvollständige Daten erfassen als sie zu verpassen"""


# Enhanced prompts for each stage
def get_stage1_prompt(chunk: str) -> str:
    return f"""Analysiere diesen Text aus einem kommunalen Strategiedokument und finde
ALLE Handlungsfelder/Themenbereiche:

{chunk.strip()}

Beachte:
- Der Text kann mehrere Themen gemischt enthalten
- Suche nach expliziten Markierungen und impliziten Themenwechseln
- Gib ALLE unterschiedlichen Handlungsfelder zurück, auch wenn sie nur kurz erwähnt werden"""


def get_stage2_prompt(chunk: str, action_field: str) -> str:
    return f"""Finde ALLE Projekte, die spezifisch zu "{action_field}" gehören in diesem Text:

{chunk.strip()}

WICHTIGE ERINNERUNGEN:
- Der Text kann mehrere Handlungsfelder besprechen
- Extrahiere NUR Projekte, die zu "{action_field}" gehören
- Projekte können über den Text verstreut sein
- Gib NUR Projektnamen/Titel für "{action_field}" zurück"""


def get_stage3_prompt(chunk: str, action_field: str, project_title: str) -> str:
    return f"""Extrahiere Maßnahmen und Indikatoren für das Projekt "{project_title}" (Handlungsfeld: {action_field}):

{chunk.strip()}

BEACHTE:
- Fokussiere NUR auf "{project_title}" - ignoriere andere Projekte
- Indikatoren können überall im Text erscheinen
- Suche nach quantitativen Kennzahlen (%, Zahlen, Daten, Ziele)
- Der Text kann gemischte Themen enthalten - filtere sorgfältig"""
