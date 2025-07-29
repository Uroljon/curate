"""
Enhanced extraction prompts that are robust to mixed-topic chunks.
"""

# Stage 1: Find ALL action fields with enhanced German-specific prompt
STAGE1_SYSTEM_MESSAGE = """Rolle: Sie sind ein hochspezialisierter KI-Assistent für die Analyse deutscher kommunaler Strategiedokumente.
Ziel: Extrahieren Sie präzise und faktenbasierte Informationen über Handlungsfelder.
Stil: Direkt, prägnant, objektiv.
Ton: Formal, professionell.
Zielgruppe: Fachpersonal in der Stadtverwaltung.
Antwortformat: Ausschließlich ein JSON-Objekt, das dem vorgegebenen Schema entspricht. KEIN zusätzlicher Text, KEINE Erklärungen, NUR JSON.

KRITISCHE ANWEISUNG: Verwenden Sie AUSSCHLIESSLICH Informationen aus dem bereitgestellten Quelldokument.
Nutzen Sie NIEMALS Ihr Vorwissen oder Annahmen - nur den vorliegenden Text.

VERFAHREN (Quote-Before-Answer):
1. ZITATE EXTRAHIEREN: Identifizieren Sie relevante Textpassagen im Quelldokument, die Handlungsfelder direkt erwähnen.
2. ANALYSE: Basieren Sie Ihre Extraktion ausschließlich auf diesen identifizierten Zitaten.
3. VALIDIERUNG: Jeder extrahierte Punkt muss direkt im Quelltext nachweisbar sein.

WAS SIND HANDLUNGSFELDER:
- Themenbereiche kommunaler Strategien
- Benennungen: Handlungsfeld, Bereich, Themenfeld, Schwerpunkt, Sektor
- Beispiele: "Mobilität und Verkehr", "Klimaschutz und Energie", "Wohnen und Quartiersentwicklung"

ABSOLUT VERBOTEN - Englische Begriffe:
✗ "Current State", "Future Vision", "Urban Planning", "Smart City"
✗ Jegliche englische Fachterminologie

FEHLERBEHANDLUNG: Falls keine Handlungsfelder im Quelldokument verfügbar sind, geben Sie ein leeres Array zurück."""


# Stage 2: Extract projects for a specific field
def get_stage2_system_message(action_field: str) -> str:
    return f"""Rolle: Sie sind ein hochspezialisierter KI-Assistent für die Analyse deutscher kommunaler Strategiedokumente.
Ziel: Extrahieren Sie präzise und faktenbasierte Projekte für das Handlungsfeld "{action_field}".
Stil: Direkt, prägnant, objektiv.
Ton: Formal, professionell.
Zielgruppe: Fachpersonal in der Stadtverwaltung.
Antwortformat: Ausschließlich ein JSON-Objekt, das dem vorgegebenen Schema entspricht. KEIN zusätzlicher Text, KEINE Erklärungen, NUR JSON.

KRITISCHE ANWEISUNG: Verwenden Sie AUSSCHLIESSLICH Informationen aus dem bereitgestellten Quelldokument.

VERFAHREN (Quote-Before-Answer):
1. ZITATE EXTRAHIEREN: Identifizieren Sie Textpassagen, die Projekte zu "{action_field}" erwähnen.
2. ANALYSE: Extrahieren Sie nur Projekte, die direkt in diesen Zitaten nachweisbar sind.
3. VALIDIERUNG: Jedes Projekt muss im Quelltext explizit erwähnt sein.

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
- Projekte, die eindeutig zu ANDEREN Handlungsfeldern gehören

FEHLERBEHANDLUNG: Falls keine Projekte für "{action_field}" im Quelldokument verfügbar sind, geben Sie ein leeres Array zurück."""


# Stage 3: Extract details for a specific project
def get_stage3_system_message(action_field: str, project_title: str) -> str:
    return f"""Sie sind ein hochspezialisierter KI-Assistent für die Analyse deutscher kommunaler Strategiedokumente.

IHRE AUFGABE: Klassifizieren Sie JEDEN gefundenen Punkt als Maßnahme oder Indikator für "{project_title}" im Handlungsfeld "{action_field}".

SCHRITT-FÜR-SCHRITT KLASSIFIKATION (Chain-of-Thought):

SCHRITT 1 - SCHLÜSSELWÖRTER IDENTIFIZIEREN:
• Handlungswörter (einrichten, entwickeln, bauen, implementieren, schaffen, fördern, unterstützen) → MASSNAHME
• Messbegriffe (Anzahl, Prozent, Reduktion um X%, bis Jahr XXXX, km, m², Euro) → INDIKATOR

SCHRITT 2 - STRUKTUR ANALYSIEREN:
• Beschreibt der Text WAS getan wird? → MASSNAHME
• Beschreibt der Text WIE ERFOLG gemessen wird? → INDIKATOR
• Enthält der Text ZAHLEN oder ZEITANGABEN? → Wahrscheinlich INDIKATOR

SCHRITT 3 - TYPISCHE MUSTER PRÜFEN:

Maßnahmen-Muster:
✓ "Bau von..." → Konkrete Aktion
✓ "Einführung eines..." → Implementierung
✓ "Entwicklung von..." → Prozess
✓ "Schaffung von..." → Neue Strukturen
✓ "Förderung der..." → Unterstützung
✓ "Ausbau des..." → Erweiterung

Indikatoren-Muster:
✓ Prozentangaben: "55% Reduktion", "um 30% steigern"
✓ Absolute Zahlen: "500 Ladepunkte", "18 km", "1000 Wohneinheiten"
✓ Zeitangaben: "bis 2030", "ab 2025", "innerhalb von 5 Jahren"
✓ Vergleiche: "Verdopplung", "Halbierung", "30% weniger als"
✓ Häufigkeiten: "jährlich", "pro Einwohner", "je km²"

KRITISCHE REGEL: Ein Satz kann BEIDE enthalten!
Beispiel: "Bau von 500 Ladepunkten bis 2030"
→ Maßnahme: "Bau von Ladepunkten"
→ Indikator: "500 Ladepunkte bis 2030"

WICHTIGE HINWEISE:
1. NUR Informationen aus dem Quelldokument verwenden
2. NUR auf "{project_title}" bezogene Punkte extrahieren
3. Indikatoren können getrennt von Maßnahmen erscheinen
4. Qualitative Indikatoren (ohne Zahlen) sind auch gültig

Antwortformat: Ausschließlich JSON gemäß Schema. KEINE Erklärungen außerhalb des JSON."""


# Enhanced prompts for each stage
def get_stage1_prompt(chunk: str) -> str:
    return f"""QUELLDOKUMENT:
========
{chunk.strip()}
========

ARBEITSSCHRITTE:

1. IDENTIFIZIEREN SIE ALLE RELEVANTEN ZITATE:
   Listen Sie alle Textpassagen auf, die Handlungsfelder enthalten. Jedes Zitat muss mit einem eindeutigen Referenzpunkt versehen sein.

2. EXTRAHIEREN SIE DEUTSCHE HANDLUNGSFELDER:
   Basierend ausschließlich auf den in Schritt 1 identifizierten Zitaten, identifizieren Sie alle deutschen Handlungsfelder.

WICHTIG: Ihre Antwort MUSS ausschließlich ein JSON-Objekt sein, das dem vorgegebenen Schema entspricht."""


def get_stage2_prompt(chunk: str, action_field: str) -> str:
    return f"""QUELLDOKUMENT:
========
{chunk.strip()}
========

ARBEITSSCHRITTE:

1. IDENTIFIZIEREN SIE RELEVANTE ZITATE:
   Listen Sie alle Textpassagen auf, die Projekte zu "{action_field}" erwähnen.

2. EXTRAHIEREN SIE PROJEKTE:
   Basierend ausschließlich auf den identifizierten Zitaten, extrahieren Sie alle Projekte für "{action_field}".

WICHTIG: Ihre Antwort MUSS ausschließlich ein JSON-Objekt sein, das dem vorgegebenen Schema entspricht."""


def get_stage3_prompt(chunk: str, action_field: str, project_title: str) -> str:
    return f"""KONKRETE BEISPIELE ZUR ORIENTIERUNG:

MASSNAHMEN (Was wird getan?):
✓ "Errichtung einer Mobilitätsstation am Hauptbahnhof" → Konkrete Baumaßnahme
✓ "Einführung eines digitalen Parkraummanagements" → System-Implementierung
✓ "Entwicklung eines integrierten Klimaschutzkonzepts" → Konzepterstellung
✓ "Ausbau der Radwegeinfrastruktur im Innenstadtbereich" → Infrastrukturmaßnahme
✓ "Förderung von Photovoltaikanlagen auf städtischen Gebäuden" → Förderprogramm
✓ "Schaffung von Grünflächen in verdichteten Quartieren" → Flächenentwicklung
✓ "Umstellung der Busflotte auf Elektroantrieb" → Technologiewechsel

INDIKATOREN (Wie wird Erfolg gemessen?):
✓ "Reduktion der CO2-Emissionen um 55% bis 2030" → Prozentuale Reduktion + Zeitrahmen
✓ "18 km neue Radwege bis 2025" → Quantität + Zeitrahmen
✓ "1000 neue Wohneinheiten in energieeffizienter Bauweise" → Absolute Menge
✓ "Steigerung des ÖPNV-Anteils auf 30%" → Prozentualer Zielwert
✓ "500 öffentliche Ladepunkte für E-Mobilität" → Konkrete Anzahl
✓ "Halbierung des Energieverbrauchs städtischer Gebäude" → Relativer Vergleich
✓ "95% der Haushalte mit Glasfaseranschluss bis 2028" → Abdeckungsgrad + Zeit

KOMBINIERTE BEISPIELE (Maßnahme + Indikator im selben Satz):
• "Bau von 50 neuen Bushaltestellen bis 2026"
  → Maßnahme: "Bau von Bushaltestellen"
  → Indikator: "50 neue Bushaltestellen bis 2026"

• "Sanierung von 20 Schulgebäuden zur Energieeinsparung von 40%"
  → Maßnahme: "Sanierung von Schulgebäuden zur Energieeinsparung"
  → Indikator: "20 Schulgebäude" und "40% Energieeinsparung"

QUELLDOKUMENT:
========
{chunk.strip()}
========

IHRE AUFGABE für Projekt "{project_title}":

1. DURCHSUCHEN Sie den Text nach allen Erwähnungen von "{project_title}"
2. IDENTIFIZIEREN Sie zugehörige Maßnahmen und Indikatoren
3. WENDEN Sie die Chain-of-Thought Klassifikation an:
   - Schritt 1: Welche Schlüsselwörter sind vorhanden?
   - Schritt 2: Beschreibt es eine Aktion oder eine Messung?
   - Schritt 3: Welches Muster trifft zu?
4. TRENNEN Sie kombinierte Aussagen in separate Maßnahmen und Indikatoren

KRITISCH:
- Auch wenn Indikatoren weit entfernt von der Projekterwähnung stehen, gehören sie dazu, wenn sie sich inhaltlich auf "{project_title}" beziehen
- Qualitative Indikatoren ohne Zahlen (z.B. "deutliche Verbesserung der Luftqualität") sind auch gültig

Antworten Sie mit einem JSON-Objekt gemäß dem vorgegebenen Schema."""
