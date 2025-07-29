#!/usr/bin/env python3
"""
Test script to demonstrate Chain-of-Thought classification for Maßnahmen vs Indikatoren.

This script tests the enhanced classification with step-by-step reasoning
and confidence scoring.
"""

import sys

sys.path.append(".")

from src.core import (
    MODEL_NAME,
    MODEL_TEMPERATURE,
    ProjectDetails,
    ProjectDetailsEnhanced,
    query_ollama_structured,
)
from src.extraction.prompts import get_stage3_prompt, get_stage3_system_message


def test_classification_examples():
    """Test classification on specific examples."""

    # Test examples with known classifications
    test_cases = [
        {
            "text": "Das Projekt umfasst den Bau von 500 neuen Ladepunkten für E-Mobilität bis 2030 im gesamten Stadtgebiet.",
            "expected_measures": ["Bau von Ladepunkten für E-Mobilität"],
            "expected_indicators": ["500 neue Ladepunkte bis 2030"],
        },
        {
            "text": (
                "Zur Förderung der nachhaltigen Mobilität wird die Radwegeinfrastruktur "
                "ausgebaut. Ziel ist es, das Radwegenetz um 18 km zu erweitern und den "
                "Radverkehrsanteil auf 25% zu steigern."
            ),
            "expected_measures": [
                "Ausbau der Radwegeinfrastruktur",
                "Förderung der nachhaltigen Mobilität",
            ],
            "expected_indicators": [
                "18 km Erweiterung des Radwegenetzes",
                "Radverkehrsanteil auf 25% steigern",
            ],
        },
        {
            "text": (
                "Die Stadt plant die Einführung eines digitalen Parkraummanagements. "
                "Dadurch soll die Parkplatzsuche um 30% verkürzt und die CO2-Emissionen "
                "im Verkehrsbereich reduziert werden."
            ),
            "expected_measures": ["Einführung eines digitalen Parkraummanagements"],
            "expected_indicators": [
                "Parkplatzsuche um 30% verkürzen",
                "Reduktion der CO2-Emissionen",
            ],
        },
        {
            "text": (
                "Im Rahmen der Klimaschutzinitiative werden städtische Gebäude energetisch "
                "saniert. Die Maßnahme umfasst 20 Schulgebäude und 15 Verwaltungsgebäude "
                "mit dem Ziel einer Energieeinsparung von 40%."
            ),
            "expected_measures": [
                "Energetische Sanierung städtischer Gebäude",
                "Sanierung von Schulgebäuden",
                "Sanierung von Verwaltungsgebäuden",
            ],
            "expected_indicators": [
                "20 Schulgebäude",
                "15 Verwaltungsgebäude",
                "40% Energieeinsparung",
            ],
        },
    ]

    # Test parameters
    action_field = "Testfeld"
    project_title = "Testprojekt"

    print("=" * 80)
    print("TESTING CHAIN-OF-THOUGHT CLASSIFICATION")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Temperature: {MODEL_TEMPERATURE}")
    print("=" * 80)

    for i, test_case in enumerate(test_cases):
        print(f"\n\nTEST CASE {i+1}")
        print("-" * 60)
        print(f"Text: {test_case['text']}")
        print(f"\nExpected Measures: {test_case['expected_measures']}")
        print(f"Expected Indicators: {test_case['expected_indicators']}")

        # Create system message and prompt
        system_message = get_stage3_system_message(action_field, project_title)
        prompt = get_stage3_prompt(test_case["text"], action_field, project_title)

        # Test with enhanced schema (Chain-of-Thought)
        print("\n🧠 Testing with Chain-of-Thought...")
        result_cot = query_ollama_structured(
            prompt=prompt,
            response_model=ProjectDetailsEnhanced,
            system_message=system_message,
            temperature=MODEL_TEMPERATURE,
        )

        if result_cot:
            print("\n✅ Chain-of-Thought Results:")
            print(f"   Measures: {result_cot.measures}")
            print(f"   Indicators: {result_cot.indicators}")

            # Print confidence scores
            if result_cot.confidence_scores:
                print("\n   Confidence Scores:")
                for item, score in result_cot.confidence_scores.items():
                    print(f"      - {item}: {score:.2f}")

            # Print reasoning
            if result_cot.reasoning:
                print("\n   Reasoning:")
                for item, reasoning in result_cot.reasoning.items():
                    print(f"      - {item}: {reasoning}")

        # Also test with original schema for comparison
        print("\n📊 Testing with Original Schema...")
        result_orig = query_ollama_structured(
            prompt=prompt,
            response_model=ProjectDetails,
            system_message=system_message,
            temperature=MODEL_TEMPERATURE,
        )

        if result_orig:
            print("\n✅ Original Results:")
            print(f"   Measures: {result_orig.measures}")
            print(f"   Indicators: {result_orig.indicators}")


def test_complex_mixed_text():
    """Test on a complex text with mixed content."""

    complex_text = """
    Die Stadt Regensburg setzt im Handlungsfeld Mobilität verschiedene Projekte um:

    1. Stadtbahn Regensburg: Der Bau einer modernen Stadtbahn verbindet die Stadtteile.
       Geplant sind 3 Linien mit einer Gesamtlänge von 28 km. Die erste Linie soll 2028
       in Betrieb gehen. Erwartet werden 50.000 Fahrgäste täglich.

    2. Mobilitätsstationen: An 12 Standorten werden intermodale Mobilitätsstationen
       errichtet. Diese bieten Umsteigemöglichkeiten zwischen Bus, Fahrrad und Carsharing.
       Bis 2026 sollen alle Stationen fertiggestellt sein.

    3. Förderung E-Mobilität: Die Ladeinfrastruktur wird massiv ausgebaut. Zusätzlich
       zu den bestehenden 120 Ladepunkten werden 380 neue Ladepunkte installiert,
       sodass insgesamt 500 öffentliche Ladepunkte zur Verfügung stehen.

    Weitere Indikatoren für den Erfolg: Modal Split soll sich zugunsten des Umweltverbunds
    auf 70% verschieben. Die CO2-Emissionen im Verkehr sollen um 45% reduziert werden.
    Die Luftqualität soll deutlich verbessert werden, mit einer Reduktion der
    Stickoxidbelastung unter den EU-Grenzwert von 40 μg/m³.
    """

    print("\n\n" + "=" * 80)
    print("TESTING COMPLEX MIXED TEXT")
    print("=" * 80)

    action_field = "Mobilität und Verkehr"
    project_title = "Stadtbahn Regensburg"

    system_message = get_stage3_system_message(action_field, project_title)
    prompt = get_stage3_prompt(complex_text, action_field, project_title)

    print(f"\n🧠 Extracting with Chain-of-Thought for project: {project_title}")

    result = query_ollama_structured(
        prompt=prompt,
        response_model=ProjectDetailsEnhanced,
        system_message=system_message,
        temperature=MODEL_TEMPERATURE,
    )

    if result:
        print("\n✅ Results:")
        print(f"\n   Measures ({len(result.measures)}):")
        for measure in result.measures:
            confidence = result.confidence_scores.get(measure, 0)
            print(f"      - {measure} (confidence: {confidence:.2f})")

        print(f"\n   Indicators ({len(result.indicators)}):")
        for indicator in result.indicators:
            confidence = result.confidence_scores.get(indicator, 0)
            print(f"      - {indicator} (confidence: {confidence:.2f})")

        # Show items filtered out by confidence
        low_conf_items = [
            (item, conf)
            for item, conf in result.confidence_scores.items()
            if conf < 0.8
        ]

        if low_conf_items:
            print("\n   ⚠️ Low confidence items (< 0.8):")
            for item, conf in low_conf_items:
                print(f"      - {item} (confidence: {conf:.2f})")


if __name__ == "__main__":
    print("Starting Chain-of-Thought Classification Test...")
    print("Note: Make sure Ollama is running with qwen3:14b model")
    print()

    # Run tests
    test_classification_examples()
    test_complex_mixed_text()

    print("\n\nTest completed!")
