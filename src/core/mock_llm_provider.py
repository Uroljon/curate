"""
Mock LLM provider for fast integration testing.

This provides predictable responses for testing the complete API pipeline
without making actual LLM API calls.
"""

import json
import os
from typing import TypeVar

from pydantic import BaseModel

from .llm_providers import LLMProvider
from .operations_schema import ExtractionOperations

T = TypeVar("T", bound=BaseModel)


class MockLLMProvider(LLMProvider):
    """Mock LLM provider that returns predictable test responses."""

    def __init__(self, model_name: str = "mock", temperature: float = 0.2):
        super().__init__(model_name, temperature)
        self.call_count = 0

    def query_structured(
        self,
        prompt: str,
        response_model: type[T],
        system_message: str | None = None,
        log_file_path: str | None = None,
        log_context: str | None = None,
        override_num_predict: int | None = None,
    ) -> T | None:
        """Return mock structured response based on prompt content."""
        self.call_count += 1

        # Log the call for debugging
        if log_file_path:
            try:
                with open(log_file_path, "a", encoding="utf-8") as f:
                    log_entry = {
                        "timestamp": "2025-08-21T12:00:00Z",
                        "provider": "mock",
                        "model": self.model_name,
                        "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                        "system_message": system_message[:100] + "..." if system_message and len(system_message) > 100 else system_message,
                        "context": log_context,
                        "call_number": self.call_count
                    }
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception:
                pass  # Ignore logging errors in tests

        # Return different responses based on the response model type
        if response_model == ExtractionOperations:
            return self._generate_mock_operations(prompt)
        else:
            # For other models, try to create an empty instance
            try:
                return response_model()
            except Exception:
                return None

    def _generate_mock_operations(self, prompt: str) -> ExtractionOperations:
        """Generate mock operations based on prompt content."""
        operations = []

        # Detect content patterns and generate appropriate operations
        prompt_lower = prompt.lower()

        # Action Field creation
        if "mobilität" in prompt_lower or "verkehr" in prompt_lower:
            operations.append({
                "operation": "CREATE",
                "entity_type": "action_field",
                "content": {
                    "title": "Mobilität und Verkehr",
                    "description": "Nachhaltige Verkehrslösungen und Mobilitätskonzepte"
                },
                "confidence": 0.9,
                "source_pages": [1],
                "source_quote": "Mobilität und Verkehr"
            })

        if "energie" in prompt_lower or "klimaschutz" in prompt_lower:
            operations.append({
                "operation": "CREATE",
                "entity_type": "action_field",
                "content": {
                    "title": "Energie und Klimaschutz",
                    "description": "Energieeffizienz und erneuerbare Energien"
                },
                "confidence": 0.88,
                "source_pages": [1, 2],
                "source_quote": "Energie und Klimaschutz"
            })

        # Project creation
        if "radweg" in prompt_lower or "fahrrad" in prompt_lower:
            operations.append({
                "operation": "CREATE",
                "entity_type": "project",
                "content": {
                    "title": "Ausbau Radwegenetz",
                    "description": "Erweiterung der Fahrradinfrastruktur um 50 km",
                    "timeline": "2025-2027",
                    "responsible_department": "Tiefbauamt"
                },
                "confidence": 0.85,
                "source_pages": [1],
                "source_quote": "Ausbau des Radwegenetzes um 50 km"
            })

        if "elektrobus" in prompt_lower or "öpnv" in prompt_lower:
            operations.append({
                "operation": "CREATE",
                "entity_type": "project",
                "content": {
                    "title": "Elektrobus-Flotte",
                    "description": "Umstellung auf Elektrobusse im ÖPNV",
                    "timeline": "2025-2026",
                    "budget": "2.5 Millionen Euro"
                },
                "confidence": 0.82,
                "source_pages": [1],
                "source_quote": "Einführung von Elektrobussen"
            })

        # Measure creation
        if "sanierung" in prompt_lower or "gebäude" in prompt_lower:
            operations.append({
                "operation": "CREATE",
                "entity_type": "measure",
                "content": {
                    "title": "Gebäudesanierung Schulen",
                    "description": "Energetische Sanierung von 20 Schulgebäuden",
                    "timeline": "2025-2026"
                },
                "confidence": 0.80,
                "source_pages": [2],
                "source_quote": "Sanierung von 20 Schulgebäuden"
            })

        # Indicator creation
        if "co2" in prompt_lower or "emission" in prompt_lower:
            operations.append({
                "operation": "CREATE",
                "entity_type": "indicator",
                "content": {
                    "title": "CO2-Reduktion Verkehrssektor",
                    "description": "Jährliche Reduzierung der CO2-Emissionen im Verkehrsbereich",
                    "unit": "Tonnen CO2/Jahr",
                    "target_values": "5.000 Tonnen Reduktion bis 2030",
                    "granularity": "annual",
                    "should_increase": False,
                    "data_source": "Umweltamt Monitoring"
                },
                "confidence": 0.87,
                "source_pages": [1, 2],
                "source_quote": "CO2-Emissionen bis 2030 um 40% reduziert"
            })

        if "erneuerbar" in prompt_lower or "solar" in prompt_lower or "photovoltaik" in prompt_lower:
            operations.append({
                "operation": "CREATE",
                "entity_type": "indicator",
                "content": {
                    "title": "Anteil erneuerbarer Energien",
                    "description": "Prozentualer Anteil erneuerbarer Energien am Gesamtverbrauch",
                    "unit": "Prozent (%)",
                    "target_values": "60% bis 2030",
                    "actual_values": "35% (Stand 2024)",
                    "granularity": "annual",
                    "should_increase": True
                },
                "confidence": 0.84,
                "source_pages": [2],
                "source_quote": "Anteil erneuerbarer Energien"
            })

        # Add connections if we have entities to connect
        if len(operations) >= 2:
            # Check if we have action fields and projects to connect
            action_fields = [op for op in operations if op.get("entity_type") == "action_field"]
            projects = [op for op in operations if op.get("entity_type") == "project"]

            if action_fields and projects:
                operations.append({
                    "operation": "CONNECT",
                    "entity_type": "action_field",
                    "entity_id": "af_1",  # Assume first action field gets ID af_1
                    "connections": [
                        {
                            "from_id": "af_1",
                            "to_id": "proj_1",  # Assume first project gets ID proj_1
                            "confidence": 0.75
                        }
                    ],
                    "confidence": 0.75
                })

        # If no specific patterns matched, create some minimal operations
        if not operations:
            operations.append({
                "operation": "CREATE",
                "entity_type": "action_field",
                "content": {
                    "title": "Allgemeine Nachhaltigkeitsmaßnahmen",
                    "description": "Übergreifende Nachhaltigkeitsstrategie"
                },
                "confidence": 0.7,
                "source_pages": [1],
                "source_quote": "Test content"
            })

        # Handle UPDATE operations if entity registry is present in prompt
        if "af_" in prompt and "proj_" in prompt:
            # This suggests we have existing entities - add some UPDATE operations
            operations.append({
                "operation": "UPDATE",
                "entity_type": "action_field",
                "entity_id": "af_1",
                "content": {
                    "description": "Erweiterte Beschreibung mit zusätzlichen Details"
                },
                "confidence": 0.78,
                "source_pages": [2],
                "source_quote": "Zusätzliche Informationen"
            })

        return ExtractionOperations(operations=operations)


def patch_llm_provider_for_testing():
    """
    Patch the LLM provider system to use mock provider.
    
    This function modifies the get_llm_provider function to return
    MockLLMProvider when LLM_BACKEND is set to 'mock'.
    """
    from . import llm_providers

    original_get_llm_provider = llm_providers.get_llm_provider

    def mock_get_llm_provider(*args, **kwargs):
        backend = kwargs.get("backend") or os.getenv("LLM_BACKEND", "mock")
        if backend == "mock":
            return MockLLMProvider()
        else:
            return original_get_llm_provider(*args, **kwargs)

    # Replace the function
    llm_providers.get_llm_provider = mock_get_llm_provider

    # Also patch in the extraction helpers module if it's imported
    try:
        from . import llm_providers as helpers_llm
        helpers_llm.get_llm_provider = mock_get_llm_provider
    except ImportError:
        pass


# Auto-patch if mock backend is requested
if os.getenv("LLM_BACKEND") == "mock":
    patch_llm_provider_for_testing()
