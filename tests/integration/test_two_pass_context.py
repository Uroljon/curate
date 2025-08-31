#!/usr/bin/env python3
"""
Test suite for two-pass operations extraction context propagation.

Verifies that context (entities and ID mappings) is correctly propagated
between chunks and passes in the two-pass extraction system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    
    # Simple test runner for direct execution
    class pytest:
        @staticmethod
        def mark(**kwargs):
            def decorator(func):
                return func
            return decorator
            
        @staticmethod
        def fixture(func):
            return func

from src.api.extraction_helpers import (
    build_operations_prompt,
    format_context_json,
    format_entity_registry,
    format_entity_id_mapping,
)
from src.core.schemas import (
    EnhancedActionField,
    EnhancedProject,
    EnhancedMeasure,
    EnhancedIndicator,
    EnrichedReviewJSON,
)


@pytest.fixture
def empty_state():
    """Create empty extraction state."""
    return EnrichedReviewJSON(
        action_fields=[],
        projects=[],
        measures=[],
        indicators=[]
    )


@pytest.fixture
def populated_state():
    """Create state with entities from previous chunks."""
    return EnrichedReviewJSON(
        action_fields=[
            EnhancedActionField(
                id="af_1",
                content={"title": "Mobilität und Verkehr", "description": "Nachhaltige Verkehrslösungen"},
                connections=[]
            ),
            EnhancedActionField(
                id="af_2", 
                content={"title": "Energie", "description": "Erneuerbare Energien"},
                connections=[]
            )
        ],
        projects=[
            EnhancedProject(
                id="proj_1",
                content={"title": "Radwegeausbau", "description": "Ausbau des städtischen Radwegenetzes"},
                connections=[]
            )
        ],
        measures=[
            EnhancedMeasure(
                id="msr_1",
                content={"title": "LED-Straßenbeleuchtung", "description": "Umstellung auf LED"},
                connections=[]
            )
        ],
        indicators=[
            EnhancedIndicator(
                id="ind_1", 
                content={"title": "CO2-Reduktion", "description": "Jährliche CO2-Einsparung", "unit": "Tonnen CO2"},
                connections=[]
            )
        ]
    )


def test_build_operations_prompt_nodes_mode_empty_state(empty_state):
    """Test nodes prompt building with empty state."""
    chunk_text = "Die Stadt plant neue Radwege im Stadtzentrum."
    page_numbers = [1, 2]
    
    prompt = build_operations_prompt("nodes", chunk_text, empty_state, page_numbers)
    
    # Should contain empty state message
    assert "ERSTER CHUNK: Noch keine Entities extrahiert" in prompt
    
    # Should contain chunk text and pages
    assert chunk_text in prompt
    assert "Seiten 1, 2" in prompt
    
    # Should be nodes-only mode
    assert "NUR ENTITY-EXTRAKTION" in prompt
    assert "CREATE: Neue Entities erstellen" in prompt
    assert "UPDATE: Bestehende Entities anreichern" in prompt
    assert "CONNECT: VERBOTEN in diesem Pass" in prompt


def test_build_operations_prompt_nodes_mode_populated_state(populated_state):
    """Test nodes prompt building with populated state from previous chunks."""
    chunk_text = "Weitere Projekte zur Elektromobilität sind geplant."
    page_numbers = [3, 4]
    
    prompt = build_operations_prompt("nodes", chunk_text, populated_state, page_numbers)
    
    # Should contain entity registry with existing entities
    assert "ENTITY REGISTRY" in prompt
    assert "Mobilität und Verkehr" in prompt
    assert "Radwegeausbau" in prompt
    assert "LED-Straßenbeleuchtung" in prompt
    assert "CO2-Reduktion" in prompt
    
    # Should contain ID mapping for existing entities
    assert "af_1 → Mobilität und Verkehr" in prompt
    assert "proj_1 → Radwegeausbau" in prompt
    assert "msr_1 → LED-Straßenbeleuchtung" in prompt
    assert "ind_1 → CO2-Reduktion" in prompt
    
    # Should guide toward UPDATE over CREATE
    assert "IMMER prüfen ob Entity schon im REGISTRY existiert" in prompt
    # The prompt now uses more specific rules instead of the simplified text
    assert "Bei EXAKT gleichen Namen" in prompt  # Updated text in prompt


def test_build_operations_prompt_connections_mode_populated_state(populated_state):
    """Test connections prompt building with populated state."""
    chunk_text = "Die Radwege sind Teil der Mobilitätsstrategie."
    page_numbers = [5]
    
    prompt = build_operations_prompt("connections", chunk_text, populated_state, page_numbers)
    
    # Should contain complete ID mapping for connections
    assert "af_1 → Mobilität und Verkehr" in prompt
    assert "proj_1 → Radwegeausbau" in prompt
    
    # Should be connections-only mode
    assert "NUR VERBINDUNGS-EXTRAKTION" in prompt
    assert "CREATE: VERBOTEN in diesem Pass" in prompt
    assert "UPDATE: VERBOTEN in diesem Pass" in prompt
    assert "CONNECT: Verbindungen zwischen bestehenden Entities" in prompt
    
    # Should emphasize ID-only connections
    assert "NUR exakte IDs aus ID-MAPPING-TABELLE verwenden" in prompt
    assert "NIEMALS Entity-Namen/Titel verwenden" in prompt


def test_context_propagation_entity_registry_completeness(populated_state):
    """Test that entity registry shows ALL entities without truncation."""
    registry = format_entity_registry(populated_state, include_descriptions=False)
    
    # Should show counts for each entity type
    assert "ACTION FIELDS (2 total)" in registry
    assert "PROJECTS (1 total)" in registry
    assert "MEASURES (1 total)" in registry
    assert "INDICATORS (1 total)" in registry
    
    # Should list ALL entity titles
    assert "Mobilität und Verkehr, Energie" in registry
    assert "Radwegeausbau" in registry
    assert "LED-Straßenbeleuchtung" in registry
    assert "CO2-Reduktion" in registry


def test_context_propagation_id_mapping_completeness(populated_state):
    """Test that ID mapping includes ALL entities for UPDATE/CONNECT operations."""
    id_mapping = format_entity_id_mapping(populated_state)
    
    # Should include ALL action fields
    assert "ACTION FIELD IDs:" in id_mapping
    assert "af_1 → Mobilität und Verkehr" in id_mapping
    assert "af_2 → Energie" in id_mapping
    
    # Should include ALL projects
    assert "PROJECT IDs:" in id_mapping
    assert "proj_1 → Radwegeausbau" in id_mapping
    
    # Should include ALL measures
    assert "MEASURE IDs:" in id_mapping
    assert "msr_1 → LED-Straßenbeleuchtung" in id_mapping
    
    # Should include ALL indicators  
    assert "INDICATOR IDs:" in id_mapping
    assert "ind_1 → CO2-Reduktion" in id_mapping


def test_context_state_accumulation_simulation():
    """Simulate context accumulation across multiple chunks."""
    # Start with empty state
    state = EnrichedReviewJSON(action_fields=[], projects=[], measures=[], indicators=[])
    
    # Chunk 1: Create first entities (empty state shows registry, not first chunk message)
    context_1 = format_context_json(state)
    assert "[None yet]" in context_1  # Empty state shows placeholder
    assert "No entities yet - use CREATE operations" in context_1
    
    # Simulate adding entities after chunk 1
    state.action_fields.append(
        EnhancedActionField(id="af_1", content={"title": "Mobilität"}, connections=[])
    )
    
    # Chunk 2: Should see entities from chunk 1
    context_2 = format_context_json(state)
    assert "Mobilität" in context_2
    assert "af_1 → Mobilität" in context_2
    assert "ACTION FIELDS (1 total)" in context_2
    
    # Simulate adding more entities
    state.projects.append(
        EnhancedProject(id="proj_1", content={"title": "Radwege"}, connections=[])
    )
    state.measures.append(
        EnhancedMeasure(id="msr_1", content={"title": "LED-Beleuchtung"}, connections=[])
    )
    
    # Chunk 3: Should see accumulated state
    context_3 = format_context_json(state)
    assert "Mobilität" in context_3
    assert "Radwege" in context_3
    assert "LED-Beleuchtung" in context_3
    assert "ACTION FIELDS (1 total)" in context_3
    assert "PROJECTS (1 total)" in context_3
    assert "MEASURES (1 total)" in context_3


def test_two_pass_mode_prompt_differences(populated_state):
    """Test that nodes and connections prompts have clear mode differences."""
    chunk_text = "Test chunk text"
    pages = [1]
    
    nodes_prompt = build_operations_prompt("nodes", chunk_text, populated_state, pages)
    connections_prompt = build_operations_prompt("connections", chunk_text, populated_state, pages)
    
    # Both should have same context (entity registry + ID mapping)
    assert "Mobilität und Verkehr" in nodes_prompt
    assert "Mobilität und Verkehr" in connections_prompt
    assert "af_1 → Mobilität und Verkehr" in nodes_prompt
    assert "af_1 → Mobilität und Verkehr" in connections_prompt
    
    # But different operational modes
    assert "NUR ENTITY-EXTRAKTION" in nodes_prompt
    assert "NUR VERBINDUNGS-EXTRAKTION" in connections_prompt
    
    assert "CREATE: Neue Entities erstellen" in nodes_prompt
    assert "CREATE: VERBOTEN in diesem Pass" in connections_prompt
    
    assert "CONNECT: VERBOTEN in diesem Pass" in nodes_prompt
    assert "CONNECT: Verbindungen zwischen bestehenden Entities" in connections_prompt


# Test runner for direct execution
def run_tests():
    """Run all tests directly without pytest."""
    print("🧪 Testing Two-Pass Context Propagation...")
    
    # Create fixtures
    empty = empty_state()
    populated = populated_state()
    
    # Run tests
    tests = [
        (test_build_operations_prompt_nodes_mode_empty_state, empty),
        (test_build_operations_prompt_nodes_mode_populated_state, populated),
        (test_build_operations_prompt_connections_mode_populated_state, populated),
        (test_context_propagation_entity_registry_completeness, populated),
        (test_context_propagation_id_mapping_completeness, populated),
        (test_context_state_accumulation_simulation, None),
        (test_two_pass_mode_prompt_differences, None),
    ]
    
    passed = 0
    failed = 0
    
    for test_func, fixture in tests:
        try:
            if fixture is not None:
                test_func(fixture)
            else:
                test_func()
            print(f"✅ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__}: {e}")
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    if PYTEST_AVAILABLE:
        pytest.main([__file__, "-v"])
    else:
        success = run_tests()
        sys.exit(0 if success else 1)