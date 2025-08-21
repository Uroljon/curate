#!/usr/bin/env python3
"""
Comprehensive test suite for context-awareness fix in operations extraction.

This test file verifies that the context-awareness improvements work as intended:
- Entity registry shows ALL entities (no truncation)
- Context is readable and scannable (not JSON dump) 
- All entity types are emphasized for duplicate checking
- UPDATE is preferred over CREATE for similar entities
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    # Try direct import first
    from src.api.extraction_helpers import (
        create_extraction_prompt,
        format_context_json,
        format_entity_id_mapping,
        format_entity_registry,
    )
    from src.core.schemas import (
        ConnectionWithConfidence,
        EnhancedActionField,
        EnhancedIndicator,
        EnhancedMeasure,
        EnhancedProject,
        EnrichedReviewJSON,
    )

    # Try importing pytest, but make it optional
    try:
        import pytest
        PYTEST_AVAILABLE = True
    except ImportError:
        PYTEST_AVAILABLE = False
        print("Note: pytest not available, using simple test runner")

        # Create a simple fixture decorator
        def pytest_fixture(func):
            def wrapper():
                return func()
            wrapper.__name__ = func.__name__
            return wrapper

        class pytest:
            fixture = staticmethod(pytest_fixture)

            @staticmethod
            def main(args):
                # Simple test runner
                print("Running tests with simple runner...")
                run_simple_tests()

except ImportError as e:
    print(f"Import error: {e}")
    print("Please run these tests in the virtual environment: source venv/bin/activate")
    sys.exit(1)


# ============================================================================
# FIXTURES - Mock Data for Testing
# ============================================================================


@pytest.fixture
def empty_state() -> EnrichedReviewJSON:
    """Create empty EnrichedReviewJSON state."""
    return EnrichedReviewJSON(action_fields=[], projects=[], measures=[], indicators=[])


@pytest.fixture
def small_state() -> EnrichedReviewJSON:
    """Create state with a few entities of each type."""
    return EnrichedReviewJSON(
        action_fields=[
            EnhancedActionField(
                id="af_1",
                content={"title": "Mobilität und Verkehr"},
                connections=[],
            ),
            EnhancedActionField(
                id="af_2",
                content={"title": "Klimaschutz"},
                connections=[],
            ),
        ],
        projects=[
            EnhancedProject(
                id="proj_1",
                content={"title": "Radwegeausbau"},
                connections=[],
            ),
            EnhancedProject(
                id="proj_2",
                content={"title": "E-Mobilität Förderung"},
                connections=[],
            ),
        ],
        measures=[
            EnhancedMeasure(
                id="msr_1",
                content={"title": "Radweg Hauptstraße"},
                connections=[],
            ),
        ],
        indicators=[
            EnhancedIndicator(
                id="ind_1",
                content={"title": "CO2-Reduktion Verkehr"},
                connections=[],
            ),
        ],
    )


@pytest.fixture
def state_with_duplicates() -> EnrichedReviewJSON:
    """Create state with potential duplicate entity names."""
    return EnrichedReviewJSON(
        action_fields=[
            EnhancedActionField(
                id="af_1",
                content={"title": "Mobilität und Verkehr"},
                connections=[],
            ),
            EnhancedActionField(
                id="af_2",
                content={"title": "Mobilität & Verkehr"},  # Similar to af_1
                connections=[],
            ),
        ],
        projects=[
            EnhancedProject(
                id="proj_1",
                content={"title": "Radwegeausbau"},
                connections=[],
            ),
            EnhancedProject(
                id="proj_2",
                content={"title": "Ausbau Radwege"},  # Similar to proj_1
                connections=[],
            ),
        ],
        measures=[],
        indicators=[],
    )


@pytest.fixture
def large_state() -> EnrichedReviewJSON:
    """Create state with many entities to test performance and no truncation."""
    action_fields = []
    projects = []
    measures = []
    indicators = []

    # Create 25 entities of each type
    for i in range(25):
        action_fields.append(
            EnhancedActionField(
                id=f"af_{i+1}",
                content={"title": f"Handlungsfeld {i+1}"},
                connections=[],
            )
        )
        projects.append(
            EnhancedProject(
                id=f"proj_{i+1}",
                content={"title": f"Projekt {i+1}"},
                connections=[],
            )
        )
        measures.append(
            EnhancedMeasure(
                id=f"msr_{i+1}",
                content={"title": f"Maßnahme {i+1}"},
                connections=[],
            )
        )
        indicators.append(
            EnhancedIndicator(
                id=f"ind_{i+1}",
                content={"title": f"Indikator {i+1}"},
                connections=[],
            )
        )

    return EnrichedReviewJSON(
        action_fields=action_fields,
        projects=projects,
        measures=measures,
        indicators=indicators,
    )


@pytest.fixture
def state_with_special_chars() -> EnrichedReviewJSON:
    """Create state with special characters and unicode."""
    return EnrichedReviewJSON(
        action_fields=[
            EnhancedActionField(
                id="af_1",
                content={"title": "Mobilität & Verkehr (Hauptthema)"},
                connections=[],
            ),
            EnhancedActionField(
                id="af_2",
                content={"title": "Klimaschutz: CO₂-Reduktion"},
                connections=[],
            ),
        ],
        projects=[
            EnhancedProject(
                id="proj_1",
                content={"title": "E-Mobilität \"Förderung\" 2024-2030"},
                connections=[],
            ),
        ],
        measures=[
            EnhancedMeasure(
                id="msr_1",
                content={"title": "Straßenbau für Räder & Fußgänger"},
                connections=[],
            ),
        ],
        indicators=[
            EnhancedIndicator(
                id="ind_1",
                content={"title": "50% weniger CO₂ bis 2030"},
                connections=[],
            ),
        ],
    )


# ============================================================================
# ENTITY REGISTRY FORMAT TESTS
# ============================================================================


def test_format_entity_registry_empty_state(empty_state):
    """Test entity registry format when no entities exist."""
    registry = format_entity_registry(empty_state)

    # Should contain the header
    assert "ENTITY REGISTRY - CHECK BEFORE ANY CREATE OPERATION" in registry

    # Should show [None yet] for all categories
    assert "ACTION FIELDS (0 total):" in registry
    assert "[None yet]" in registry
    assert "PROJECTS (0 total):" in registry
    assert "MEASURES (0 total):" in registry
    assert "INDICATORS (0 total):" in registry

    # Should have proper formatting
    assert "╔" in registry  # Unicode box characters
    assert "═" in registry


def test_format_entity_registry_with_all_entities(small_state):
    """Test entity registry shows ALL entities with proper counts."""
    registry = format_entity_registry(small_state)

    # Check counts are correct
    assert "ACTION FIELDS (2 total):" in registry
    assert "PROJECTS (2 total):" in registry
    assert "MEASURES (1 total):" in registry
    assert "INDICATORS (1 total):" in registry

    # Check all entity titles are present
    assert "Mobilität und Verkehr" in registry
    assert "Klimaschutz" in registry
    assert "Radwegeausbau" in registry
    assert "E-Mobilität Förderung" in registry
    assert "Radweg Hauptstraße" in registry
    assert "CO2-Reduktion Verkehr" in registry

    # Should not contain [None yet] when entities exist
    assert "[None yet]" not in registry


def test_format_entity_registry_no_truncation(large_state):
    """Test that registry shows ALL entities without truncation."""
    registry = format_entity_registry(large_state)

    # Check counts show all 25 entities of each type
    assert "ACTION FIELDS (25 total):" in registry
    assert "PROJECTS (25 total):" in registry
    assert "MEASURES (25 total):" in registry
    assert "INDICATORS (25 total):" in registry

    # Check first and last entities are present (no truncation)
    assert "Handlungsfeld 1" in registry
    assert "Handlungsfeld 25" in registry
    assert "Projekt 1" in registry
    assert "Projekt 25" in registry
    assert "Maßnahme 1" in registry
    assert "Maßnahme 25" in registry
    assert "Indikator 1" in registry
    assert "Indikator 25" in registry


def test_format_entity_registry_special_characters(state_with_special_chars):
    """Test entities with special characters are displayed correctly."""
    registry = format_entity_registry(state_with_special_chars)

    # Check special characters are preserved
    assert "Mobilität & Verkehr (Hauptthema)" in registry
    assert "Klimaschutz: CO₂-Reduktion" in registry
    assert "E-Mobilität \"Förderung\" 2024-2030" in registry
    assert "Straßenbau für Räder & Fußgänger" in registry
    assert "50% weniger CO₂ bis 2030" in registry


# ============================================================================
# ID MAPPING TESTS
# ============================================================================


def test_format_entity_id_mapping_empty(empty_state):
    """Test ID mapping for empty state."""
    id_mapping = format_entity_id_mapping(empty_state)

    assert id_mapping == "No entities yet - use CREATE operations"


def test_format_entity_id_mapping_complete(small_state):
    """Test ID mapping shows all entity IDs correctly."""
    id_mapping = format_entity_id_mapping(small_state)

    # Check section headers
    assert "ACTION FIELD IDs:" in id_mapping
    assert "PROJECT IDs:" in id_mapping
    assert "MEASURE IDs:" in id_mapping
    assert "INDICATOR IDs:" in id_mapping

    # Check ID mappings with arrow format
    assert "af_1 → Mobilität und Verkehr" in id_mapping
    assert "af_2 → Klimaschutz" in id_mapping
    assert "proj_1 → Radwegeausbau" in id_mapping
    assert "proj_2 → E-Mobilität Förderung" in id_mapping
    assert "msr_1 → Radweg Hauptstraße" in id_mapping
    assert "ind_1 → CO2-Reduktion Verkehr" in id_mapping


def test_id_mapping_preserves_all_entities(large_state):
    """Test ID mapping includes all entities (no truncation)."""
    id_mapping = format_entity_id_mapping(large_state)

    # Check first and last IDs are present
    assert "af_1 → Handlungsfeld 1" in id_mapping
    assert "af_25 → Handlungsfeld 25" in id_mapping
    assert "proj_1 → Projekt 1" in id_mapping
    assert "proj_25 → Projekt 25" in id_mapping
    assert "msr_1 → Maßnahme 1" in id_mapping
    assert "msr_25 → Maßnahme 25" in id_mapping
    assert "ind_1 → Indikator 1" in id_mapping
    assert "ind_25 → Indikator 25" in id_mapping


# ============================================================================
# CONTEXT JSON FORMAT TESTS
# ============================================================================


def test_format_context_json_empty_state():
    """Test context format for empty state."""
    context = format_context_json(None)

    assert context == "ERSTER CHUNK: Noch keine Entities extrahiert. Beginnen Sie mit CREATE-Operationen."


def test_format_context_json_replaces_json_dump(small_state):
    """Test that context uses registry format instead of JSON dump."""
    context = format_context_json(small_state)

    # Should NOT contain JSON dump markers
    assert "AKTUELLER EXTRAKTIONSSTAND (bisher gefundene Strukturen):" not in context
    assert '{"action_fields":' not in context
    assert '"id":' not in context

    # Should contain registry format
    assert "ENTITY REGISTRY - CHECK BEFORE ANY CREATE OPERATION" in context
    assert "ACTION FIELD IDs:" in context

    # Should contain critical rules
    assert "KRITISCHE REGELN:" in context
    assert "IMMER prüfen ob Entity schon im REGISTRY existiert" in context


def test_context_includes_duplicate_examples(small_state):
    """Test that context includes duplicate detection examples."""
    context = format_context_json(small_state)

    # Check duplicate examples are present
    assert "Mobilität und Verkehr" in context
    assert "Mobilität & Verkehr" in context
    assert "Verkehrswesen" in context
    assert "Radwegeausbau" in context
    assert "Ausbau Radwege" in context
    assert "Radverkehrsnetz" in context
    assert "CO2-Reduktion" in context
    assert "CO₂-Reduktion" in context
    assert "Kohlendioxid-Reduktion" in context


def test_context_contains_all_four_rules(small_state):
    """Test that context contains all 4 critical rules."""
    context = format_context_json(small_state)

    # Check all 4 numbered rules are present
    assert "1. IMMER prüfen ob Entity schon im REGISTRY existiert" in context
    assert "2. Bei ähnlichen Namen → UPDATE statt CREATE" in context
    assert "3. Beispiele für Duplikate:" in context
    assert "4. NUR CREATE wenn wirklich neu und einzigartig" in context


# ============================================================================
# PROMPT GENERATION TESTS
# ============================================================================


def test_operations_prompt_uses_registry_format(small_state):
    """Test that operations prompt uses entity registry instead of JSON dump."""
    prompt = create_extraction_prompt(
        "operations",
        "Test chunk text about mobility.",
        small_state,
        [1, 2]
    )

    # Should contain registry format
    assert "ENTITY REGISTRY - CHECK BEFORE ANY CREATE OPERATION" in prompt
    assert "ACTION FIELD IDs:" in prompt

    # Should NOT contain old JSON dump format
    assert "AKTUELLER EXTRAKTIONSSTAND (bisher gefundene Strukturen):" not in prompt
    assert '{"action_fields":' not in prompt


def test_prompt_emphasizes_all_entity_types(small_state):
    """Test that prompt emphasizes checking ALL entity types."""
    prompt = create_extraction_prompt(
        "operations",
        "Test chunk text.",
        small_state,
        [1]
    )

    # Should mention ENTITY REGISTRY (not just action fields)
    assert "ENTITY REGISTRY" in prompt
    assert "1. Entity bereits im ENTITY REGISTRY? → UPDATE verwenden" in prompt
    assert "2. Ähnlicher Name im REGISTRY? → UPDATE verwenden" in prompt


def test_prompt_includes_decision_logic(small_state):
    """Test that prompt includes 4-step decision process."""
    prompt = create_extraction_prompt(
        "operations",
        "Test chunk text.",
        small_state,
        [1]
    )

    # Check 4-step decision process
    assert "1. Entity bereits im ENTITY REGISTRY? → UPDATE verwenden" in prompt
    assert "2. Ähnlicher Name im REGISTRY? → UPDATE verwenden" in prompt
    assert "3. Teilweise Überlappung? → UPDATE verwenden" in prompt
    assert "4. Wirklich neu? → Erst dann CREATE" in prompt


def test_prompt_quality_control_references_registry(small_state):
    """Test that quality control section references ENTITY REGISTRY."""
    prompt = create_extraction_prompt(
        "operations",
        "Test chunk text.",
        small_state,
        [1]
    )

    # Should reference ENTITY REGISTRY in quality control
    assert "Verwenden Sie NUR exakte Entity-IDs aus ENTITY REGISTRY" in prompt

    # Should NOT reference old format
    assert "AKTUELLER EXTRAKTIONSSTAND" not in prompt


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_extraction_context_flow_integration(small_state):
    """Test complete context flow from state to prompt."""
    # This tests the full integration: state → registry → context → prompt

    # Step 1: Generate registry
    registry = format_entity_registry(small_state)
    assert "Mobilität und Verkehr" in registry
    assert "Klimaschutz" in registry

    # Step 2: Generate ID mapping
    id_mapping = format_entity_id_mapping(small_state)
    assert "af_1 → Mobilität und Verkehr" in id_mapping

    # Step 3: Generate context
    context = format_context_json(small_state)
    assert registry.strip() in context
    assert id_mapping in context

    # Step 4: Generate prompt
    prompt = create_extraction_prompt("operations", "Test text", small_state, [1])
    assert context in prompt


def test_duplicate_detection_scenario(state_with_duplicates):
    """Test scenario where duplicates should be detected."""
    context = format_context_json(state_with_duplicates)

    # Both similar entities should be visible in registry
    assert "Mobilität und Verkehr" in context
    assert "Mobilität & Verkehr" in context
    assert "Radwegeausbau" in context
    assert "Ausbau Radwege" in context

    # Decision logic should promote UPDATE
    assert "Bei ähnlichen Namen → UPDATE statt CREATE" in context
    assert "Teilweise Überlappung? → UPDATE verwenden" in context


# ============================================================================
# EDGE CASES TESTS
# ============================================================================


def test_entity_with_empty_title():
    """Test handling of entities with missing/empty titles."""
    state = EnrichedReviewJSON(
        action_fields=[
            EnhancedActionField(
                id="af_1",
                content={"title": ""},  # Empty title
                connections=[],
            ),
            EnhancedActionField(
                id="af_2",
                content={},  # Missing title
                connections=[],
            ),
        ],
        projects=[],
        measures=[],
        indicators=[],
    )

    # Should handle gracefully without crashes
    registry = format_entity_registry(state)
    assert "ACTION FIELDS (2 total):" in registry

    id_mapping = format_entity_id_mapping(state)
    assert "af_1 →" in id_mapping
    assert "af_2 →" in id_mapping


def test_very_long_entity_titles():
    """Test handling of very long entity titles."""
    long_title = "Sehr langes Handlungsfeld mit vielen Worten und Details " * 5

    state = EnrichedReviewJSON(
        action_fields=[
            EnhancedActionField(
                id="af_1",
                content={"title": long_title},
                connections=[],
            ),
        ],
        projects=[],
        measures=[],
        indicators=[],
    )

    # Should handle long titles without issues
    registry = format_entity_registry(state)
    assert long_title in registry

    id_mapping = format_entity_id_mapping(state)
    assert f"af_1 → {long_title}" in id_mapping


def run_simple_tests():
    """Simple test runner when pytest is not available."""
    print("=" * 80)
    print("CONTEXT-AWARENESS FIX VERIFICATION TESTS")
    print("=" * 80)

    # Create test data
    empty_state = EnrichedReviewJSON(action_fields=[], projects=[], measures=[], indicators=[])

    small_state = EnrichedReviewJSON(
        action_fields=[
            EnhancedActionField(id="af_1", content={"title": "Mobilität und Verkehr"}, connections=[]),
            EnhancedActionField(id="af_2", content={"title": "Klimaschutz"}, connections=[]),
        ],
        projects=[
            EnhancedProject(id="proj_1", content={"title": "Radwegeausbau"}, connections=[]),
        ],
        measures=[
            EnhancedMeasure(id="msr_1", content={"title": "Radweg Hauptstraße"}, connections=[]),
        ],
        indicators=[
            EnhancedIndicator(id="ind_1", content={"title": "CO2-Reduktion Verkehr"}, connections=[]),
        ],
    )

    tests_passed = 0
    tests_total = 0

    def run_test(test_name, test_func):
        nonlocal tests_passed, tests_total
        tests_total += 1
        try:
            test_func()
            print(f"✅ {test_name}")
            tests_passed += 1
        except Exception as e:
            print(f"❌ {test_name}: {e}")

    # Test 1: Entity registry shows all entities
    def test_registry_shows_all():
        registry = format_entity_registry(small_state)
        assert "ACTION FIELDS (2 total):" in registry
        assert "PROJECTS (1 total):" in registry
        assert "Mobilität und Verkehr" in registry
        assert "Klimaschutz" in registry
        assert "Radwegeausbau" in registry
        assert "[None yet]" not in registry

    run_test("Entity registry shows all entities", test_registry_shows_all)

    # Test 2: ID mapping includes all entities
    def test_id_mapping_complete():
        id_mapping = format_entity_id_mapping(small_state)
        assert "af_1 → Mobilität und Verkehr" in id_mapping
        assert "af_2 → Klimaschutz" in id_mapping
        assert "proj_1 → Radwegeausbau" in id_mapping
        assert "msr_1 → Radweg Hauptstraße" in id_mapping
        assert "ind_1 → CO2-Reduktion Verkehr" in id_mapping

    run_test("ID mapping includes all entities", test_id_mapping_complete)

    # Test 3: Context uses registry format (not JSON dump)
    def test_context_uses_registry():
        context = format_context_json(small_state)
        assert "ENTITY REGISTRY - CHECK BEFORE ANY CREATE OPERATION" in context
        assert "ACTION FIELD IDs:" in context
        assert "KRITISCHE REGELN:" in context
        # Should NOT contain old JSON dump
        assert '"action_fields":' not in context
        assert "AKTUELLER EXTRAKTIONSSTAND (bisher gefundene Strukturen):" not in context

    run_test("Context uses registry format (not JSON dump)", test_context_uses_registry)

    # Test 4: Context includes duplicate examples
    def test_duplicate_examples():
        context = format_context_json(small_state)
        assert "Mobilität und Verkehr" in context
        assert "Mobilität & Verkehr" in context
        assert "Radwegeausbau" in context
        assert "Ausbau Radwege" in context
        assert "CO2-Reduktion" in context
        assert "CO₂-Reduktion" in context

    run_test("Context includes duplicate examples", test_duplicate_examples)

    # Test 5: Prompt uses registry format
    def test_prompt_uses_registry():
        prompt = create_extraction_prompt("operations", "Test text", small_state, [1])
        assert "ENTITY REGISTRY" in prompt
        assert "1. Entity bereits im ENTITY REGISTRY? → UPDATE verwenden" in prompt
        # Should NOT contain old format
        assert "AKTUELLER EXTRAKTIONSSTAND" not in prompt

    run_test("Prompt uses registry format", test_prompt_uses_registry)

    # Test 6: Empty state handling
    def test_empty_state_handling():
        context = format_context_json(None)
        assert context == "ERSTER CHUNK: Noch keine Entities extrahiert. Beginnen Sie mit CREATE-Operationen."

        registry = format_entity_registry(empty_state)
        assert "[None yet]" in registry
        assert "ACTION FIELDS (0 total):" in registry

        id_mapping = format_entity_id_mapping(empty_state)
        assert id_mapping == "No entities yet - use CREATE operations"

    run_test("Empty state handling", test_empty_state_handling)

    print("=" * 80)
    print(f"RESULTS: {tests_passed}/{tests_total} tests passed")
    if tests_passed == tests_total:
        print("🎉 ALL TESTS PASSED! Context-awareness fix is working correctly.")
        print("Key improvements verified:")
        print("  ✓ Entity registry shows ALL entities (no truncation)")
        print("  ✓ Context is readable and scannable (not JSON dump)")
        print("  ✓ All entity types are emphasized for duplicate checking")
        print("  ✓ UPDATE is preferred over CREATE via decision logic")
        print("  ✓ Duplicate examples help LLM identify similar entities")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    print("=" * 80)


if __name__ == "__main__":
    if PYTEST_AVAILABLE:
        # Run tests with pytest if available
        pytest.main([__file__, "-v"])
    else:
        # Run simple tests
        run_simple_tests()
