#!/usr/bin/env python3
"""
Comprehensive test suite for critical functions in operations-based extraction.

Tests the core functions responsible for entity registry management, prompt generation,
and operations processing to identify bugs and edge case failures.
"""

import sys
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import the functions we want to test
from src.api.extraction_helpers import (
    create_extraction_prompt,
    format_context_json,
    format_entity_id_mapping,
    format_entity_registry,
)
from src.core.operations_schema import (
    EntityOperation,
    OperationType,
)
from src.core.schemas import (
    ConnectionWithConfidence,
    EnhancedActionField,
    EnhancedIndicator,
    EnhancedMeasure,
    EnhancedProject,
    EnrichedReviewJSON,
)
from src.extraction.operations_executor import OperationExecutor, validate_operations


class _TestResult:
    """Simple test result container."""

    def __init__(self, test_name: str, passed: bool, error_msg: str = "", details: str = ""):
        self.test_name = test_name
        self.passed = passed
        self.error_msg = error_msg
        self.details = details


class _TestRunner:
    """Simple test runner."""

    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def run_test(self, test_func, test_name: str) -> _TestResult:
        """Run a single test function."""
        try:
            test_func()
            result = _TestResult(test_name, True)
            self.passed += 1
            print(f"✅ {test_name}")
        except AssertionError as e:
            result = _TestResult(test_name, False, str(e), "Assertion failed")
            self.failed += 1
            print(f"❌ {test_name}: {e}")
        except Exception as e:
            result = _TestResult(test_name, False, str(e), "Unexpected error")
            self.failed += 1
            print(f"💥 {test_name}: {e}")

        self.results.append(result)
        return result

    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY: {self.passed}/{total} passed")
        print(f"{'='*60}")

        if self.failed > 0:
            print(f"\n❌ FAILED TESTS ({self.failed}):")
            for result in self.results:
                if not result.passed:
                    print(f"  - {result.test_name}: {result.error_msg}")


# =============================================================================
# Test Data Helpers
# =============================================================================

def create_sample_state() -> EnrichedReviewJSON:
    """Create a sample state with various entities."""
    return EnrichedReviewJSON(
        action_fields=[
            EnhancedActionField(
                id="af_1",
                content={"title": "Mobilität und Verkehr", "description": "Nachhaltige Verkehrslösungen"},
                connections=[],
                sources=None
            ),
            EnhancedActionField(
                id="af_2",
                content={"title": "Klimaschutz & Umwelt", "description": "Umweltschutzmaßnahmen"},
                connections=[],
                sources=None
            )
        ],
        projects=[
            EnhancedProject(
                id="proj_1",
                content={"title": "Radwege ausbauen", "description": "Ausbau des Radwegenetzes"},
                connections=[],
                sources=None
            )
        ],
        measures=[
            EnhancedMeasure(
                id="msr_1",
                content={"title": "Neue Radwege planen", "description": "Planung neuer Radwege"},
                connections=[],
                sources=None
            )
        ],
        indicators=[
            EnhancedIndicator(
                id="ind_1",
                content={"title": "CO2-Reduktion", "unit": "Tonnen/Jahr", "target_values": "500"},
                connections=[],
                sources=None
            )
        ]
    )


def create_empty_state() -> EnrichedReviewJSON:
    """Create an empty state."""
    return EnrichedReviewJSON(
        action_fields=[],
        projects=[],
        measures=[],
        indicators=[]
    )


def create_large_state() -> EnrichedReviewJSON:
    """Create a state with many entities to test scaling."""
    action_fields = []
    projects = []
    measures = []
    indicators = []

    # Create 50 entities of each type
    for i in range(50):
        action_fields.append(EnhancedActionField(
            id=f"af_{i+1}",
            content={"title": f"Action Field {i+1}", "description": f"Description {i+1}"},
            connections=[],
            sources=None
        ))

        projects.append(EnhancedProject(
            id=f"proj_{i+1}",
            content={"title": f"Project {i+1}", "description": f"Project description {i+1}"},
            connections=[],
            sources=None
        ))

        measures.append(EnhancedMeasure(
            id=f"msr_{i+1}",
            content={"title": f"Measure {i+1}", "description": f"Measure description {i+1}"},
            connections=[],
            sources=None
        ))

        indicators.append(EnhancedIndicator(
            id=f"ind_{i+1}",
            content={"title": f"Indicator {i+1}", "unit": "Units", "target_values": f"Target {i+1}"},
            connections=[],
            sources=None
        ))

    return EnrichedReviewJSON(
        action_fields=action_fields,
        projects=projects,
        measures=measures,
        indicators=indicators
    )


def create_unicode_state() -> EnrichedReviewJSON:
    """Create a state with Unicode/German characters."""
    return EnrichedReviewJSON(
        action_fields=[
            EnhancedActionField(
                id="af_1",
                content={"title": "Mobilität & Verkehr", "description": "Straßenverkehrsförderung"},
                connections=[],
                sources=None
            ),
            EnhancedActionField(
                id="af_2",
                content={"title": "Energieeffizienz", "description": "Wärmedämmung & Heizungsoptimierung"},
                connections=[],
                sources=None
            )
        ],
        projects=[
            EnhancedProject(
                id="proj_1",
                content={"title": "Fahrradwege erweitern", "description": "Überdachte Fahrradständer"},
                connections=[],
                sources=None
            )
        ],
        measures=[],
        indicators=[]
    )


def create_edge_case_state() -> EnrichedReviewJSON:
    """Create a state with edge case data."""
    return EnrichedReviewJSON(
        action_fields=[
            EnhancedActionField(
                id="af_1",
                content={"title": "", "description": "Empty title case"},  # Empty title
                connections=[],
                sources=None
            ),
            EnhancedActionField(
                id="af_2",
                content={},  # Missing title entirely
                connections=[],
                sources=None
            ),
            EnhancedActionField(
                id="af_3",
                content={"title": "A" * 200},  # Very long title
                connections=[],
                sources=None
            )
        ],
        projects=[],
        measures=[],
        indicators=[]
    )


# =============================================================================
# Tests for format_entity_registry()
# =============================================================================

def test_format_entity_registry_empty():
    """Test format_entity_registry with empty state."""
    empty_state = create_empty_state()
    result = format_entity_registry(empty_state)

    assert "ACTION FIELDS (0 total):" in result
    assert "[None yet]" in result
    assert "PROJECTS (0 total):" in result
    assert "MEASURES (0 total):" in result
    assert "INDICATORS (0 total):" in result


def test_format_entity_registry_sample():
    """Test format_entity_registry with sample state."""
    sample_state = create_sample_state()
    result = format_entity_registry(sample_state)

    assert "ACTION FIELDS (2 total):" in result
    assert "Mobilität und Verkehr" in result
    assert "Klimaschutz & Umwelt" in result
    assert "PROJECTS (1 total):" in result
    assert "Radwege ausbauen" in result
    assert "MEASURES (1 total):" in result
    assert "Neue Radwege planen" in result
    assert "INDICATORS (1 total):" in result
    assert "CO2-Reduktion" in result


def test_format_entity_registry_large():
    """Test format_entity_registry with many entities (no truncation)."""
    large_state = create_large_state()
    result = format_entity_registry(large_state)

    # Should show all 50 entities, no truncation
    assert "ACTION FIELDS (50 total):" in result
    assert "PROJECTS (50 total):" in result
    assert "MEASURES (50 total):" in result
    assert "INDICATORS (50 total):" in result

    # Check that first and last entities are both present
    assert "Action Field 1" in result
    assert "Action Field 50" in result
    assert "Project 1" in result
    assert "Project 50" in result


def test_format_entity_registry_unicode():
    """Test format_entity_registry with Unicode characters."""
    unicode_state = create_unicode_state()
    result = format_entity_registry(unicode_state)

    assert "Mobilität & Verkehr" in result
    assert "Energieeffizienz" in result
    assert "Fahrradwege erweitern" in result


def test_format_entity_registry_edge_cases():
    """Test format_entity_registry with edge case data."""
    edge_state = create_edge_case_state()
    result = format_entity_registry(edge_state)

    assert "ACTION FIELDS (3 total):" in result
    # Should handle empty titles gracefully
    assert result is not None
    # Very long title should be included (no truncation)
    assert "A" * 200 in result


# =============================================================================
# Tests for format_entity_id_mapping()
# =============================================================================

def test_format_entity_id_mapping_empty():
    """Test format_entity_id_mapping with empty state."""
    empty_state = create_empty_state()
    result = format_entity_id_mapping(empty_state)

    assert result == "No entities yet - use CREATE operations"


def test_format_entity_id_mapping_sample():
    """Test format_entity_id_mapping with sample state."""
    sample_state = create_sample_state()
    result = format_entity_id_mapping(sample_state)

    assert "ACTION FIELD IDs:" in result
    assert "af_1 → Mobilität und Verkehr" in result
    assert "af_2 → Klimaschutz & Umwelt" in result
    assert "PROJECT IDs:" in result
    assert "proj_1 → Radwege ausbauen" in result
    assert "MEASURE IDs:" in result
    assert "msr_1 → Neue Radwege planen" in result
    assert "INDICATOR IDs:" in result
    assert "ind_1 → CO2-Reduktion" in result


def test_format_entity_id_mapping_duplicates():
    """Test format_entity_id_mapping with entities that have similar titles."""
    # Create state with similar but different entities
    state = EnrichedReviewJSON(
        action_fields=[
            EnhancedActionField(
                id="af_1",
                content={"title": "Mobilität und Verkehr"},
                connections=[],
                sources=None
            ),
            EnhancedActionField(
                id="af_2",
                content={"title": "Mobilität & Verkehr"},  # Similar but different
                connections=[],
                sources=None
            )
        ],
        projects=[],
        measures=[],
        indicators=[]
    )

    result = format_entity_id_mapping(state)

    # Both should be listed with different IDs
    assert "af_1 → Mobilität und Verkehr" in result
    assert "af_2 → Mobilität & Verkehr" in result


# =============================================================================
# Tests for format_context_json()
# =============================================================================

def test_format_context_json_empty():
    """Test format_context_json with None/empty context."""
    result = format_context_json(None)
    expected = "ERSTER CHUNK: Noch keine Entities extrahiert. Beginnen Sie mit CREATE-Operationen."
    assert result == expected

    result = format_context_json(create_empty_state())
    # Should call format_entity_registry and format_entity_id_mapping
    assert "No entities yet" in result or "None yet" in result


def test_format_context_json_sample():
    """Test format_context_json with sample state."""
    sample_state = create_sample_state()
    result = format_context_json(sample_state)

    # Should contain registry and ID mapping
    assert "ENTITY REGISTRY" in result
    assert "ACTION FIELD IDs:" in result
    assert "KRITISCHE REGELN:" in result
    assert "Mobilität und Verkehr" in result


# =============================================================================
# Tests for create_extraction_prompt()
# =============================================================================

def test_create_extraction_prompt_simplified():
    """Test create_extraction_prompt with simplified template."""
    sample_state = create_sample_state()
    chunk_text = "Test chunk text about mobility."
    page_numbers = [1, 2]

    result = create_extraction_prompt("simplified", chunk_text, sample_state, page_numbers)

    assert "Extrahieren Sie aus diesem Textabschnitt" in result
    assert "KONSISTENZ-ANWEISUNGEN:" in result
    assert chunk_text in result
    assert "Mobilität und Verkehr" in result  # Context should be included


def test_create_extraction_prompt_operations():
    """Test create_extraction_prompt with operations template."""
    sample_state = create_sample_state()
    chunk_text = "Test chunk text about climate protection."
    page_numbers = [3, 4, 5]

    result = create_extraction_prompt("operations", chunk_text, sample_state, page_numbers)

    assert "Analysieren Sie diesen Textabschnitt und erstellen Sie OPERATIONEN" in result
    assert "VERPFLICHTENDE PRÜFUNG vor jeder CREATE-Operation:" in result
    assert "Seiten 3, 4, 5" in result
    assert chunk_text in result
    assert "Mobilität und Verkehr" in result  # Context should be included


def test_create_extraction_prompt_operations_empty_context():
    """Test create_extraction_prompt with operations template and empty context."""
    chunk_text = "First chunk text."
    page_numbers = [1]

    result = create_extraction_prompt("operations", chunk_text, None, page_numbers)

    assert "ERSTER CHUNK: Noch keine Entities extrahiert" in result
    assert chunk_text in result


def test_create_extraction_prompt_invalid_template():
    """Test create_extraction_prompt with invalid template type."""
    sample_state = create_sample_state()

    try:
        create_extraction_prompt("invalid_template", "text", sample_state, [1])
        assert False, "Should have raised ValueError for invalid template"
    except ValueError as e:
        assert "Unknown template type: invalid_template" in str(e)


def test_create_extraction_prompt_no_page_numbers():
    """Test create_extraction_prompt without page numbers."""
    sample_state = create_sample_state()
    chunk_text = "Test chunk text."

    result = create_extraction_prompt("operations", chunk_text, sample_state, None)

    assert "Seiten N/A" in result or "N/A" in result


# =============================================================================
# Tests for validate_operations()
# =============================================================================

def test_validate_operations_valid():
    """Test validate_operations with valid operations."""
    operations = [
        EntityOperation(
            operation=OperationType.CREATE,
            entity_type="action_field",
            content={"title": "New Action Field"},
            confidence=0.9
        ),
        EntityOperation(
            operation=OperationType.UPDATE,
            entity_type="action_field",  # Required field
            entity_id="af_1",
            content={"description": "Updated description"},
            confidence=0.8
        )
    ]

    sample_state = create_sample_state()
    errors = validate_operations(operations, sample_state)

    assert errors == [], f"Expected no errors, got: {errors}"


def test_validate_operations_create_without_content():
    """Test validate_operations with CREATE operation missing content."""
    operations = [
        EntityOperation(
            operation=OperationType.CREATE,
            entity_type="action_field",
            content=None,  # Missing content
            confidence=0.9
        )
    ]

    errors = validate_operations(operations)

    assert len(errors) == 1
    assert "CREATE requires content" in errors[0]


def test_validate_operations_update_without_entity_id():
    """Test validate_operations with UPDATE operation missing entity_id."""
    operations = [
        EntityOperation(
            operation=OperationType.UPDATE,
            entity_type="action_field",
            entity_id=None,  # Missing entity_id
            content={"description": "Updated"},
            confidence=0.8
        )
    ]

    errors = validate_operations(operations)

    assert len(errors) == 1
    assert "UPDATE requires entity_id" in errors[0]


def test_validate_operations_connect_without_connections():
    """Test validate_operations with CONNECT operation missing connections."""
    operations = [
        EntityOperation(
            operation=OperationType.CONNECT,
            entity_type="action_field",
            entity_id="af_1",
            connections=None,  # Missing connections
            confidence=0.7
        )
    ]

    errors = validate_operations(operations)

    assert len(errors) == 1
    assert "CONNECT requires connections" in errors[0]


def test_validate_operations_nonexistent_entity():
    """Test validate_operations with UPDATE referencing nonexistent entity."""
    operations = [
        EntityOperation(
            operation=OperationType.UPDATE,
            entity_type="action_field",
            entity_id="nonexistent_id",
            content={"description": "Updated"},
            confidence=0.8
        )
    ]

    sample_state = create_sample_state()
    errors = validate_operations(operations, sample_state)

    assert len(errors) == 1
    assert "Entity nonexistent_id not found in current state" in errors[0]


def test_validate_operations_mixed():
    """Test validate_operations with mixed valid and invalid operations."""
    operations = [
        EntityOperation(
            operation=OperationType.CREATE,
            entity_type="action_field",
            content={"title": "Valid Create"},
            confidence=0.9
        ),
        EntityOperation(
            operation=OperationType.UPDATE,
            entity_type="action_field",
            entity_id=None,  # Invalid - missing entity_id
            content={"description": "Invalid Update"},
            confidence=0.8
        ),
        EntityOperation(
            operation=OperationType.CONNECT,
            entity_type="action_field",
            entity_id="af_1",
            connections=[{"from_id": "af_1", "to_id": "proj_1", "confidence": 0.8}],
            confidence=0.7
        )
    ]

    sample_state = create_sample_state()
    errors = validate_operations(operations, sample_state)

    assert len(errors) == 1
    assert "UPDATE requires entity_id" in errors[0]


# =============================================================================
# Tests for OperationExecutor.apply_operations()
# =============================================================================

def test_apply_operations_create():
    """Test apply_operations with CREATE operations."""
    executor = OperationExecutor()
    empty_state = create_empty_state()

    operations = [
        EntityOperation(
            operation=OperationType.CREATE,
            entity_type="action_field",
            content={"title": "New Action Field", "description": "Description"},
            confidence=0.9
        )
    ]

    new_state, log = executor.apply_operations(empty_state, operations)

    assert len(new_state.action_fields) == 1
    assert new_state.action_fields[0].content["title"] == "New Action Field"
    assert log.successful_operations == 1
    assert log.total_operations == 1


def test_apply_operations_update():
    """Test apply_operations with UPDATE operations."""
    executor = OperationExecutor()
    sample_state = create_sample_state()

    operations = [
        EntityOperation(
            operation=OperationType.UPDATE,
            entity_type="action_field",
            entity_id="af_1",
            content={"description": "Updated description"},
            confidence=0.8
        )
    ]

    new_state, log = executor.apply_operations(sample_state, operations)

    # Find the updated entity
    updated_af = next(af for af in new_state.action_fields if af.id == "af_1")
    assert "Updated description" in updated_af.content["description"]
    assert log.successful_operations == 1


def test_apply_operations_connect():
    """Test apply_operations with CONNECT operations."""
    executor = OperationExecutor()
    sample_state = create_sample_state()

    operations = [
        EntityOperation(
            operation=OperationType.CONNECT,
            entity_type="action_field",
            entity_id="af_1",
            connections=[
                {
                    "from_id": "af_1",
                    "to_id": "proj_1",
                    "confidence": 0.8
                }
            ],
            confidence=0.7
        )
    ]

    new_state, log = executor.apply_operations(sample_state, operations)

    # Find the connected entity
    connected_af = next(af for af in new_state.action_fields if af.id == "af_1")
    assert len(connected_af.connections) == 1
    assert connected_af.connections[0].target_id == "proj_1"
    assert log.successful_operations == 1


def test_apply_operations_invalid_entity_id():
    """Test apply_operations with invalid entity ID."""
    executor = OperationExecutor()
    sample_state = create_sample_state()

    operations = [
        EntityOperation(
            operation=OperationType.UPDATE,
            entity_type="action_field",
            entity_id="nonexistent_id",
            content={"description": "This should fail"},
            confidence=0.8
        )
    ]

    new_state, log = executor.apply_operations(sample_state, operations)

    # Should not modify state and should log failure
    assert new_state == sample_state  # State unchanged
    assert log.successful_operations == 0
    assert log.total_operations == 1


def test_apply_operations_empty_list():
    """Test apply_operations with empty operations list."""
    executor = OperationExecutor()
    sample_state = create_sample_state()

    operations = []

    new_state, log = executor.apply_operations(sample_state, operations)

    assert new_state == sample_state
    assert log.successful_operations == 0
    assert log.total_operations == 0


def test_apply_operations_update_merge_strings():
    """Test apply_operations UPDATE merges string fields correctly."""
    executor = OperationExecutor()
    sample_state = create_sample_state()

    operations = [
        EntityOperation(
            operation=OperationType.UPDATE,
            entity_type="action_field",
            entity_id="af_1",
            content={"description": "Additional description"},
            confidence=0.8
        )
    ]

    new_state, log = executor.apply_operations(sample_state, operations)

    updated_af = next(af for af in new_state.action_fields if af.id == "af_1")
    # Should merge, not replace
    assert "Nachhaltige Verkehrslösungen" in updated_af.content["description"]
    assert "Additional description" in updated_af.content["description"]
    # Should NOT have double periods
    assert ".." not in updated_af.content["description"]


def test_apply_operations_no_double_periods():
    """Test that merging descriptions with periods doesn't create double periods."""
    executor = OperationExecutor()
    
    # Create a project with a description ending in a period
    sample_state = EnrichedReviewJSON(
        action_fields=[],
        projects=[
            EnhancedProject(
                id="proj_1",
                content={"title": "Test Project", "description": "First description with period."},
                connections=[]
            )
        ],
        measures=[],
        indicators=[]
    )
    
    # Update with another description
    operations = [
        EntityOperation(
            operation=OperationType.UPDATE,
            entity_type="project",
            entity_id="proj_1",
            content={"description": "Second description part."},
            confidence=0.8
        )
    ]
    
    new_state, log = executor.apply_operations(sample_state, operations)
    
    updated_proj = next(p for p in new_state.projects if p.id == "proj_1")
    # Should have both parts
    assert "First description" in updated_proj.content["description"]
    assert "Second description" in updated_proj.content["description"]
    # Should NOT have double periods
    assert ".." not in updated_proj.content["description"]
    # Should be properly formatted
    assert updated_proj.content["description"] == "First description with period. Second description part."


# =============================================================================
# Test Runner
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    runner = _TestRunner()

    print("🧪 Running Critical Functions Test Suite")
    print("="*60)

    # Entity Registry Tests
    print("\n📋 Entity Registry Functions:")
    runner.run_test(test_format_entity_registry_empty, "format_entity_registry_empty")
    runner.run_test(test_format_entity_registry_sample, "format_entity_registry_sample")
    runner.run_test(test_format_entity_registry_large, "format_entity_registry_large")
    runner.run_test(test_format_entity_registry_unicode, "format_entity_registry_unicode")
    runner.run_test(test_format_entity_registry_edge_cases, "format_entity_registry_edge_cases")

    runner.run_test(test_format_entity_id_mapping_empty, "format_entity_id_mapping_empty")
    runner.run_test(test_format_entity_id_mapping_sample, "format_entity_id_mapping_sample")
    runner.run_test(test_format_entity_id_mapping_duplicates, "format_entity_id_mapping_duplicates")

    runner.run_test(test_format_context_json_empty, "format_context_json_empty")
    runner.run_test(test_format_context_json_sample, "format_context_json_sample")

    # Prompt Generation Tests
    print("\n📝 Prompt Generation Functions:")
    runner.run_test(test_create_extraction_prompt_simplified, "create_extraction_prompt_simplified")
    runner.run_test(test_create_extraction_prompt_operations, "create_extraction_prompt_operations")
    runner.run_test(test_create_extraction_prompt_operations_empty_context, "create_extraction_prompt_operations_empty_context")
    runner.run_test(test_create_extraction_prompt_invalid_template, "create_extraction_prompt_invalid_template")
    runner.run_test(test_create_extraction_prompt_no_page_numbers, "create_extraction_prompt_no_page_numbers")

    # Operations Processing Tests
    print("\n⚙️ Operations Processing Functions:")
    runner.run_test(test_validate_operations_valid, "validate_operations_valid")
    runner.run_test(test_validate_operations_create_without_content, "validate_operations_create_without_content")
    runner.run_test(test_validate_operations_update_without_entity_id, "validate_operations_update_without_entity_id")
    runner.run_test(test_validate_operations_connect_without_connections, "validate_operations_connect_without_connections")
    runner.run_test(test_validate_operations_nonexistent_entity, "validate_operations_nonexistent_entity")
    runner.run_test(test_validate_operations_mixed, "validate_operations_mixed")

    runner.run_test(test_apply_operations_create, "apply_operations_create")
    runner.run_test(test_apply_operations_update, "apply_operations_update")
    runner.run_test(test_apply_operations_connect, "apply_operations_connect")
    runner.run_test(test_apply_operations_invalid_entity_id, "apply_operations_invalid_entity_id")
    runner.run_test(test_apply_operations_empty_list, "apply_operations_empty_list")
    runner.run_test(test_apply_operations_update_merge_strings, "apply_operations_update_merge_strings")
    runner.run_test(test_apply_operations_no_double_periods, "apply_operations_no_double_periods")

    runner.summary()
    return runner


if __name__ == "__main__":
    runner = run_all_tests()

    # Exit with error code if tests failed
    sys.exit(1 if runner.failed > 0 else 0)
